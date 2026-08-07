import fnmatch
import json
import os

import torch
import torch.nn as nn

from .data import get_loaders

# Open LLM Leaderboard v1, with the number of few-shot examples each task uses.
# NOTE: the official v1 suite uses gsm8k 5-shot. We substitute mathqa because
# gsm8k scores sit near the floor for every pruning method compared here, so it
# does not separate them.
LEADERBOARD_V1_TASKS = [
    ("arc_challenge", 25),
    ("truthfulqa_mc", 0),
    ("winogrande", 5),
    ("mathqa", 0),
    ("hellaswag", 10),
    ("mmlu", 5),
]

# Friendly names -> lm-eval-harness task patterns
TASK_ALIASES = {
    "mmlu": "hendrycksTest-*",
}


def eval_ppl(model, tokenizer, device=torch.device("cuda:0"), layerwise=False):
    """WikiText2 perplexity. Set layerwise=True when the model is offloaded to CPU."""
    dataset = "wikitext2"
    print(f"evaluating on {dataset}")

    _, testloader = get_loaders(dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer)

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        if layerwise:
            ppl_test = eval_ppl_wikitext_layerwise(model, testloader, 1, device)
        else:
            ppl_test = eval_ppl_wikitext(model, testloader, 1, device)
    return ppl_test


def eval_ppl_wikitext(model, testenc, bs=1, device=None):
    testenc = testenc.input_ids

    nsamples = testenc.numel() // model.seqlen

    nlls = []
    print(f"nsamples {nsamples}")

    for i in range(0, nsamples, bs):
        if i % 10 == 0:
            print(f"sample {i}")

        j = min(i+bs, nsamples)

        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        neg_log_likelihood = loss.float() * model.seqlen * (j-i)
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    torch.cuda.empty_cache()

    return ppl.item()


# Layer-by-layer offloaded variant of eval_ppl_wikitext.
@torch.no_grad()
def eval_ppl_wikitext_layerwise(model, testenc, bs=1, device=None):
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen
    print(f"nsamples {nsamples}")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(device)
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device
    )
    # Modern transformers (>=4.45) implementation
    cache = {"i": 0, "kwargs": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["kwargs"] = {
                k: v for k, v in kwargs.items()
                if k not in ("past_key_value", "use_cache")
            }
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(device)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    layer_kwargs = cache["kwargs"]

    for i in range(len(layers)):
        if i % 10 == 0:
            print(f"layer {i}")
        layer = layers[i].to(device)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(device)
    model.lm_head = model.lm_head.to(device)

    testenc = testenc.to(device)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    return ppl.item()


def eval_zero_shot(model_name, model, tokenizer,
                   task_list=["boolq","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"],
                   num_fewshot=0, add_special_tokens=False):
    """
    Evaluate an already-pruned in-memory model with the LM Harness.

    Requires the modified EleutherAI lm-evaluation-harness distributed with Wanda
    (see the README): it adds `pretrained_model` and `tokenizer` arguments to
    simple_evaluate so an in-memory pruned model can be evaluated directly rather
    than reloaded from a checkpoint path. Stock lm_eval does not accept those.
    """
    from lm_eval import tasks, evaluator

    def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)

    task_names = pattern_match(task_list, tasks.ALL_TASKS)

    model_args = f"pretrained={model_name},use_accelerate=True"

    results = evaluator.simple_evaluate(
        model="hf-causal-experimental",
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=1,
        device=None,
        no_cache=True,
        limit=None,
        description_dict={},
        decontamination_ngrams_path=None,
        check_integrity=False,
        pretrained_model=model,
        tokenizer=tokenizer,
        add_special_tokens=add_special_tokens
    )

    return results


def run_evals(args, model, tokenizer):
    """
    Run every harness task requested on the command line.
    """
    if args.eval and (args.num_fewshot is None or len(args.num_fewshot) != len(args.eval)):
        raise ValueError("--num_fewshot must be provided and match the length of --eval")

    task_names = list(args.eval or [])
    task_shots = list(args.num_fewshot or [])

    if args.eval_llm_leaderboard_v1:
        for task, shots in LEADERBOARD_V1_TASKS:
            task_names.append(task)
            task_shots.append(shots)

    if not task_names:
        return

    if args.offload_to_cpu:
        # Pruning ran with the model on CPU; spread it back over the GPUs to evaluate.
        from accelerate import dispatch_model, infer_auto_device_map
        from accelerate.utils import get_balanced_memory
        max_memory = get_balanced_memory(
            model,
            dtype=model.dtype,
            no_split_module_classes=model._no_split_modules,
            low_zero=False,
        )
        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=model._no_split_modules,
            dtype=model.dtype,
        )
        model = dispatch_model(model, device_map=device_map)
    print(model.hf_device_map)

    for name, num_shot in zip(task_names, task_shots):
        task_pattern = TASK_ALIASES.get(name, name)
        print("*" * 32)
        print(f"Evaluating {name} with {num_shot}-shot")

        save_filepath = None
        if args.save_eval:
            os.makedirs(args.save_eval, exist_ok=True)
            save_filepath = os.path.join(
                args.save_eval, f"log_{args.prune_method}_{name}_seed_{args.seed}.json")
            # Already evaluated on a previous run; reuse it rather than redo it.
            if os.path.exists(save_filepath):
                with open(save_filepath) as f:
                    print(json.load(f))
                continue

        results = eval_zero_shot(args.model, model, tokenizer, [task_pattern], num_shot)
        print(results)

        if save_filepath:
            with open(save_filepath, "w") as f:
                json.dump(results, f, indent=2, default=float)
