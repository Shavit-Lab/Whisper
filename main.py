import argparse
import gc
import os
import time
import numpy as np
import torch
from importlib.metadata import version
from transformers import AutoTokenizer, AutoModelForCausalLM

from lib.prune import (check_sparsity, prune_magnitude, prune_ria, prune_ria_wisp,
                       prune_ria_wispplus, prune_sparsegpt, prune_wanda, prune_whisper,
                       prune_wisp, prune_wispplus)
from lib.eval import eval_ppl, run_evals

# Layers pruned with general activation baselines by default.
# Difference-informed criteria applied only to remaining layers:
# mlp.up_proj, mlp.gate_proj. Pass the flag with no values to disable.
DEFAULT_FALLBACK = ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj",
                    "self_attn.o_proj", "mlp.down_proj"]

# Ablation knobs. Passing any of them requires --testing.
TESTING_ONLY_ARGS = ["n_pairs", "gamma", "top_frac", "ria_a", "starts_with",
                     "wanda_fallback", "sparsegpt_fallback", "true_sequential"]

IMPLEMENTED_METHODS = ["magnitude", "wanda", "ria", "wisp", "wispplus", "ria-wisp",
                       "ria-wispplus", "sparsegpt", "whisper"]
# Selectable so the interface is stable, but not wired up yet.
UNIMPLEMENTED_METHODS = ["alps", "alps-whisper"]

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())


def get_llm_from_path(model_path, offload_to_cpu=False):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        device_map="cpu" if offload_to_cpu else "auto"
    )
    model.seqlen = min(model.config.max_position_embeddings, 4096)
    return model


def explicitly_passed(parser, argv=None):
    """
    Returns the set of dests that actually appeared on the command line.
    Used to check for ablation knobs that require --testing.
    """
    saved = [(action, action.default) for action in parser._actions]
    try:
        for action, _ in saved:
            action.default = argparse.SUPPRESS
        return set(vars(parser.parse_args(argv)))
    finally:
        for action, default in saved:
            action.default = default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to the model')
    parser.add_argument('--prune_method', type=str, required=True, choices=IMPLEMENTED_METHODS + UNIMPLEMENTED_METHODS, help='Pruning method to use.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level. 0 leaves the model dense and only evaluates it.')
    parser.add_argument("--sparsity_type", type=str, default="unstructured", help='"unstructured", or "n:m" for structured n:m sparsity (e.g. 2:4).')
    parser.add_argument("--pairing", type=str, default="centered", choices=["centered", "random", "knn"], help="How Wisp/Whisper form activation differences: the exact centered form, random token pairings, or nearest-neighbour pairings.")
    parser.add_argument('--save', type=str, default=None, help='Path to save perplexity results.')
    parser.add_argument("--save_eval", type=str, default=None, help="Path to save evaluation results.")
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--offload_to_cpu', action="store_true", help="Load model on CPU and offload layers to GPU one at a time during pruning. Use for large models that don't fit in GPU memory.")
    parser.add_argument("--eval", type=str, nargs='+', default=None, help="List of evaluation tasks")
    parser.add_argument("--num_fewshot", type=int, nargs='+', default=None, help="Number of few-shot examples for each eval task (must match length of --eval)")
    parser.add_argument("--eval_llm_leaderboard_v1", action="store_true", help="Evaluate on Open LLM Leaderboard v1 tasks (arc_challenge 25-shot, hellaswag 10-shot, mmlu 5-shot, truthfulqa_mc 0-shot, winogrande 5-shot, mathqa 0-shot; mathqa replaces the suite's gsm8k, which generally performs poorly across methods)")

    parser.add_argument("--testing", action="store_true", help="Enable testing mode")
    parser.add_argument('--n_pairs', type=int, default=1, help="Token pairings drawn per calibration sample for the difference-informed metric. Ignored when --pairing is centered; the neighbour count k when it is knn.")
    parser.add_argument('--gamma', type=float, default=0.01, help="Amount of regularization to add from the normal Hessian to the difference Hessian (Whisper).")
    parser.add_argument('--top_frac', type=float, default=0.005, help="Proportion of top pairs to consider for WispPlus saliency (K).")
    parser.add_argument('--ria_a', type=float, default=0.5, help="Exponent on the activation term of the RIA metric.")
    parser.add_argument("--starts_with", type=str, nargs='+', help="Prune only layers that start with any of the given strings.")
    parser.add_argument("--wanda_fallback", type=str, nargs='*', default=DEFAULT_FALLBACK, help=f"Layer names to prune with the baseline saliency instead of difference-informed saliency: Wanda for wisp/wispplus, RIA for the ria- variants (default: {' '.join(DEFAULT_FALLBACK)}; pass the flag with no values to disable)")
    parser.add_argument("--sparsegpt_fallback", type=str, nargs='*', default=DEFAULT_FALLBACK, help=f"Layer names to prune with SparseGPT geometry instead of difference-informed saliency (default: {' '.join(DEFAULT_FALLBACK)}; pass the flag with no values to disable)")
    parser.add_argument("--true_sequential", action="store_true", help="Prune the model in a true sequential manner")

    args = parser.parse_args()

    if not args.testing:
        passed = explicitly_passed(parser)
        offenders = [dest for dest in TESTING_ONLY_ARGS if dest in passed]
        if offenders:
            verb = "requires" if len(offenders) == 1 else "require"
            parser.error(f"{', '.join('--' + dest for dest in offenders)} {verb} --testing")

    if args.prune_method in UNIMPLEMENTED_METHODS:
        parser.error(f"--prune_method {args.prune_method} is not implemented yet")

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    print(f"Loading llm model from {args.model}")
    model = get_llm_from_path(args.model, offload_to_cpu=args.offload_to_cpu)
    model.eval()
    # For certain models, the fast tokenizer is necessary.
    use_fast_list = ["granite"]
    use_fast = any(item in args.model.lower() for item in use_fast_list)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=use_fast)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    device = torch.device("cuda:0")
    print("use device ", device)
    print("model dtype ", model.dtype)

    pruning_start_time = time.time()

    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m,
                            starts_with=args.starts_with, offload_to_cpu=args.offload_to_cpu)
        elif args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m,
                        starts_with=args.starts_with, true_sequential=args.true_sequential,
                        offload_to_cpu=args.offload_to_cpu)
        elif args.prune_method == "ria":
            prune_ria(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m,
                      a=args.ria_a, starts_with=args.starts_with,
                      true_sequential=args.true_sequential, offload_to_cpu=args.offload_to_cpu)
        elif args.prune_method == "wisp":
            prune_wisp(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m,
                       pairing=args.pairing, n_pairs=args.n_pairs, starts_with=args.starts_with,
                       true_sequential=args.true_sequential, wanda_fallback=args.wanda_fallback,
                       offload_to_cpu=args.offload_to_cpu)
        elif args.prune_method == "wispplus":
            prune_wispplus(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m,
                           top_frac=args.top_frac, n_pairs=args.n_pairs, starts_with=args.starts_with,
                           true_sequential=args.true_sequential, wanda_fallback=args.wanda_fallback,
                           offload_to_cpu=args.offload_to_cpu)
        elif args.prune_method == "ria-wisp":
            prune_ria_wisp(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m,
                           a=args.ria_a, pairing=args.pairing, n_pairs=args.n_pairs,
                           starts_with=args.starts_with, true_sequential=args.true_sequential,
                           ria_fallback=args.wanda_fallback, offload_to_cpu=args.offload_to_cpu)
        elif args.prune_method == "ria-wispplus":
            prune_ria_wispplus(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m,
                               a=args.ria_a, top_frac=args.top_frac, n_pairs=args.n_pairs,
                               starts_with=args.starts_with, true_sequential=args.true_sequential,
                               ria_fallback=args.wanda_fallback, offload_to_cpu=args.offload_to_cpu)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m,
                            starts_with=args.starts_with, true_sequential=args.true_sequential,
                            offload_to_cpu=args.offload_to_cpu)
        elif args.prune_method == "whisper":
            prune_whisper(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m,
                          gamma=args.gamma, pairing=args.pairing, n_pairs=args.n_pairs,
                          starts_with=args.starts_with, true_sequential=args.true_sequential,
                          sparsegpt_fallback=args.sparsegpt_fallback,
                          offload_to_cpu=args.offload_to_cpu)
    else:
        print("sparsity_ratio is 0, skipping pruning and evaluating the dense model")

    pruning_end_time = time.time()
    pruning_time = pruning_end_time - pruning_start_time
    print(f"Pruning time: {pruning_time:.2f} seconds")

    gc.collect()
    torch.cuda.empty_cache()

    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)

    ppl_test = eval_ppl(model, tokenizer, device, layerwise=args.offload_to_cpu)
    print(f"wikitext perplexity {ppl_test}")

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        save_filepath = os.path.join(args.save, f"log_{args.prune_method}_seed_{args.seed}.txt")
        with open(save_filepath, "w") as f:
            print("method\tactual_sparsity\tppl_test\tpruning_time", file=f, flush=True)
            print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}\t{pruning_time:.2f}", file=f, flush=True)
       
    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

    run_evals(args, model, tokenizer)


if __name__ == '__main__':
    main()
