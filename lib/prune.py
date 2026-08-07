import torch
import torch.nn as nn
from tqdm import tqdm

from .wanda import Wanda
from .ria import RIA, RIAWisp, RIAWispPlus
from .wisp import Wisp
from .wispplus import WispPlus
from .sparsegpt import SparseGPT
from .whisper import Whisper
from .data import get_loaders

SEQUENTIAL_GROUPS = [
    ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
    ["self_attn.o_proj"],
    ["mlp.up_proj", "mlp.gate_proj"],
    ["mlp.down_proj"],
]


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def check_sparsity(model):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            zeros = W.numel() - W.count_nonzero().item()
            count += zeros
            total_params += W.numel()

            sub_count += zeros
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache
    return float(count)/total_params


def prepare_calibration_input(model, dataloader, device, nsamples, offload_to_cpu=False):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if offload_to_cpu:
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        if hasattr(model.model, 'rotary_emb') and model.model.rotary_emb is not None:
            model.model.rotary_emb = model.model.rotary_emb.to(device)
    elif "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None, "position_embeddings": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            # Modern transformers (>=4.45) implementation
            cache['position_embeddings'] = kwargs.get('position_embeddings')
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    if offload_to_cpu:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        if hasattr(model.model, 'rotary_emb') and model.model.rotary_emb is not None:
            model.model.rotary_emb = model.model.rotary_emb.cpu()
        torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids, position_embeddings


def move_calibration_state(dev, inps, outs, attention_mask, position_ids, position_embeddings):
    """
    Move the calibration tensors onto dev.

    Only needed for multi-GPU device maps.
    """
    inps, outs = inps.to(dev), outs.to(dev)
    if attention_mask is not None:
        attention_mask = attention_mask.to(dev)
    if position_ids is not None:
        position_ids = position_ids.to(dev)
    if position_embeddings is not None:
        position_embeddings = (position_embeddings[0].to(dev), position_embeddings[1].to(dev))
    return inps, outs, attention_mask, position_ids, position_embeddings


def apply_mask(module, W_metric, sparsity_ratio, prune_n=0, prune_m=0):
    """
    Zero the lowest-scoring weights of module according to W_metric.

    Unstructured pruning is per-neuron, following Wanda and RIA.
    """
    W_mask = (torch.zeros_like(W_metric) == 1)
    if prune_n != 0:
        for ii in range(W_metric.shape[1]):
            if ii % prune_m == 0:
                tmp = W_metric[:,ii:(ii+prune_m)].float()
                W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
    else:
        sort_res = torch.sort(W_metric, dim=-1, stable=True)
        indices = sort_res[1][:,:int(W_metric.shape[1]*sparsity_ratio)]
        W_mask.scatter_(1, indices, True)
        del sort_res, indices, W_metric

    module.weight.data[W_mask] = 0


@torch.no_grad()
def prune_mask(args, model, tokenizer, device, make_collector, prune_n=0, prune_m=0,
               starts_with=None, true_sequential=False, offload_to_cpu=False):
    """
    Driver for the saliency-mask methods: Wanda, Wisp, Wisp+, RIA, RIA-WISP, RIA-WISP+.

    Each collector accumulates a per-weight score during calibration; the lowest
    scoring weights are then zeroed. Weights are never otherwise modified.

    make_collector(name, module) picks the collector class for one layer, which
    is also where the wanda_fallback list is applied.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibration data")
    dataloader, _ = get_loaders("wikitext2",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    inps, outs, attention_mask, position_ids, position_embeddings = prepare_calibration_input(
        model, dataloader, device, args.nsamples, offload_to_cpu=offload_to_cpu)

    layers = model.model.layers
    for i in tqdm(range(len(layers)), desc="Pruning layers"):
        layer = layers[i]

        if offload_to_cpu:
            dev = device
            layer.to(dev)
        elif f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids, position_embeddings = move_calibration_state(
                dev, inps, outs, attention_mask, position_ids, position_embeddings)
        else:
            dev = device

        layer_kwargs = dict(attention_mask=attention_mask, position_ids=position_ids,
                            position_embeddings=position_embeddings)

        full = find_layers(layer)
        if starts_with is not None:
            full = {name: mod for name, mod in full.items() if any(name.startswith(s) for s in starts_with)}

        sequential = SEQUENTIAL_GROUPS if true_sequential else [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names if n in full}
            if not subset:
                continue

            collectors = {name: make_collector(name, subset[name]) for name in subset}

            def add_batch(name):
                def tmp(_, inp, out):
                    collectors[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in collectors:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
            for h in handles:
                h.remove()

            for name in subset:
                saliency_name = type(collectors[name]).__name__.lower()
                print(f"pruning layer {i} name {name} with {saliency_name} saliency")
                apply_mask(subset[name], collectors[name].saliency(),
                           args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m)
                torch.cuda.empty_cache()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
        inps, outs = outs, inps

        if offload_to_cpu:
            layer.to('cpu')
            torch.cuda.empty_cache()

        layers[i] = layer

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_gpt(args, model, tokenizer, device, make_collector, prune_n=0, prune_m=0,
              starts_with=None, true_sequential=False, offload_to_cpu=False):
    """
    Driver for the Hessian methods: SparseGPT, Whisper.

    Each collector accumulates a Hessian during calibration, then prunes and
    compensates the surviving weights itself inside fasterprune.

    make_collector(name, module) picks the collector class for one layer, which
    is also where the sparsegpt_fallback list is applied.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibration data")
    dataloader, _ = get_loaders("wikitext2",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    inps, outs, attention_mask, position_ids, position_embeddings = prepare_calibration_input(
        model, dataloader, device, args.nsamples, offload_to_cpu=offload_to_cpu)
    torch.cuda.empty_cache()

    print('Ready.')

    layers = model.model.layers
    for i in tqdm(range(len(layers)), desc="Pruning layers"):
        layer = layers[i]

        if offload_to_cpu:
            dev = device
            layer.to(dev)
        elif f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids, position_embeddings = move_calibration_state(
                dev, inps, outs, attention_mask, position_ids, position_embeddings)
        else:
            dev = device

        layer_kwargs = dict(attention_mask=attention_mask, position_ids=position_ids,
                            position_embeddings=position_embeddings)

        full = find_layers(layer)
        if starts_with is not None:
            full = {name: mod for name, mod in full.items() if any(name.startswith(s) for s in starts_with)}

        sequential = SEQUENTIAL_GROUPS if true_sequential else [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names if n in full}
            if not subset:
                continue

            collectors = {name: make_collector(name, subset[name]) for name in subset}

            def add_batch(name):
                def tmp(_, inp, out):
                    collectors[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in collectors:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
            for h in handles:
                h.remove()

            for name in subset:
                saliency_name = type(collectors[name]).__name__.lower()
                print(f"pruning layer {i} name {name} with {saliency_name} saliency")
                collectors[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m,
                                             percdamp=0.01, blocksize=128)
                collectors[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
        inps, outs = outs, inps

        torch.cuda.empty_cache()

        if offload_to_cpu:
            layer.to('cpu')
            torch.cuda.empty_cache()

        layers[i] = layer

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0,
                    starts_with=None, offload_to_cpu=False):
    """Magnitude pruning. Needs no calibration data, so it shares no driver."""
    print("Starting Magnitude ...")
    layers = model.model.layers

    for i in tqdm(range(len(layers)), desc="Pruning layers"):
        layer = layers[i]
        if offload_to_cpu:
            layer.to(device)

        subset = find_layers(layer)
        if starts_with is not None:
            subset = {name: mod for name, mod in subset.items() if any(name.startswith(s) for s in starts_with)}

        for name in subset:
            W = subset[name].weight.data
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten())[0][int(W.numel()*args.sparsity_ratio)]
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0

        if offload_to_cpu:
            layer.to('cpu')
            torch.cuda.empty_cache()

        layers[i] = layer
    torch.cuda.empty_cache()


def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0,
                starts_with=None, true_sequential=False, offload_to_cpu=False):
    print('Starting Wanda ...')
    prune_mask(args, model, tokenizer, device,
               lambda name, module: Wanda(module),
               prune_n=prune_n, prune_m=prune_m, starts_with=starts_with,
               true_sequential=true_sequential, offload_to_cpu=offload_to_cpu)


def prune_ria(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0,
              a=0.5, starts_with=None, true_sequential=False, offload_to_cpu=False):
    print('Starting RIA ...')
    prune_mask(args, model, tokenizer, device,
               lambda name, module: RIA(module, a=a),
               prune_n=prune_n, prune_m=prune_m, starts_with=starts_with,
               true_sequential=true_sequential, offload_to_cpu=offload_to_cpu)


def prune_wisp(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0,
               pairing="centered", n_pairs=1, starts_with=None, true_sequential=False,
               wanda_fallback=None, offload_to_cpu=False):
    print('Starting WISP ...')

    def make_collector(name, module):
        if wanda_fallback is not None and name in wanda_fallback:
            return Wanda(module)
        return Wisp(module, pairing=pairing, n_pairs=n_pairs)

    prune_mask(args, model, tokenizer, device, make_collector,
               prune_n=prune_n, prune_m=prune_m, starts_with=starts_with,
               true_sequential=true_sequential, offload_to_cpu=offload_to_cpu)


def prune_wispplus(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0,
                   top_frac=0.005, n_pairs=1, starts_with=None, true_sequential=False,
                   wanda_fallback=None, offload_to_cpu=False):
    print('Starting WISP+ ...')

    def make_collector(name, module):
        if wanda_fallback is not None and name in wanda_fallback:
            return Wanda(module)
        return WispPlus(module, top_frac=top_frac, n_pairs=n_pairs)

    prune_mask(args, model, tokenizer, device, make_collector,
               prune_n=prune_n, prune_m=prune_m, starts_with=starts_with,
               true_sequential=true_sequential, offload_to_cpu=offload_to_cpu)


def prune_ria_wisp(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0,
                   a=0.5, pairing="centered", n_pairs=1, starts_with=None, true_sequential=False,
                   ria_fallback=None, offload_to_cpu=False):
    """
    Wisp's activation differences scored with the RIA weight term.

    The fallback layers get plain RIA, the same way Wisp's fall back to Wanda.
    """
    print('Starting RIA-WISP ...')

    def make_collector(name, module):
        if ria_fallback is not None and name in ria_fallback:
            return RIA(module, a=a)
        return RIAWisp(module, a=a, pairing=pairing, n_pairs=n_pairs)

    prune_mask(args, model, tokenizer, device, make_collector,
               prune_n=prune_n, prune_m=prune_m, starts_with=starts_with,
               true_sequential=true_sequential, offload_to_cpu=offload_to_cpu)


def prune_ria_wispplus(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0,
                       a=0.5, top_frac=0.005, n_pairs=1, starts_with=None, true_sequential=False,
                       ria_fallback=None, offload_to_cpu=False):
    """
    Wisp+'s per-neuron activation differences scored with the RIA weight term.
    """
    print('Starting RIA-WISP+ ...')

    def make_collector(name, module):
        if ria_fallback is not None and name in ria_fallback:
            return RIA(module, a=a)
        return RIAWispPlus(module, a=a, top_frac=top_frac, n_pairs=n_pairs)

    prune_mask(args, model, tokenizer, device, make_collector,
               prune_n=prune_n, prune_m=prune_m, starts_with=starts_with,
               true_sequential=true_sequential, offload_to_cpu=offload_to_cpu)


def prune_sparsegpt(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0,
                    starts_with=None, true_sequential=False, offload_to_cpu=False):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting SparseGPT ...')
    prune_gpt(args, model, tokenizer, device,
              lambda name, module: SparseGPT(module),
              prune_n=prune_n, prune_m=prune_m, starts_with=starts_with,
              true_sequential=true_sequential, offload_to_cpu=offload_to_cpu)


def prune_whisper(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0,
                  gamma=0.01, pairing="centered", n_pairs=1, starts_with=None,
                  true_sequential=False, sparsegpt_fallback=None, offload_to_cpu=False):
    print('Starting WHISPER ...')
    print(f"Regularization factor: {gamma}")

    def make_collector(name, module):
        if sparsegpt_fallback is not None and name in sparsegpt_fallback:
            return SparseGPT(module)
        return Whisper(module, gamma=gamma, pairing=pairing, n_pairs=n_pairs)

    prune_gpt(args, model, tokenizer, device, make_collector,
              prune_n=prune_n, prune_m=prune_m, starts_with=starts_with,
              true_sequential=true_sequential, offload_to_cpu=offload_to_cpu)
