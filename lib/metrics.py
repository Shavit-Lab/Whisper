import torch

# How the difference-informed methods form activation differences.
PAIRINGS = ("centered", "random", "knn")


def check_pairing(pairing):
    if pairing not in PAIRINGS:
        raise ValueError(f"pairing must be one of {PAIRINGS}, got {pairing!r}")
    return pairing


def knn_index(inp, k=1, exclude_zero=True, band=0):
    """inp: (D, N) activations, columns are tokens.
    Returns (N, k) long tensor: the k nearest other tokens for each token."""
    X = inp.t()                                    # (N, D)
    sq = (X * X).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.t())
    n = X.shape[0]
    ar = torch.arange(n, device=X.device)
    d2[ar, ar] = float("inf")                      # no self-pairing
    if exclude_zero:
        d2[d2 <= 1e-12] = float("inf")             # duplicate tokens
    if band:                                       # skip close tokens
        m = (ar[:, None] - ar[None, :]).abs() <= band
        d2[m] = float("inf")
    return d2.topk(k, dim=1, largest=False).indices


def diff_gram(inp, pairing="centered", n_pairs=1):
    """inp: (D, N) activations, columns are tokens.
    Returns the (D, D) Gram matrix (Hessian) of activation differences.

    "centered" is the exact deterministic form and ignores n_pairs. 
    "random" draws n_pairs random token pairings.
    "knn" pairs each token with its k = n_pairs nearest neighbours.
    """
    if pairing == "centered":
        inp_c = inp - inp.mean(dim=1, keepdim=True)
        return 2 * inp_c.matmul(inp_c.t())

    if pairing == "knn":
        nn_idx = knn_index(inp, k=n_pairs)          # (N, n_pairs)
        partners = [nn_idx[:, j] for j in range(n_pairs)]
    else:
        partners = [torch.randperm(inp.shape[1], device=inp.device)
                    for _ in range(n_pairs)]

    gram = torch.zeros((inp.shape[0], inp.shape[0]), device=inp.device, dtype=inp.dtype)
    for partner in partners:
        diff = inp.index_select(1, partner) - inp
        gram += diff.matmul(diff.t()) / n_pairs
    return gram
