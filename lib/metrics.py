import torch


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
