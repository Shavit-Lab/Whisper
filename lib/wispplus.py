import torch
import torch.nn as nn
import torch.nn.functional as F

from .wanda import Wanda


class WispPlus(Wanda):
    """Per-neuron saliency that preserves differentiation of similar inputs.

    For each of n_pairs random pairings of the token axis:
      - form the input differences diff = x_pi - x                  (columns, N)
      - push them through the layer to get the output differences   (N, rows)
      - rank tokens per neuron by |output diff| / ||input diff||
      - keep the top_frac fraction (K in the paper) of tokens for each neuron and
        average |diff| over them

    No centered or knn variant.
    """

    def __init__(self, layer, layer_id=0, layer_name="none", top_frac=0.005, n_pairs=1):
        super().__init__(layer, layer_id=layer_id, layer_name=layer_name)
        self.top_frac = top_frac
        self.n_pairs = n_pairs
        self.feature_diff = torch.zeros((self.rows, self.columns), device=self.dev,
                                        dtype=torch.float32)

    @torch.no_grad()
    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.feature_diff *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        w = self.layer.weight
        b = self.layer.bias if isinstance(self.layer, nn.Linear) else None
        feat = torch.zeros(self.rows, self.columns, device=self.dev, dtype=torch.float32)
        k = max(1, min(int(self.top_frac * inp.shape[1]), inp.shape[1]))

        for _ in range(self.n_pairs):
            perm = torch.randperm(inp.shape[1], device=inp.device)
            diff = inp.index_select(1, perm) - inp                                # (columns, N)

            diff_T = diff.t().contiguous()                                        # (N, columns)
            output_diff = torch.abs(F.linear(diff_T.to(w.dtype), w, b).float())   # (N, rows)

            input_diff = torch.linalg.vector_norm(diff_T, ord=2, dim=1)           # (N,)
            input_diff[input_diff == 0] = -1
            mapping_ratios = output_diff / input_diff.unsqueeze(1)                # (N, rows)
            _, top_indices = torch.topk(mapping_ratios, k, dim=0)                 # (k, rows)

            abs_diff_T = torch.abs(diff_T)                                        # (N, columns)
            for j in range(k):
                feat.add_(abs_diff_T[top_indices[j]])                             # (rows, columns)

        feat.div_(k * self.n_pairs)
        self.feature_diff += (feat / self.nsamples).to(self.feature_diff.dtype)

    def activation(self):
        """
        feature_diff is already a full (rows, columns) mean of absolute differences
        """
        return self.feature_diff
