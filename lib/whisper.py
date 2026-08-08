import math

import torch
import torch.nn as nn
import transformers

from .metrics import check_pairing, diff_gram
from .sparsegpt import SparseGPT

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class Whisper(SparseGPT):
    """
    Pruning using a lightly regularized Hessian of differences.

    Accumulate a second Hessian H_diff built from differences between token activations.
    Prune using a blend of the two: gamma * H + (1 - gamma) * H_diff.

    The difference Hessian can be formed three ways:

    "centered" (default determinsitic implementation, ignores n_pairs)
    "random" (stochastic implementation)
    "knn" (stochastic implementation with differences between similar inputs)
    """

    def __init__(self, layer, gamma=0.01, pairing="centered", n_pairs=1):
        super().__init__(layer)
        self.H_diff = torch.zeros((self.columns, self.columns), device=self.dev)
        self.gamma = gamma
        self.pairing = check_pairing(pairing)
        self.n_pairs = n_pairs

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.H_diff *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
        self.H_diff += diff_gram(inp, self.pairing, self.n_pairs)

    def fasterprune(self, *args, **kwargs):
        diff_scale = torch.mean(torch.diag(self.H_diff))
        scale_ratio = torch.mean(torch.diag(self.H)) / diff_scale
        if diff_scale > 0 and torch.isfinite(scale_ratio):
            self.H = self.gamma * self.H + (1 - self.gamma) * self.H_diff * scale_ratio
        else:
            # Guard against empty difference Hessian, never observed in practice.
            print("Difference Hessian is empty, falling back to SparseGPT for this layer")
        self.H_diff = None
        super().fasterprune(*args, **kwargs)

    def free(self):
        self.H_diff = None
        super().free()
