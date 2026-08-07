import torch
import torch.nn as nn

from .metrics import knn_index
from .wanda import Wanda


class Wisp(Wanda):
    """
    Wanda saliency that preserves differences in outputs.

    The difference norm can be formed three ways:

    "centered" (default determinsitic implementation, ignores n_pairs)
    "random" (stochastic implementation)
    "knn" (stochastic implementation with differences between similar inputs)
    """

    PAIRINGS = ("centered", "random", "knn")

    def __init__(self, layer, layer_id=0, layer_name="none", pairing="centered", n_pairs=1):
        super().__init__(layer, layer_id=layer_id, layer_name=layer_name)
        if pairing not in self.PAIRINGS:
            raise ValueError(f"pairing must be one of {self.PAIRINGS}, got {pairing!r}")
        self.pairing = pairing
        self.n_pairs = n_pairs

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.add_batch_diff(inp)

    def add_batch_diff(self, inp):
        if self.pairing == "centered":
            inp_c = inp - inp.mean(dim=1, keepdim=True)
            self.scaler_row += 2 * torch.norm(inp_c, p=2, dim=1) ** 2 / self.nsamples
            return

        if self.pairing == "knn":
            nn_idx = knn_index(inp, k=self.n_pairs)          # (N, n_pairs)
            partners = [nn_idx[:, j] for j in range(self.n_pairs)]
        else:
            partners = [torch.randperm(inp.shape[1], device=inp.device)
                        for _ in range(self.n_pairs)]

        for partner in partners:
            diff = inp.index_select(1, partner) - inp
            self.scaler_row += torch.norm(diff, p=2, dim=1) ** 2 / (self.n_pairs * self.nsamples)
