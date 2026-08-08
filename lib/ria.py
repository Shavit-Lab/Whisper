import torch

from .wanda import Wanda
from .wisp import Wisp
from .wispplus import WispPlus

## RIA: https://github.com/biomedical-cybernetics/Relative-importance-and-activation-pruning
class RelativeImportance:
    """
    The RIA weight term performs both row and column normalization.
    """

    def __init__(self, *args, a=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.a = a

    def saliency(self):
        W = torch.abs(self.layer.weight.data)
        relative_importance = W / W.sum(dim=0) + W / W.sum(dim=1, keepdim=True)
        return relative_importance * self.activation() ** self.a


class RIA(RelativeImportance, Wanda):
    """RIA over Wanda's activation norms."""


class RIAWisp(RelativeImportance, Wisp):
    """RIA over Wisp's activation differences."""


class RIAWispPlus(RelativeImportance, WispPlus):
    """RIA over Wisp+'s per-neuron activation differences."""
