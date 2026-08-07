import torch
import torch.nn as nn

# Wanda: https://github.com/locuslab/wanda
class Wanda:

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

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
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples

    def activation(self):
        """
        The activation statistic the saliency weighs |W| by. Allows easy override by subclasses.
        """
        return torch.sqrt(self.scaler_row.reshape((1, -1)))

    def saliency(self):
        """
        Per-weight pruning metric. Allows easy override by subclasses.
        """
        return torch.abs(self.layer.weight.data) * self.activation()