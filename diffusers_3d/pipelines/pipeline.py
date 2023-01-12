import torch
from torch import nn


class DiffusionPipeline(nn.Module):
    @property
    def device(self) -> torch.device:
        param = next(self.parameters())
        return param.device
