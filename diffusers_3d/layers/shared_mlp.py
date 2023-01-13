from typing import List

import torch
import torch.nn as nn


class Swish(nn.Module):
    def forward(self,x):
        return  x * torch.sigmoid(x)


class SharedMLP(nn.Module):
    def __init__(self, num_channels: List[int], dim: int = 1):
        super().__init__()

        if dim == 1:
            conv_fn = nn.Conv1d
        elif dim == 2:
            conv_fn = nn.Conv2d
        else:
            raise ValueError(dim)

        layers = []
        for in_channels, out_channels in zip(num_channels, num_channels[1:]):
            layers += [
                conv_fn(in_channels, out_channels, kernel_size=1),
                nn.GroupNorm(8, out_channels),
                # nn.SiLU(inplace=True),
                Swish(),
            ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)
