import torch
import torch.nn as nn


class SE3D(nn.Module):
    def __init__(self, num_channels: int, reduction: int = 8):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction, num_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.fc(x.flatten(2).mean(-1))[:, :, None, None, None]
        return x * weights
