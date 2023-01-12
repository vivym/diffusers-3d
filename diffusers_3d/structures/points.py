from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class PointTensor:
    coords: torch.Tensor    # B, N, 3
    features: torch.Tensor  # B, N, C if is_channel_last else B, C, N
    is_channel_last: bool = True
    timesteps: Optional[torch.Tensor] = None    # B
    t_embed: Optional[torch.Tensor] = None

    def clone(self) -> "PointTensor":
        return PointTensor(
            coords=self.coords,
            features=self.features,
            is_channel_last=self.is_channel_last,
            timesteps=self.timesteps,
            t_embed=self.t_embed,
        )
