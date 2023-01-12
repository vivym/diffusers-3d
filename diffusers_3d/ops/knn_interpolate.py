from typing import Tuple

import torch


def three_nn_interpolate(
    src_coords: torch.Tensor, src_features: torch.Tensor, tgt_coords: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    src_coords = src_coords.contiguous()
    src_features = src_features.contiguous()
    tgt_coords = tgt_coords.contiguous()

    # tgt_features, indices, weights
    return torch.ops.diffusers_3d.three_nn_interpolate(
        src_coords, src_features, tgt_coords
    )
