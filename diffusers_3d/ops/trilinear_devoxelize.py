from typing import Tuple

import torch


def trilinear_devoxelize(
    coords: torch.Tensor, voxel_features: torch.Tensor, voxel_resolution: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coords = coords.contiguous()
    voxel_features = voxel_features.contiguous()

    # point_features, indices, weights
    return torch.ops.diffusers_3d.trilinear_devoxelize(
        coords, voxel_features, voxel_resolution
    )
