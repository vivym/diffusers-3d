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


def main():
    coords, features, resolution, outs, inds, wgts = torch.load(
        "../PVD/trilinear_devoxelize.pth", map_location="cuda"
    )

    coords = coords.permute(0, 2, 1)

    features = features.view(*features.shape[:2], resolution, resolution, resolution)

    point_features, indices, weights = trilinear_devoxelize(
        coords, features, resolution
    )

    indices = indices.permute(0, 2, 1)
    weights = weights.permute(0, 2, 1)

    print((point_features == outs).all())
    print((indices == inds).all())
    print((wgts == weights).all())


if __name__ == "__main__":
    main()
