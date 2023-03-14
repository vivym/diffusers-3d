import pytorch_lightning as pl
import torch
import torch.nn as nn

from diffusers_3d.data.datasets import ShapeNetCoreV2PC15KDataset
from sparse_ops.voxelize import voxelize
from sparse_ops.devoxelize import trilinear_devoxelize


def main():
    (
        pcs, features, voxel_size, points_range_min, points_range_max
    ) = torch.load("tmp.pth")

    voxels = voxelize(
        point_coords=pcs.contiguous(),
        point_features=features.contiguous(),
        voxel_size=voxel_size.contiguous(),
        points_range_min=points_range_min.contiguous(),
        points_range_max=points_range_max.contiguous(),
    )


if __name__ == "__main__":
    main()
