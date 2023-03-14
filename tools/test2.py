import torch
import spconv.pytorch as spconv
from diffusers_3d.ops.trilinear_devoxelize import trilinear_devoxelize
from sparse_ops.devoxelize import trilinear_devoxelize


def main():
    from tqdm import tqdm
    for i in tqdm(range(10000)):
        (
            coords, voxel_size, points_range_min, points_range_max,
            sp_tensor_indices, sp_tensor_features,
        ) = torch.load(f"test{i % 4:03d}.pth")

        # print("coords", coords.min(), coords.max())
        # print("voxel_size", voxel_size)
        # print("points_range_min", points_range_min)
        # print("points_range_max", points_range_max)
        # print("sp_tensor_indices", sp_tensor_indices.min(), sp_tensor_indices.max())
        # print("sp_tensor_features", sp_tensor_features.min(), sp_tensor_features.max())

        voxel_features, *_ = trilinear_devoxelize(
            coords,
            voxel_size=voxel_size,
            points_range_min=points_range_min,
            points_range_max=points_range_max,
            voxel_coords=sp_tensor_indices[:, 1:].long(),
            voxel_features=sp_tensor_features,
            voxel_batch_indices=sp_tensor_indices[:, 0].long(),
        )
        print("voxel_features", voxel_features.mean())


if __name__ == "__main__":
    main()
