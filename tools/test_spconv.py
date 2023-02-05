import torch
from spconv.pytorch.utils import PointToVoxel, gather_features_by_pc_voxel_id


def main():
    device = torch.device("cuda")

    voxelizer = PointToVoxel(
        vsize_xyz=(1 / 32, 1 / 32, 1 / 32),
        coors_range_xyz=(0, 0, 0, 1, 1, 1),
        num_point_features=3,
        max_num_voxels=10000,
        max_num_points_per_voxel=32,
        device=device,
    )
    pcs = torch.rand(1000, 3, device=device)
    voxels, coords, num_points_per_voxel, pc_voxel_id = voxelizer.generate_voxel_with_id(pcs)

    print(voxels.shape)
    print(coords.shape)
    print(num_points_per_voxel.shape)
    print(pc_voxel_id.shape)


if __name__ == "__main__":
    main()
