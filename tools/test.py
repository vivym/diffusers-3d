import torch
import torch.nn as nn

from diffusers_3d.layers.modules.voxelization import Voxelization
from diffusers_3d.layers.se import SE3D
from diffusers_3d.layers.voxelization import Voxelizer
from diffusers_3d.structures.points import PointTensor


def main():
    in_channels = 32
    out_channels = 32
    kernel_size = 3
    dropout_prob = 0.1

    voxel_layers = [
        nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        ),
        nn.GroupNorm(num_groups=8, num_channels=out_channels),
        nn.SiLU(inplace=True),
    ]

    if dropout_prob > 0:
        voxel_layers.append(nn.Identity())

    voxel_layers += [
        nn.Conv3d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        ),
        nn.GroupNorm(num_groups=8, num_channels=out_channels),
        nn.SiLU(inplace=True),
    ]

    voxel_layers.append(SE3D(out_channels))

    voxel_layers = nn.Sequential(*voxel_layers)
    voxel_layers.eval()
    voxel_layers.cuda()

    state_dict = torch.load("../PVD/PVD/output/train_generation_pl_no_noise_init/2023-01-15-18-10-08/epoch_1799.pth", map_location="cpu")
    print("state_dict", list(state_dict.keys()))
    new_state_dict = {}
    for k, v in state_dict["model_state"].items():
        # k = k[len("model.module."):]
        # new_state_dict[k] = v
        if k.startswith("model.module.down_blocks.0.1.voxel_layers."):
            k = k.replace("model.module.down_blocks.0.1.voxel_layers.", "")
            new_state_dict[k] = v

    voxel_layers.load_state_dict(new_state_dict)

    # voxelizer = Voxelization(16, normalize=True, eps=0.)
    voxelizer = Voxelizer(32, normalize=True, eps=0.)

    (
        _points_features, _points_coords, _voxel_features, _voxel_coords
    ) = torch.load("../PVD/PVD/voxelizer1.pth")

    points = PointTensor(
        coords=_points_coords,
        features=_points_features,
        is_channel_last=False,
    )
    voxels: PointTensor = voxelizer(points)

    print("_voxel_features", torch.allclose(_voxel_features, voxels.features), (_voxel_features - voxels.features).abs().max())
    print("_voxel_coords", torch.allclose(_voxel_coords, voxels.coords), (_voxel_coords - voxels.coords).abs().max())

    with torch.no_grad():
        voxel_features = voxel_layers[0](voxels.features)

    (
        _tmp2, _tmp, _voxels_coords, _voxel_features
    ) = torch.load("../PVD/PVD/devoxelize1.pth")

    print("_voxel_features", _tmp.shape, voxel_features.shape)

    print("_voxel_features", torch.allclose(_tmp, voxel_features), (_tmp - voxel_features).abs().max(), (_tmp == voxel_features).all())
    # print("_tmp", _tmp)
    print("voxels.features", voxels.features.mean(), voxels.features.min(), voxels.features.max())
    # print("voxels.features", voxels.features)


if __name__ == "__main__":
    main()
