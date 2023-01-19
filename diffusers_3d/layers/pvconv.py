import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers_3d.structures.points import PointTensor
# from diffusers_3d.ops.trilinear_devoxelize import trilinear_devoxelize

from .voxelization import Voxelizer
from .se import SE3D
from .shared_mlp import SharedMLP

from .modules.voxelization import Voxelization
from .modules.functional.devoxelization import trilinear_devoxelize

tmp_index = 0
tmp_index2 = 0


class Attention(nn.Module):
    def __init__(self, num_channels: int, num_groups: int, dim: int = 3):
        super().__init__()

        assert num_channels % num_groups == 0

        if dim == 3:
            self.q_proj = nn.Conv3d(num_channels, num_channels, kernel_size=1)
            self.k_proj = nn.Conv3d(num_channels, num_channels, kernel_size=1)
            self.v_proj = nn.Conv3d(num_channels, num_channels, kernel_size=1)

            self.out_proj = nn.Conv3d(num_channels, num_channels, kernel_size=1)
        elif dim == 1:
            self.q_proj = nn.Conv1d(num_channels, num_channels, kernel_size=1)
            self.k_proj = nn.Conv1d(num_channels, num_channels, kernel_size=1)
            self.v_proj = nn.Conv1d(num_channels, num_channels, kernel_size=1)

            self.out_proj = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        else:
            raise ValueError(dim)

        self.norm = nn.GroupNorm(
            num_groups=num_groups, num_channels=num_channels
        )
        self.act_fn = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor):
        q = self.q_proj(x).flatten(2)
        k = self.k_proj(x).flatten(2)
        v = self.v_proj(x).flatten(2)

        qk = q.permute(0, 2, 1) @ k

        w = F.softmax(qk, dim=-1)

        h = (v @ w.permute(0, 2, 1)).reshape_as(x)
        h = self.out_proj(h)

        return self.act_fn(self.norm(x + h))


class PVConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        voxel_resolution: int,
        dropout_prob: float = 0.1,
        use_attention: bool = False,
        use_se: bool = False,
        normalize: bool = True,
        eps: float = 0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.voxel_resolution = voxel_resolution

        self.voxelizer = Voxelizer(
            voxel_resolution, normalize=normalize, eps=eps
        )
        # self.voxelizer = Voxelization(voxel_resolution, normalize=normalize, eps=eps)

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
            voxel_layers.append(nn.Dropout(dropout_prob))

        voxel_layers += [
            nn.Conv3d(
                out_channels, out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            Attention(out_channels, 8) if use_attention else nn.SiLU(inplace=True),
        ]

        if use_se:
            voxel_layers.append(SE3D(out_channels))

        self.voxel_layers = nn.Sequential(*voxel_layers)
        self.point_layers = SharedMLP(num_channels=[in_channels, out_channels])

    def forward(self, points: PointTensor) -> PointTensor:
        voxels: PointTensor = self.voxelizer(points)
        # voxel_features, voxel_coords = self.voxelizer(
        #     points.features, points.coords.permute(0, 2, 1)
        # )

        global tmp_index
        # torch.save((points.features, points.coords, voxel_features, voxel_coords), f"voxelizer{tmp_index}.pth")
        (
            _points_features, _points_coords, _voxel_features, _voxel_coords
        ) = torch.load(f"../PVD/PVD/voxelizer{tmp_index}.pth")
        tmp_index += 1

        print("voxel_resolution", self.voxel_resolution)
        print("_points_features", torch.allclose(_points_features, points.features), (_points_features - points.features).abs().max())
        print("_points_coords", torch.allclose(_points_coords, points.coords), (_points_coords - points.coords).abs().max())
        print("_voxel_features", torch.allclose(_voxel_features, voxels.features), (_voxel_features - voxels.features).abs().max())
        print("_voxel_coords", torch.allclose(_voxel_coords, voxels.coords), (_voxel_coords - voxels.coords).abs().max())

        # voxel_features = torch.ones_like(voxel_features)
        # voxel_features = _voxel_features
        # voxels = PointTensor(
        #     coords=voxel_coords.permute(0, 2, 1),
        #     features=voxel_features,
        # )

        voxel_features = self.voxel_layers[0](voxels.features)
        point_features = self.point_layers(points.features)

        # print("weight", self.voxel_layers[0].weight.abs().max())
        # print("bias", self.voxel_layers[0].bias.abs().max())

        # voxel_features, *_ = trilinear_devoxelize(
        #     voxels.coords, voxel_features, self.voxel_resolution
        # )
        tmp = voxel_features.clone()
        voxel_features = trilinear_devoxelize(
            voxel_features, voxels.coords.permute(0, 2, 1),
            self.voxel_resolution, self.training
        )
        # print("voxel_features", voxel_features.shape)

        global tmp_index2
        # torch.save((points.features, points.coords, voxel_features, voxel_coords), f"voxelizer{tmp_index}.pth")
        (
            _tmp2, _tmp, _voxels_coords, _voxel_features
        ) = torch.load(f"../PVD/PVD/devoxelize{tmp_index2}.pth")
        tmp_index2 += 1

        print("-" * 40)
        print("_tmp2", torch.allclose(_tmp2, voxels.features), (_tmp2 - voxels.features).abs().max())
        print("_tmp", torch.allclose(_tmp, tmp), (_tmp - tmp).abs().max())
        print("_voxel_features", torch.allclose(_voxel_features, voxel_features), (_voxel_features - voxel_features).abs().max())
        print("_voxels_coords", torch.allclose(_voxels_coords, voxels.coords), (_voxels_coords - voxels.coords).abs().max())
        print("^" * 40)

        x = points.clone()
        x.coords = points.coords
        x.features = voxel_features + point_features
        # x.features = point_features
        return x
