import torch
import torch.nn as nn
import torch.nn.functional as F
from spconv.pytorch.utils import PointToVoxel

from diffusers_3d.structures.points import PointTensor
from diffusers_3d.ops.trilinear_devoxelize import trilinear_devoxelize

from .voxelization import Voxelizer
from .se import SE3D
from .shared_mlp import SharedMLP


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

        # self.voxelizer = Voxelizer(
        #     voxel_resolution, normalize=normalize, eps=eps
        # )
        self.voxelizer = PointToVoxel(
            vsize_xyz=(1 / voxel_resolution, 1 / voxel_resolution, 1 / voxel_resolution),
            coors_range_xyz=(0, 0, 0, 1, 1, 1),
            num_point_features=3 + in_channels,
            max_num_voxels=2048,
            max_num_points_per_voxel=8,
            device=torch.device("cuda"),
        )

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
        from torch.profiler import record_function

        with record_function("voxelizer"):
            # voxels: PointTensor = self.voxelizer(points)
            # print("points", points.features.shape)
            # print("voxels", voxels.features.shape, (voxels.features != 0).any(1).sum(), (voxels.features != 0).any(1).sum() / voxels.features.numel())
            for coords, features in zip(points.coords, points.features):
                (
                    voxel_features, voxel_coords, num_points_per_voxel, pc_voxel_id
                ) = self.voxelizer.generate_voxel_with_id(
                    torch.cat([coords, features.permute(1, 0)], dim=-1)
                )
                print("voxel_features", voxel_features.shape)
                print("voxel_coords", voxel_coords.shape)
                print("num_points_per_voxel", num_points_per_voxel.shape, num_points_per_voxel.float().mean())
                print("pc_voxel_id", pc_voxel_id.shape)
        exit(0)

        with record_function("voxel_layers"):
            voxel_features = self.voxel_layers(voxels.features)

        with record_function("point_layers"):
            point_features = self.point_layers(points.features)

        voxel_features, *_ = trilinear_devoxelize(
            voxels.coords, voxel_features, self.voxel_resolution
        )

        x = points.clone()
        x.coords = points.coords
        x.features = point_features + voxel_features
        return x
