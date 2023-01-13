import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.act_fn = Swish()

    def forward(self, x: torch.Tensor):
        q = self.q_proj(x).flatten(2)
        k = self.k_proj(x).flatten(2)
        v = self.v_proj(x).flatten(2)

        qk = q.permute(0, 2, 1) @ k

        w = F.softmax(qk, dim=-1)

        h = (v @ w.permute(0, 2, 1)).reshape_as(x)
        h = self.out_proj(h)

        return self.act_fn(self.norm(x + h))


class Swish(nn.Module):
    def forward(self,x):
        return  x * torch.sigmoid(x)


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

        voxel_layers = [
            nn.Conv3d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            # nn.SiLU(inplace=True),
            Swish(),
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
            # Attention(out_channels, 8) if use_attention else nn.SiLU(inplace=True),
            Attention(out_channels, 8) if use_attention else Swish(),
        ]

        if use_se:
            voxel_layers.append(SE3D(out_channels))

        self.voxel_layers = nn.Sequential(*voxel_layers)
        self.point_layers = SharedMLP(num_channels=[in_channels, out_channels])

    def forward(self, points: PointTensor) -> PointTensor:
        voxels: PointTensor = self.voxelizer(points)

        features, coords, voxel_features, voxel_coords = torch.load(
            "../PVD/voxel.pth", map_location="cuda"
        )
        print("!" * 10, voxel_coords.dtype)
        data = torch.load("../PVD/inputs.pth", map_location="cuda")
        print("in_features", features.shape, torch.allclose(data, features), torch.allclose(data, points.features))
        print("in_features", torch.allclose(features, points.features))
        print("coords", torch.allclose(coords.permute(0, 2, 1), points.coords))
        print("voxel_features", torch.allclose(voxel_features, voxels.features))
        print("voxel_features diff", (voxel_features - voxels.features).abs().max())
        print("voxel_coords", torch.allclose(voxel_coords.permute(0, 2, 1), voxels.coords))
        diff = voxel_coords.permute(0, 2, 1) - voxels.coords
        print("voxel_coords diff", diff.abs().max())

        # voxel_features = self.voxel_layers(voxels.features)
        tmp1 = voxel_features
        voxel_features = self.voxel_layers(voxel_features)
        point_features = self.point_layers(points.features)

        tmp = self.voxel_layers(voxels.features)

        print("@@@@", (tmp - voxel_features).abs().max(), (tmp1 - voxels.features).abs().max())

        pvd_point_features = torch.load(
            "../PVD/point_features.pth", map_location="cuda"
        )
        print("point_features", torch.allclose(pvd_point_features, point_features))

        # print("voxels.features", voxels.features.mean())
        # print("voxel_features", voxel_features.mean())
        # print("point_features", point_features.mean())

        in_features, coords, resolution, out_features = torch.load(
            "../PVD/trilinear_devoxelize.pth", map_location="cuda"
        )
        print("in_features", torch.allclose(in_features, voxel_features))
        print("coords", torch.allclose(coords.permute(0, 2, 1), voxels.coords))
        print("resolution", resolution, self.voxel_resolution)

        # voxel_features, *_ = trilinear_devoxelize(
        #     voxels.coords, voxel_features, self.voxel_resolution
        # )

        print("coords diff", (coords.permute(0, 2, 1) - voxels.coords).abs().max())
        print("in_features diff", (voxel_features - in_features).abs().max())

        voxel_features, *_ = trilinear_devoxelize(
            voxels.coords, voxel_features, self.voxel_resolution
        )

        print("out_features diff", (voxel_features - out_features).abs().max())

        print((out_features == voxel_features).all())
        print("out_features", torch.allclose(out_features, voxel_features))

        # print("trilinear_devoxelize voxel_features", voxel_features.mean())

        # exit(0)

        x = points.clone()
        x.coords = points.coords
        x.features = voxel_features + point_features
        return x
