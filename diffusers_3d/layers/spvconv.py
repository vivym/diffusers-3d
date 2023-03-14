import torch
import torch.nn as nn
import torch.nn.functional as F
from sparse_ops.voxelize import voxelize
from sparse_ops.devoxelize import trilinear_devoxelize
import spconv.pytorch as spconv

from diffusers_3d.structures.points import PointTensor

from .voxelization import get_voxel_coords
from .se import SE3D
from .shared_mlp import SharedMLP

g_voxels = None


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

        voxel_layers = [
            spconv.SparseConv3d(
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

        use_attention = False
        use_se = False

        voxel_layers += [
            spconv.SparseConv3d(
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

        self.voxel_layers = spconv.SparseSequential(*voxel_layers)
        self.point_layers = SharedMLP(num_channels=[in_channels, out_channels])

    def forward(self, points: PointTensor) -> PointTensor:
        coords, _ = get_voxel_coords(points.coords, self.voxel_resolution)

        voxel_size = torch.as_tensor([1., 1., 1.], dtype=coords.dtype)
        points_range_min = torch.as_tensor([0., 0., 0.], dtype=coords.dtype)
        points_range_max = torch.as_tensor(
            [self.voxel_resolution, self.voxel_resolution, self.voxel_resolution],
            dtype=coords.dtype,
        )
        voxels = voxelize(
            point_coords=coords.contiguous(),
            point_features=points.features.permute(0, 2, 1).contiguous(),
            voxel_size=voxel_size.contiguous(),
            points_range_min=points_range_min.contiguous(),
            points_range_max=points_range_max.contiguous(),
        )

        sp_tensor = spconv.SparseConvTensor(
            features=voxels.features,
            indices=torch.cat([
                voxels.batch_indices[:, None],
                voxels.coords,
            ], dim=-1).to(torch.int32),
            spatial_shape=(
                self.voxel_resolution, self.voxel_resolution, self.voxel_resolution
            ),
            batch_size=voxels.batch_indices.max().item() + 1,
        )

        sp_tensor = self.voxel_layers(sp_tensor)

        voxel_features, *_ = trilinear_devoxelize(
            coords.contiguous(),
            voxel_size=voxel_size.contiguous(),
            points_range_min=points_range_min.contiguous(),
            points_range_max=points_range_max.contiguous(),
            voxel_coords=sp_tensor.indices[:, 1:].long().contiguous(),
            voxel_features=sp_tensor.features.contiguous(),
            voxel_batch_indices=sp_tensor.indices[:, 0].long().contiguous(),
        )
        voxel_features = voxel_features.permute(0, 2, 1)

        point_features = self.point_layers(points.features)

        x = points.clone()
        x.coords = points.coords
        x.features = point_features + voxel_features
        return x
