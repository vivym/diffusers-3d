from typing import List, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn

from diffusers_3d.layers.embeddings import SinusoidalTimestepEmbedding
from diffusers_3d.layers.pvconv import PVConv, Attention
from diffusers_3d.layers.pointnet2 import PointNetSAModule, PointNetFPModule
from diffusers_3d.layers.shared_mlp import SharedMLP
from diffusers_3d.structures.points import PointTensor


@dataclass
class PVConvSpec:
    out_channels: int
    num_layers: int
    voxel_resolution: int
    in_channels: Optional[int] = None
    kernel_size: int = 3
    dropout_prob: float = 0.1
    use_attention: bool = True
    use_se: bool = True
    normalize: bool = True
    eps: float = 0.


@dataclass
class SAModuleSpec:
    num_points: int
    radius: float
    max_samples_per_query: int
    out_channels: List[int]
    in_channels: Optional[int] = None


@dataclass
class FPModuleSpec:
    out_channels: List[int]
    in_channels: Optional[int] = None


class DownBlock(nn.Sequential):
    def __init__(
        self,
        pvconv_spec: Optional[PVConvSpec],
        sa_module_spec: SAModuleSpec,
    ):
        layers = []

        if pvconv_spec is not None:
            for i in range(pvconv_spec.num_layers):
                if i == 0:
                    in_channels = pvconv_spec.in_channels
                else:
                    in_channels = pvconv_spec.out_channels
                layers.append(
                    PVConv(
                        in_channels=in_channels,
                        out_channels=pvconv_spec.out_channels,
                        kernel_size=pvconv_spec.kernel_size,
                        voxel_resolution=pvconv_spec.voxel_resolution,
                        dropout_prob=pvconv_spec.dropout_prob,
                        use_attention=pvconv_spec.use_attention and i == 0,
                        use_se=pvconv_spec.use_se,
                        normalize=pvconv_spec.normalize,
                        eps=pvconv_spec.eps,
                    )
                )

        layers.append(PointNetSAModule(
            num_points=sa_module_spec.num_points,
            radius=sa_module_spec.radius,
            max_samples_per_query=sa_module_spec.max_samples_per_query,
            mlp_spec=[sa_module_spec.in_channels] + sa_module_spec.out_channels,
        ))

        super().__init__(*layers)


class UpBlock(nn.Module):
    def __init__(
        self,
        fp_module_spec: FPModuleSpec,
        pvconv_spec: Optional[PVConvSpec],
    ):
        super().__init__()

        self.fp_module = PointNetFPModule(
            mlp_spec=[fp_module_spec.in_channels] + fp_module_spec.out_channels
        )

        convs = []
        if pvconv_spec is not None:
            for i in range(pvconv_spec.num_layers):
                if i == 0:
                    in_channels = pvconv_spec.in_channels
                else:
                    in_channels = pvconv_spec.out_channels
                convs.append(
                    PVConv(
                        in_channels=in_channels,
                        out_channels=pvconv_spec.out_channels,
                        kernel_size=pvconv_spec.kernel_size,
                        voxel_resolution=pvconv_spec.voxel_resolution,
                        dropout_prob=pvconv_spec.dropout_prob,
                        use_attention=pvconv_spec.use_attention and i == 0,
                        use_se=pvconv_spec.use_se,
                        normalize=pvconv_spec.normalize,
                        eps=pvconv_spec.eps,
                    )
                )
        self.convs = nn.Sequential(*convs)

    def forward(self, points: PointTensor, ref_points: PointTensor) -> PointTensor:
        points = self.fp_module(points, ref_points=ref_points)
        return self.convs(points)


class PVCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        down_block_specs: List[Tuple[PVConvSpec, SAModuleSpec]],
        up_block_specs: List[Tuple[FPModuleSpec, PVConvSpec]],
    ):
        super().__init__()

        self.in_channels = in_channels

        self.time_embedding = SinusoidalTimestepEmbedding(
            num_channels=time_embed_dim
        )

        self.time_embed_proj = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        down_block_in_channels = []

        self.down_blocks = nn.ModuleList()
        for i, (pvconv_spec, sa_module_spec) in enumerate(down_block_specs):
            down_block_in_channels.append(in_channels if i > 0 else in_channels - 3)

            extra_channels = time_embed_dim if i > 0 else 0
            if pvconv_spec is not None:
                pvconv_spec.in_channels = in_channels + extra_channels
                pvconv_spec.use_attention = pvconv_spec.use_attention and i > 0

                sa_module_spec.in_channels = pvconv_spec.out_channels
            else:
                sa_module_spec.in_channels = in_channels + extra_channels

            self.down_blocks.append(
                DownBlock(
                    pvconv_spec=pvconv_spec,
                    sa_module_spec=sa_module_spec,
                )
            )

            in_channels = sa_module_spec.out_channels[-1]

        self.global_attn = Attention(
            num_channels=in_channels, num_groups=8, dim=1
        )
        self.up_blocks = nn.ModuleList()
        for i, (fp_module_spec, pvconv_spec) in enumerate(up_block_specs):
            fp_module_spec.in_channels = \
                in_channels + time_embed_dim + down_block_in_channels[-1 - i]

            if pvconv_spec is not None:
                pvconv_spec.in_channels = fp_module_spec.out_channels[-1]
                pvconv_spec.use_attention = pvconv_spec.use_attention and i > 0

            self.up_blocks.append(
                UpBlock(
                    fp_module_spec=fp_module_spec,
                    pvconv_spec=pvconv_spec,
                )
            )

            if pvconv_spec is not None:
                in_channels = pvconv_spec.out_channels
            else:
                in_channels = fp_module_spec.out_channels[-1]

        self.out_proj = nn.Sequential(
            SharedMLP(num_channels=[in_channels, in_channels * 2]),
            nn.Dropout(p=0.1),
            nn.Conv1d(in_channels * 2, self.in_channels, kernel_size=1),
        )

    def forward(self, points: PointTensor):
        assert not points.is_channel_last

        t_embed = self.time_embedding(points.timesteps)
        t_embed = self.time_embed_proj(t_embed)
        t_embed = t_embed[:, :, None].expand(*t_embed.shape, points.coords.shape[1])
        points.t_embed = t_embed

        in_points_list: List[PointTensor] = []
        for i, down_block in enumerate(self.down_blocks):
            in_points_list.append(points.clone())
            if i > 0:
                points.features = torch.cat(
                    [points.features, t_embed], dim=1
                )

            points = down_block(points)

        # only use extra features in the last fp module
        in_points_list[0].features = in_points_list[0].features[:, 3:, :]

        # gloabl attention
        points.features = self.global_attn(points.features)

        for i, up_block in enumerate(self.up_blocks):
            points.features = torch.cat([points.features, t_embed], dim=1)
            points = up_block(points, ref_points=in_points_list[-1 - i])

        return self.out_proj(points.features)


def pvcnn_base(in_channels: int = 3, time_embed_dim: int = 64):
    down_block_specs = [
        (
            PVConvSpec(out_channels=32, num_layers=2, voxel_resolution=32),
            SAModuleSpec(
                num_points=1024,
                radius=0.1,
                max_samples_per_query=32,
                out_channels=[32, 64],
            ),
        ),
        (
            PVConvSpec(out_channels=64, num_layers=3, voxel_resolution=16),
            SAModuleSpec(
                num_points=256,
                radius=0.2,
                max_samples_per_query=32,
                out_channels=[64, 128],
            ),
        ),
        (
            PVConvSpec(out_channels=128, num_layers=3, voxel_resolution=8),
            SAModuleSpec(
                num_points=64,
                radius=0.4,
                max_samples_per_query=32,
                out_channels=[128, 256],
            ),
        ),
        (
            None,
            SAModuleSpec(
                num_points=16,
                radius=0.8,
                max_samples_per_query=32,
                out_channels=[256, 256, 512],
            ),
        ),
    ]

    up_block_specs = [
        (
            FPModuleSpec(out_channels=[256, 256]),
            PVConvSpec(out_channels=256, num_layers=3, voxel_resolution=8),
        ),
        (
            FPModuleSpec(out_channels=[256, 256]),
            PVConvSpec(out_channels=256, num_layers=3, voxel_resolution=8),
        ),
        (
            FPModuleSpec(out_channels=[256, 128]),
            PVConvSpec(out_channels=128, num_layers=2, voxel_resolution=16),
        ),
        (
            FPModuleSpec(out_channels=[128, 128, 64]),
            PVConvSpec(out_channels=64, num_layers=2, voxel_resolution=32),
        ),
    ]

    return PVCNN(
        in_channels=in_channels,
        time_embed_dim=time_embed_dim,
        down_block_specs=down_block_specs,
        up_block_specs=up_block_specs,
    )


def main():
    device = torch.device("cuda")

    model = pvcnn_base(in_channels=3)
    print(model)
    model.to(device)

    points = PointTensor(
        coords=torch.randn(10, 2048, 3).to(device),
        features=torch.randn(10, 3, 2048).to(device),
        is_channel_last=False,
        timesteps=torch.randint(0, 10, size=(10,)).to(device),
    )
    model(points)


if __name__ == "__main__":
    main()
