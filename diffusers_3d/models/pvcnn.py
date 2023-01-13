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
                pvconv_spec.use_attention = pvconv_spec.use_attention and (i % 2 == 1)

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
                pvconv_spec.use_attention = \
                    pvconv_spec.use_attention and (i % 2 == 1) and (i < len(up_block_specs) - 1)
                # TODO:
                pvconv_spec.use_attention = False

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
            print("in_points_list", i, in_points_list[-1].features.mean())
            if i > 0:
                points.features = torch.cat(
                    [points.features, points.t_embed], dim=1
                )

            points = down_block(points)

        # only use extra features in the last fp module
        in_points_list[0].features = in_points_list[0].features[:, 3:, :]

        # gloabl attention
        points.features = self.global_attn(points.features)

        for i, up_block in enumerate(self.up_blocks):
            points.features = torch.cat([points.features, points.t_embed], dim=1)
            points = up_block(points, ref_points=in_points_list[-1 - i])
            print("up points.features", points.features.mean())

        feats = torch.load("../PVD/AAA.pth", map_location="cuda")
        print("!!!" * 8, torch.allclose(feats, points.features))
        print("feats", feats.mean())
        print("points.features", points.features.mean())

        return self.out_proj(points.features), in_points_list


def pvcnn_base(in_channels: int = 3, time_embed_dim: int = 64):
    down_block_specs = [
        (
            PVConvSpec(out_channels=32, num_layers=1, voxel_resolution=32),
            SAModuleSpec(
                num_points=1024,
                radius=0.1,
                max_samples_per_query=32,
                out_channels=[32, 64],
            ),
        ),
        (
            PVConvSpec(out_channels=64, num_layers=1, voxel_resolution=16),
            SAModuleSpec(
                num_points=256,
                radius=0.2,
                max_samples_per_query=32,
                out_channels=[64, 128],
            ),
        ),
        (
            PVConvSpec(out_channels=128, num_layers=1, voxel_resolution=8),
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
    data_t, t = torch.load("../PVD/model_inputs.pth", map_location="cpu")

    state_dict = torch.load("../PVD/model.pth", map_location="cpu")
    # print("state_dict", list(state_dict.keys()))
    new_state_dict = {}
    for k, v in state_dict.items():
        # print(k, v.shape)
        if k.startswith("embedf"):
            k = k.replace("embedf", "time_embed_proj")
        elif k.startswith("classifier"):
            k = k.replace("classifier", "out_proj")
        elif k.startswith("global_att"):
            k = k.replace("global_att", "global_attn")
            k = k.replace("q.", "q_proj.")
            k = k.replace("k.", "k_proj.")
            k = k.replace("v.", "v_proj.")
            k = k.replace("out.", "out_proj.")
        elif k.startswith("sa_layers"):
            k = k.replace("sa_layers", "down_blocks")
            k = k.replace("point_features", "point_layers")
            k = k.replace("q.", "q_proj.")
            k = k.replace("k.", "k_proj.")
            k = k.replace("v.", "v_proj.")
            k = k.replace("out.", "out_proj.")
            if not k.split(".")[2].isnumeric():
                k = ".".join(k.split(".")[:2]) + f".0." + ".".join(k.split(".")[2:])
        elif k.startswith("fp_layers"):
            k = k.replace("fp_layers", "up_blocks")
            k = k.replace("point_features", "point_layers")
            k = k.replace("q.", "q_proj.")
            k = k.replace("k.", "k_proj.")
            k = k.replace("v.", "v_proj.")
            k = k.replace("out.", "out_proj.")
            # fp_layers.0.0.mlp.layers.0.weight -> up_blocks.0.fp_module.mlp.layers.0.weight
            # fp_layers.0.1.voxel_layers.0.weight -> up_blocks.0.convs.0.voxel_layers.0.weight
            layer_idx = int(k.split(".")[2])
            if layer_idx == 0:
                k = ".".join(k.split(".")[:2]) + f".fp_module." + ".".join(k.split(".")[3:])
            else:
                k = ".".join(k.split(".")[:2]) + f".convs.{layer_idx - 1}." + ".".join(k.split(".")[3:])

        new_state_dict[k] = v

    device = torch.device("cuda")

    model = pvcnn_base(in_channels=3)
    model.load_state_dict(new_state_dict, strict=True)

    print(model)
    model.to(device)

    # data = torch.load("../PVD/inputs.pth", map_location="cpu")
    # assert torch.allclose(data, data_t)

    points = PointTensor(
        coords=data_t.permute(0, 2, 1).to(device),
        features=data_t.to(device),
        is_channel_last=False,
        timesteps=t.to(device),
    )

    # from tqdm import tqdm
    # for _ in tqdm(range(10000)):
    #     model(points)

    _, _, eps_recon = torch.load("../PVD/inputs_outputs.pth", map_location="cpu")

    model.eval()

    with torch.no_grad():
        results, in_points_list = model(points)
    # print("results", results.shape)
    # print("results", results)
    # print("eps_recon", eps_recon)

    in_features_list, coords_list, features, coords, temb = torch.load("../PVD/sa_blocks_features.pth", map_location="cuda")
    for in_features, in_points in zip(in_features_list, in_points_list):
        print("PVD in_features", in_features.mean())
        print("in_points", in_points.features.mean())
        print(torch.allclose(in_features, in_points.features, atol=1e-3, rtol=1e-3))
    print("temb", temb.mean())

    # print("t_embed", t_embed)
    # print("temb", temb)
    # print("t_embed == temb", torch.allclose(t_embed.cpu(), temb, rtol=1e-4, atol=1e-4))


if __name__ == "__main__":
    main()
