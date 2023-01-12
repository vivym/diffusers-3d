from typing import List, Optional

import torch
import torch.nn as nn
from einops import repeat

from diffusers_3d.ops.ball_query import ball_query
from diffusers_3d.ops.fps import furthest_point_sampling
from diffusers_3d.ops.knn_interpolate import three_nn_interpolate
from diffusers_3d.structures import PointTensor

from .shared_mlp import SharedMLP


class QueryAndGroup(nn.Module):
    def __init__(self, radius: float, max_samples_per_query: int, use_coords: bool = True):
        super().__init__()

        self.radius = radius
        self.max_samples_per_query = max_samples_per_query
        self.use_coords = use_coords

    def forward(
        self,
        points: PointTensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        assert not points.is_channel_last

        # b, n, m
        idxs = ball_query(
            points.coords, queries,
            radius=self.radius,
            max_samples_per_query=self.max_samples_per_query,
        )
        # b, n, 3 -> b, (n * m), 3 -> b, n, m, 3
        grouped_coords = torch.gather(
            points.coords,
            dim=1,
            index=repeat(idxs, "b n m -> b (n m) c", c=3),
        ).view(*idxs.shape, 3)
        grouped_coords -= queries[:, :, None, :]

        if points.features is None:
            assert self.use_coords
            grouped_features = grouped_coords
        else:
            # b, c, n -> b, c, (n * m) -> b, c, n, m
            grouped_features = torch.gather(
                points.features,
                dim=2,
                index=repeat(idxs, "b n m -> b c (n m)", c=points.features.shape[1]),
            ).view(*points.features.shape[:2], *idxs.shape[1:])

            if self.use_coords:
                grouped_features = torch.cat([
                    grouped_coords.permute(0, 3, 1, 2), grouped_features
                ], dim=1)

            if points.t_embed is not None:
                # b, c, n -> b, c, (n * m) -> b, c, n, m
                t_embed = torch.gather(
                    points.t_embed,
                    dim=2,
                    index=repeat(idxs, "b n m -> b c (n m)", c=points.t_embed.shape[1]),
                ).view(*points.t_embed.shape[:2], *idxs.shape[1:])
            else:
                t_embed = None

        return grouped_features, t_embed

    def extra_repr(self) -> str:
        msg = f"radius={self.radius}, max_samples_per_query={self.max_samples_per_query}"
        if self.use_coords:
            msg += ", use coordinates"
        return msg


class PointNetSAModuleMSG(nn.Module):
    def __init__(
        self,
        num_points: int,
        radii: List[float],
        max_samples_per_query: List[int],
        mlp_specs: List[List[int]],
        use_coords: bool = True,
    ):
        super().__init__()

        assert len(radii) == len(max_samples_per_query) == len(mlp_specs)

        self.num_points = num_points
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()

        for radius, max_samples_per_query_i, mlp_spec in zip(
            radii, max_samples_per_query, mlp_specs
        ):
            self.groupers.append(QueryAndGroup(
                radius=radius,
                max_samples_per_query=max_samples_per_query_i,
                use_coords=use_coords,
            ))

            if use_coords:
                mlp_spec[0] += 3

            self.mlps.append(SharedMLP(num_channels=mlp_spec, dim=2))

    def forward(self, x: PointTensor) -> PointTensor:
        features_list = []
        t_embed_list = []

        sampled_coords = torch.gather(
            x.coords,
            dim=1,
            index=repeat(
                furthest_point_sampling(x.coords, self.num_points),
                "b m -> b m c", c=3,
            ),
        )

        for grouper, mlp in zip(self.groupers, self.mlps):
            grouped_features, grouped_t_embed = grouper(x, sampled_coords)
            grouped_features = mlp(grouped_features)
            # b, c, n, m -> b, c, n
            features_list.append(grouped_features.max(dim=-1).values)
            t_embed_list.append(grouped_t_embed.max(dim=-1).values)

        x = x.clone()
        x.coords = sampled_coords
        x.features = torch.cat(features_list, dim=1)
        x.t_embed = torch.cat(t_embed_list, dim=1)

        return x


class PointNetSAModule(PointNetSAModuleMSG):
    def __init__(
        self,
        num_points: int,
        radius: float,
        max_samples_per_query: int,
        mlp_spec: List[int],
        use_coords: bool = True,
    ):
        super().__init__(
            num_points=num_points,
            radii=[radius],
            max_samples_per_query=[max_samples_per_query],
            mlp_specs=[mlp_spec],
            use_coords=use_coords,
        )


class PointNetFPModule(nn.Module):
    def __init__(self, mlp_spec: List[int]):
        super().__init__()

        self.mlp = SharedMLP(mlp_spec)

    def forward(self, points: PointTensor, ref_points: PointTensor) -> PointTensor:
        # TODO: optimize this
        features, *_ = three_nn_interpolate(
            coords=points.coords,
            ref_coords=ref_points.coords,
            features=ref_points.features,
        )
        t_embed, *_ = three_nn_interpolate(
            coords=points.coords,
            ref_coords=ref_points.coords,
            features=ref_points.t_embed,
        )

        if points.features is not None:
            features = torch.cat([features, points.features], dim=1)

        x = points.clone()
        x.features = features
        x.t_embed = t_embed
        return x
