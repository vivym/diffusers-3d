import torch
import torch.nn as nn
from torch_scatter import scatter

from diffusers_3d.structures.points import PointTensor


@torch.no_grad()
# @torch.jit.script
def get_voxel_coords(
    coords: torch.Tensor, resolution: int, normalize: bool = True, eps: float = 0.
):
    coords = coords - coords.mean(dim=1, keepdim=True)
    if normalize:
        coords = coords / (
            coords.norm(
                p=2, dim=2, keepdim=True
            ).max(dim=1, keepdim=True).values * 2.0 + eps
        ) + 0.5
    else:
        coords = (coords + 1) / 2.0

    coords = torch.clamp(coords * resolution, 0, resolution - 1)
    coords_int = torch.round(coords).to(torch.int64)

    idxs = coords_int[..., 0] * int(resolution ** 2) + \
           coords_int[..., 1] * int(resolution ** 1) + \
           coords_int[..., 2] * int(resolution ** 0)

    return coords, idxs


class Voxelizer(nn.Module):
    def __init__(
        self,
        resolution: int,
        normalize: bool = True,
        eps: float = 0,
    ):
        super().__init__()

        self.resolution = resolution
        self.normalize = normalize
        self.eps = eps

    def forward(self, points: PointTensor):
        voxel_coords, voxel_coord_idxs = get_voxel_coords(
            points.coords, self.resolution, self.normalize, self.eps
        )

        if points.is_channel_last:
            # B, R^3, C
            voxel_features = scatter(
                points.features,
                index=voxel_coord_idxs[:, :, None],
                dim=1,
                dim_size=self.resolution ** 3,
                reduce="mean",
            )
            # B, R, R, R, C
            voxel_features = voxel_features.view(
                points.features.shape[0],
                self.resolution,
                self.resolution,
                self.resolution,
                points.features.shape[2],
            )
        else:
            # B, C, R^3
            voxel_features = scatter(
                points.features,
                index=voxel_coord_idxs[:, None, :],
                dim=2,
                dim_size=self.resolution ** 3,
                reduce="mean",
            )
            # B, C, R, R, R
            voxel_features = voxel_features.view(
                *points.features.shape[:2],
                self.resolution,
                self.resolution,
                self.resolution,
            )

        return PointTensor(coords=voxel_coords, features=voxel_features)

    def extra_repr(self):
        msg = f"resolution={self.resolution}"
        if self.normalize:
            msg += f", normalized eps = {self.eps}"
        return msg
