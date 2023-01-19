from typing import Tuple

import torch


def three_nn_interpolate(
    src_coords: torch.Tensor, src_features: torch.Tensor, tgt_coords: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    src_coords = src_coords.contiguous()
    src_features = src_features.contiguous()
    tgt_coords = tgt_coords.contiguous()

    # tgt_features, indices, weights
    return torch.ops.diffusers_3d.three_nn_interpolate(
        src_coords, src_features, tgt_coords
    )


def main():
    (
        points_coords, centers_coords, centers_features, points_features, temb, features, t_embed,
        interpolated_features, interpolated_temb,
    ) = torch.load(
        "../PVD/PVD/three_nn_interpolate.pth", map_location="cpu"
    )

    print("features", torch.allclose(interpolated_features, features))
    print("t_embed", torch.allclose(interpolated_temb, t_embed))
    return

    # points_coords = points_coords.permute(0, 2, 1)
    # centers_coords = centers_coords.permute(0, 2, 1)
    # indices = indices.permute(0, 2, 1)
    # weights = weights.permute(0, 2, 1)

    res = three_nn_interpolate(centers_coords, centers_features, points_coords)

    print((res[0] == points_features).all())
    print((res[1] == indices).all())
    print((res[2] == weights).all())


if __name__ == "__main__":
    main()
