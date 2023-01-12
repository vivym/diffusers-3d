import torch


@torch.no_grad()
def furthest_point_sampling(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    points = points.contiguous()

    return torch.ops.diffusers_3d.furthest_point_sampling(points, num_samples)


def main():
    points = torch.randn(3, 1024, 3, dtype=torch.float32, device="cuda")
    idxs = furthest_point_sampling(points, 256)
    print(idxs.shape, idxs.min(), idxs.max())


if __name__ == "__main__":
    main()
