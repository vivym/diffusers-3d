import torch


@torch.no_grad()
def ball_query(
    points: torch.Tensor, queries: torch.Tensor, radius: float, max_samples_per_query: int
) -> torch.Tensor:
    points = points.contiguous()
    queries = queries.contiguous()

    return torch.ops.diffusers_3d.ball_query(
        points, queries, radius, max_samples_per_query
    )


def main():
    points = torch.randn(3, 1024, 3, dtype=torch.float32, device="cuda")
    idxs = ball_query(points, points, 0.1, 50)
    print(idxs.shape, idxs.min(), idxs.max())


if __name__ == "__main__":
    main()
