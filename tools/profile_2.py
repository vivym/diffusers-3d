import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv3d(32, 32, kernel_size=1)
        self.conv3 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.linear = nn.Linear(32, 32)

    def forward(self, x):
        with record_function("conv1"):
            x = self.conv1(x)

        with record_function("conv3"):
            x = self.conv3(x)

        with record_function("linear"):
            x = self.linear(x.flatten(2).permute(0, 2, 1))

        return x


def main():
    device = torch.device("cuda")

    model = Model()
    model.eval()
    model.to(device)

    sample_shape = (1024, 32, 16, 16, 16)

    sample = torch.randn(sample_shape, device=device)

    with torch.no_grad():
        for _ in range(10):
            output = model(sample)
            print(output.mean())

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
        ) as prof:
            with record_function("model_inference"):
                model(sample)
                print(output.mean())

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


if __name__ == "__main__":
    main()
