import numpy as np
import torch
from tqdm import tqdm

from diffusers_3d.pl_modules.pvcnn import PVCNN
from diffusers_3d.models.pvcnn import pvcnn_debug
from diffusers_3d.structures.points import PointTensor


def main():
    device = torch.device("cuda")

    model = PVCNN()
    model.model = pvcnn_debug()
    # print(model)

    state_dict = torch.load("../PVD/PVD/output/train_generation_pl_no_noise_init/2023-01-15-18-10-08/epoch_1799.pth", map_location="cpu")
    print("state_dict", list(state_dict.keys()))
    new_state_dict = {}
    for k, v in state_dict["model_state"].items():
        k = k[len("model.module."):]
        new_state_dict[k] = v

    model.model.load_state_dict(new_state_dict)

    model.model.eval()
    model.to(device)

    sample_shape = (16, 2048, 3)

    sample = torch.randn(sample_shape, device=device)
    sample = torch.load("../PVD/PVD/tmp/noise.pth").permute(0, 2, 1)

    # set step values
    model.noise_scheduler.set_timesteps(1000)

    with torch.no_grad():
        t = 1
        sample = torch.load(f"../PVD/PVD/tmp/sample{t + 1}.pth").permute(0, 2, 1)

        model_output = model(sample, t)
        tmp = torch.load(f"../PVD/PVD/tmp/model_output{t}.pth").permute(0, 2, 1)
        print("model_output", torch.allclose(
            tmp, model_output,
            atol=1e-4, rtol=1e-4,
        ))
        print("model_output diff", (tmp - model_output).abs().max())


if __name__ == "__main__":
    main()
