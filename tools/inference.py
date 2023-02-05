import torch
from tqdm import tqdm

from diffusers_3d.pl_modules.pvcnn import PVCNN


def main():
    device = torch.device("cuda")

    model = PVCNN.load_from_checkpoint(
        "wandb/lightning_logs/version_36/checkpoints/epoch_002929.ckpt"
    )
    model.eval()
    model.to(device)

    sample_shape = (25, 2048, 3)

    sample = torch.randn(sample_shape, device=device)

    # set step values
    model.noise_scheduler.set_timesteps(1000)

    with torch.no_grad():
        for t in tqdm(model.noise_scheduler.timesteps):
            # 1. predict noise model_output
            model_output = model(sample, t)

            # 2. compute previous image: x_t -> x_t-1
            sample = model.noise_scheduler.step(
                model_output, t, sample
            ).prev_sample

    print("sample", sample.shape)
    torch.save(sample, "generated.pth")


if __name__ == "__main__":
    main()
