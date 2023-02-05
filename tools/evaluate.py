from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from diffusers_3d.data.datasets import ShapeNetCoreV2PC15KDataset
from diffusers_3d.pl_modules.pvcnn import PVCNN


def visualize_pointcloud_batch(
    path, pointclouds, pred_labels = None, labels = None, categories = None, vis_label=False, target=None,  elev=30, azim=225
):
    batch_size = len(pointclouds)
    fig = plt.figure(figsize=(20,20))

    ncols = int(np.sqrt(batch_size))
    nrows = max(1, (batch_size-1) // ncols+1)
    for idx, pc in enumerate(pointclouds):
        if vis_label:
            label = categories[labels[idx].item()]
            pred = categories[pred_labels[idx]]
            colour = 'g' if label == pred else 'r'
        elif target is None:

            colour = 'g'
        else:
            colour = target[idx]
        pc = pc.cpu().numpy()
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')
        ax.scatter(pc[:, 0], pc[:, 2], pc[:, 1], c=colour, s=5)
        ax.view_init(elev=elev, azim=azim)
        ax.axis('off')
        if vis_label:
            ax.set_title('GT: {0}\nPred: {1}'.format(label, pred))

    plt.savefig(path)
    plt.close(fig)


def main():
    ckpt_path = Path("wandb/lightning_logs/version_36/checkpoints/epoch_002929.ckpt")
    output_path = ckpt_path.parent.parent / "eval"

    if not output_path.exists():
        output_path.mkdir()

    dataset = ShapeNetCoreV2PC15KDataset(
        root_path="data/ShapeNetCore.v2.PC15k",
        categories=["chair"],
        train_batch_size=16,
        val_batch_size=64,
        num_workers=0,
    )

    val_dataloader = dataset.val_dataloader()

    device = torch.device("cuda")

    model = PVCNN.load_from_checkpoint(
        "wandb/lightning_logs/version_36/checkpoints/epoch_002929.ckpt"
    )
    model.eval()
    model.to(device)

    for i, batch in tqdm(enumerate(val_dataloader)):
        pcs, mean, std = batch.pcs, batch.mean, batch.std
        pcs = pcs.to(device)
        mean = mean[:, None, :].to(device)
        std = std[:, None, :].to(device)

        sample = torch.randn_like(pcs)

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

        pcs = pcs * std + mean
        sample = sample * std + mean

        torch.save((pcs, sample), output_path / f"output_{i:06d}.pth")

        visualize_pointcloud_batch(output_path / f"vis_{i:06d}_ref.jpg", pcs)
        visualize_pointcloud_batch(output_path / f"vis_{i:06d}_gen.jpg", sample)


if __name__ == "__main__":
    main()
