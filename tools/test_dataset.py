import torch

from diffusers_3d.data.datasets import ShapeNetCoreV2PC15KDataset


def main():
    dataset = ShapeNetCoreV2PC15KDataset(
        root_path="data/ShapeNetCore.v2.PC15k",
        categories=["chair"],
        train_batch_size=16,
        val_batch_size=16,
        num_workers=0,
    )

    train_dataloader = dataset.train_dataloader()

    for batch in train_dataloader:
        pcs = batch.pcs
        print(pcs.shape)
        torch.save(pcs, "generated.pth")
        break


if __name__ == "__main__":
    main()
