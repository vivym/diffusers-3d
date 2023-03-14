import pytorch_lightning as pl
import torch
import torch.nn as nn

from diffusers_3d.data.datasets import ShapeNetCoreV2PC15KDataset
from diffusers_3d.layers.spvconv import PVConv
from diffusers_3d.structures.points import PointTensor
from sparse_ops.voxelize import voxelize
from sparse_ops.devoxelize import trilinear_devoxelize


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(10, 20)
        self.pvconv = PVConv(
            in_channels=256, out_channels=256,
            kernel_size=3, voxel_resolution=32,
        )

    def forward(self, x):
        ...

    def training_step(self, batch, idx):
        pcs = batch.pcs
        # print("pcs", pcs.shape)

        features = torch.randn(*pcs.shape[:2], 256, device=pcs.device)

        self.pvconv(PointTensor(
            coords=pcs,
            features=features.permute(0, 2, 1),
            is_channel_last=False,
        ))

        # voxel_size = torch.as_tensor([0.1, 0.1, 0.1], dtype=pcs.dtype)
        # points_range_min = pcs.reshape(-1, 3).min(0)[0].cpu().contiguous()
        # points_range_max = pcs.reshape(-1, 3).max(0)[0].cpu().contiguous()

        # # print("voxel_size", voxel_size)
        # # print("points_range_min", points_range_min)
        # # print("points_range_max", points_range_max)

        # # torch.save((pcs, features, voxel_size, points_range_min, points_range_max), "tmp.pth")

        # voxels = voxelize(
        #     point_coords=pcs.contiguous(),
        #     point_features=features.contiguous(),
        #     voxel_size=voxel_size.contiguous(),
        #     points_range_min=points_range_min.contiguous(),
        #     points_range_max=points_range_max.contiguous(),
        # )

        # # print("voxelized")

        # voxel_features, *_ = trilinear_devoxelize(
        #     pcs,
        #     voxel_size=voxel_size,
        #     points_range_min=points_range_min,
        #     points_range_max=points_range_max,
        #     voxel_coords=voxels.coords,
        #     voxel_features=voxels.features,
        #     voxel_batch_indices=voxels.batch_indices,
        # )

        # # print("voxel_features", voxel_features.mean())

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=1e-3,
        )


def main():
    model = Model()
    dm = ShapeNetCoreV2PC15KDataset(
        root_path="data/ShapeNetCore.v2.PC15k",
        categories=["chair"],
        train_batch_size=4,
        val_batch_size=4,
        num_workers=0,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        default_root_dir="wandb",
    )
    trainer.fit(
        model=model,
        datamodule=dm,
    )
    # for batch in dm.train_dataloader():
    #     pcs = batch.pcs.to("cuda")
    #     # pcs = pcs[:, :16, :]
    #     print("pcs", pcs.shape)

    #     features = torch.randn(*pcs.shape[:2], 256, device=pcs.device)

    #     voxel_size = torch.as_tensor([0.1, 0.1, 0.1], dtype=pcs.dtype)
    #     points_range_min = pcs.reshape(-1, 3).min(0)[0].cpu().contiguous()
    #     points_range_max = pcs.reshape(-1, 3).max(0)[0].cpu().contiguous()

    #     print("voxel_size", voxel_size)
    #     print("points_range_min", points_range_min)
    #     print("points_range_max", points_range_max)

    #     # torch.save((pcs, features, voxel_size, points_range_min, points_range_max), "tmp.pth")

    #     voxels = voxelize(
    #         point_coords=pcs.contiguous(),
    #         point_features=features.contiguous(),
    #         voxel_size=voxel_size.contiguous(),
    #         points_range_min=points_range_min.contiguous(),
    #         points_range_max=points_range_max.contiguous(),
    #     )

    #     print("voxelized")

    #     voxel_features, *_ = trilinear_devoxelize(
    #         pcs,
    #         voxel_size=voxel_size,
    #         points_range_min=points_range_min,
    #         points_range_max=points_range_max,
    #         voxel_coords=voxels.coords,
    #         voxel_features=voxels.features,
    #         voxel_batch_indices=voxels.batch_indices,
    #     )

    #     print("voxel_features", voxel_features.mean())


if __name__ == "__main__":
    main()
