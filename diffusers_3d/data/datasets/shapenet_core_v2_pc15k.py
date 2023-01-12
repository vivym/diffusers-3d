import json
from dataclasses import dataclass
from typing import List
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

synset_id_to_category = {
    "02691156": "airplane", "02773838": "bag", "02801938": "basket",
    "02808440": "bathtub", "02818832": "bed", "02828884": "bench",
    "02876657": "bottle", "02880940": "bowl", "02924116": "bus",
    "02933112": "cabinet", "02747177": "can", "02942699": "camera",
    "02954340": "cap", "02958343": "car", "03001627": "chair",
    "03046257": "clock", "03207941": "dishwasher", "03211117": "monitor",
    "04379243": "table", "04401088": "telephone", "02946921": "tin_can",
    "04460130": "tower", "04468005": "train", "03085013": "keyboard",
    "03261776": "earphone", "03325088": "faucet", "03337140": "file",
    "03467517": "guitar", "03513137": "helmet", "03593526": "jar",
    "03624134": "knife", "03636649": "lamp", "03642806": "laptop",
    "03691459": "speaker", "03710193": "mailbox", "03759954": "microphone",
    "03761084": "microwave", "03790512": "motorcycle", "03797390": "mug",
    "03928116": "piano", "03938244": "pillow", "03948459": "pistol",
    "03991062": "pot", "04004475": "printer", "04074963": "remote_control",
    "04090263": "rifle", "04099429": "rocket", "04225987": "skateboard",
    "04256520": "sofa", "04330267": "stove", "04530566": "vessel",
    "04554684": "washer", "02992529": "cellphone",
    "02843684": "birdhouse", "02871439": "bookshelf",
    # "02858304": "boat", no boat in our dataset, merged into vessels
    # "02834778": "bicycle", not in our taxonomy
}
category_to_synset_id = {v: k for k, v in synset_id_to_category.items()}


@dataclass
class DataItem:
    pcs: torch.Tensor
    labels: torch.Tensor


def collate_fn(batch: List[DataItem]) -> DataItem:
    return DataItem(
        pcs=torch.stack([data.pcs for data in batch], dim=0),
        labels=torch.as_tensor([data.labels for data in batch], dtype=torch.int64),
    )


class DatasetImpl(Dataset):
    def __init__(
        self,
        root_path: Path,
        split: str,
        categories: List[str],
        num_points: int = 2048,
    ):
        super().__init__()

        self.root_path = root_path
        self.split = split
        self.categories = categories
        self.num_points = num_points

        with open(self.root_path / f"{split}.json") as f:
            meta_infos = json.load(f)

        meta_infos = filter(lambda x: x["category"] in categories, meta_infos)
        self.meta_infos = list(meta_infos)

        with open(self.root_path / "stats_infos.json") as f:
            self.stats_infos = json.load(f)

        self.synset_id_to_label = {
            synset_id: i
            for i, synset_id in enumerate(sorted(synset_id_to_category.keys()))
        }

    def __len__(self) -> int:
        return len(self.meta_infos)

    def __getitem__(self, index: int):
        meta_info = self.meta_infos[index]

        synset_id = meta_info["synset_id"]
        label = self.synset_id_to_label[synset_id]
        pc_mean = self.stats_infos[synset_id]["mean"]
        pc_std = self.stats_infos[synset_id]["std"]
        pc_std_all = self.stats_infos[synset_id]["std_all"]

        pc_mean = np.asarray(pc_mean).reshape(1, -1)
        pc_std = np.asarray(pc_std).reshape(1, -1)
        pc_std_all = np.asarray(pc_std_all).reshape(1, -1)

        pc = np.load(self.root_path / meta_info["pc_path"]).astype(np.float32)

        sampled_idxs = np.arange(pc.shape[0], dtype=np.int64)
        sampled_idxs = np.random.choice(sampled_idxs, size=self.num_points)

        pc = (pc[sampled_idxs] - pc_mean) / pc_std_all
        pc = torch.from_numpy(pc)

        return DataItem(pcs=pc, labels=label)


class ShapeNetCoreV2PC15KDataset(pl.LightningDataModule):
    def __init__(
        self,
        root_path: str,
        categories: List[str],
        num_points: int = 2048,
        train_batch_size: int = 256,
        val_batch_size: int = 256,
        test_batch_size: int = 256,
        num_workers: int = 16,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.root_path = Path(root_path).resolve()
        self.categories = categories
        self.num_points = num_points
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

        self.train_meta_path = self.root_path / "train.json"
        self.val_meta_path = self.root_path / "val.json"
        self.test_meta_path = self.root_path / "test.json"
        self.stats_infos_path = self.root_path / "stats_infos.json"

        for category in categories:
            assert category in category_to_synset_id, category

    def prepare_data(self):
        if (
            self.train_meta_path.exists() and self.val_meta_path.exists() and
            self.test_meta_path.exists() and self.stats_infos_path.exists()
        ):
            return

        splits = ["train", "val", "test"]
        pc_infos = {split: [] for split in splits}
        stats_infos = {synset_id: [] for synset_id in synset_id_to_category.keys()}

        for synset_id_path in tqdm(list(self.root_path.glob("*"))):
            synset_id = synset_id_path.name
            if not synset_id_path.is_dir() or synset_id not in synset_id_to_category:
                continue
            category = synset_id_to_category[synset_id]

            pcs = []

            for split in splits:
                for pc_path in (synset_id_path / split).glob("*.npy"):
                    pc = np.load(pc_path)
                    assert pc.shape[0] == 15000
                    pcs.append(pc)

                    pc_infos[split].append({
                        "category": category,
                        "synset_id": synset_id,
                        "pc_path": str(pc_path.relative_to(self.root_path)),
                    })

            pcs = np.concatenate(pcs, axis=0)
            pc_mean = pcs.mean(0).tolist()
            pc_std = pcs.std(0).tolist()
            pc_std_all = pcs.reshape(-1).std(0).item()

            stats_infos[synset_id] = {
                "mean": pc_mean,
                "std": pc_std,
                "std_all": pc_std_all,
            }

        with open(self.root_path / "stats_infos.json", "w") as f:
            json.dump(stats_infos, f)

        for split in splits:
            with open(self.root_path / f"{split}.json", "w") as f:
                json.dump(pc_infos[split], f)

    def train_dataloader(self):
        dataset = DatasetImpl(
            root_path=self.root_path,
            split="train",
            categories=self.categories,
            num_points=self.num_points,
        )

        return DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        dataset = DatasetImpl(
            root_path=self.root_path,
            split="val",
            categories=self.categories,
            num_points=self.num_points,
        )

        return DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        dataset = DatasetImpl(
            root_path=self.root_path,
            split="test",
            categories=self.categories,
            num_points=self.num_points,
        )

        return DataLoader(
            dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )


def main():
    dataset = ShapeNetCoreV2PC15KDataset(
        root_path="./data/ShapeNetCore.v2.PC15k",
        categories=["car"],
    )
    dataset.prepare_data()
    dataloader = dataset.train_dataloader()

    for batch in dataloader:
        break


if __name__ == "__main__":
    main()
