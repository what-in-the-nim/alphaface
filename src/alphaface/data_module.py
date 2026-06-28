from __future__ import annotations

import pytorch_lightning as L
from torch.utils.data import DataLoader

from .dataset.get_dataloader import FaceImageDatasetClip, _make_transforms


class AlphaFaceDataModule(L.LightningDataModule):
    """Wraps the CLIP face-swap dataset for the Lightning Trainer.

    Lightning injects the distributed sampler under DDP automatically, so the custom
    ``DistributedSampler``/``DataLoaderX`` wiring from the old loop is no longer needed.
    """

    def __init__(self, db_path: str, batch_size: int, num_workers: int = 4) -> None:
        super().__init__()
        self.db_path = db_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_set: FaceImageDatasetClip | None = None

    def setup(self, stage: str | None = None) -> None:
        t_transform, s_transform = _make_transforms()
        self.train_set = FaceImageDatasetClip(self.db_path, t_transform=t_transform, s_transform=s_transform)

    def train_dataloader(self) -> DataLoader:
        assert self.train_set is not None, "call setup() before train_dataloader()"
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )
