from __future__ import annotations

import os
import queue as Queue
import random
import threading
from collections.abc import Callable
from functools import partial
from glob import glob

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ..utils.utils_distributed_sampler import DistributedSampler, get_dist_info, worker_init_fn


def normalize_by_127_5(img: torch.Tensor) -> torch.Tensor:
    img = (img * 255.0).int()
    return (img / 127.5) - 1.0


def tensor2img(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor * 255.0).int()


def get_img_list(path: str) -> list[str]:
    files: list[str] = []
    for ext in ("*.gif", "*.png", "*.jpg"):
        files.extend(glob(os.path.join(path, ext)))
    return files


def synchronized_horizontal_flip_manual(image1: Image.Image, image2: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Applies synchronized horizontal flip to both images."""
    if random.random() > 0.5:
        image1 = TF.hflip(image1)
        image2 = TF.hflip(image2)
    return image1, image2


def load_text_from_file(txt_path: str) -> str:
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return line
    raise ValueError(f"No non-empty line found in {txt_path}")


class FaceImageDataset_ImageOnly(Dataset):
    def __init__(
        self,
        db_path: str,
        t_transform: Callable | None = None,
        s_transform: Callable | None = None,
    ) -> None:
        self.img_list = get_img_list(os.path.join(db_path, "img"))
        np.random.shuffle(self.img_list)
        self.t_transform = t_transform
        self.s_transform = s_transform
        self.num_sample = len(self.img_list)

    def __len__(self) -> int:
        return self.num_sample

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        src_idx = random.randint(0, self.num_sample - 1)
        tar_idx = random.randint(0, self.num_sample - 1)
        while src_idx == tar_idx:
            tar_idx = random.randint(0, self.num_sample - 1)

        src_rgb_img = Image.open(self.img_list[src_idx]).convert("RGB")
        tar_rgb_img = Image.open(self.img_list[tar_idx]).convert("RGB")

        if self.t_transform is not None and self.s_transform is not None:
            img1_t = self.t_transform(src_rgb_img)
            img2_t = self.t_transform(tar_rgb_img)
        return img1_t, img2_t


class FaceImageDataset_CLIP(Dataset):
    def __init__(
        self,
        db_path: str,
        t_transform: Callable | None = None,
        s_transform: Callable | None = None,
    ) -> None:
        # Prefer packed/ directory when present; fall back to legacy 3-file layout.
        packed_dir = os.path.join(db_path, "packed")
        self.use_packed = os.path.isdir(packed_dir) and bool(get_img_list(packed_dir))

        if self.use_packed:
            self.img_list = get_img_list(packed_dir)
        else:
            self.img_list = get_img_list(os.path.join(db_path, "img"))
            self.mask_path = os.path.join(db_path, "mask")
            self.text_path = os.path.join(db_path, "txt")

        np.random.shuffle(self.img_list)
        self.t_transform = t_transform
        self.s_transform = s_transform
        self.num_sample = len(self.img_list)

    def __len__(self) -> int:
        return self.num_sample

    def __getitem__(self, idx: int):
        src_idx = random.randint(0, self.num_sample - 1)
        tar_idx = random.randint(0, self.num_sample - 1)
        while src_idx == tar_idx:
            tar_idx = random.randint(0, self.num_sample - 1)

        if self.use_packed:
            return self._getitem_packed(src_idx, tar_idx)
        return self._getitem_legacy(src_idx, tar_idx)

    def _getitem_packed(self, src_idx: int, tar_idx: int) -> tuple:
        from ..preprocess.pack_png import unpack_png

        src = unpack_png(self.img_list[src_idx])
        tar = unpack_png(self.img_list[tar_idx])

        src_rgb_img = Image.fromarray(src.img_rgb)
        tar_rgb_img = Image.fromarray(tar.img_rgb)

        # mask stored as 0=face, 255=bg — convert to RGB for t_transform consistency
        src_msk_img = (
            Image.fromarray(src.mask).convert("RGB")
            if src.mask is not None
            else Image.new("RGB", src_rgb_img.size, (255, 255, 255))
        )
        tar_msk_img = (
            Image.fromarray(tar.mask).convert("RGB")
            if tar.mask is not None
            else Image.new("RGB", tar_rgb_img.size, (255, 255, 255))
        )

        src_rgb_img, src_msk_img = synchronized_horizontal_flip_manual(src_rgb_img, src_msk_img)
        tar_rgb_img, tar_msk_img = synchronized_horizontal_flip_manual(tar_rgb_img, tar_msk_img)

        img1_t = self.t_transform(src_rgb_img)
        img2_t = self.t_transform(tar_rgb_img)
        mask1_t = self.t_transform(src_msk_img)
        mask2_t = self.t_transform(tar_msk_img)

        src_text_str = src.caption or ""
        tar_text_str = tar.caption or ""

        def _to_tensor(arr):
            return torch.from_numpy(arr.copy()) if arr is not None else None

        return (
            img1_t,
            img2_t,
            1 - mask1_t,
            1 - mask2_t,
            src_text_str,
            tar_text_str,
            _to_tensor(src.clip_img_emb),
            _to_tensor(tar.clip_img_emb),
            _to_tensor(src.clip_txt_emb),
            _to_tensor(tar.clip_txt_emb),
            _to_tensor(src.id_emb),
            _to_tensor(tar.id_emb),
        )

    def _getitem_legacy(
        self, src_idx: int, tar_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, str]:
        src_img_path = self.img_list[src_idx]
        tar_img_path = self.img_list[tar_idx]

        src_rgb_img = Image.open(src_img_path).convert("RGB")
        tar_rgb_img = Image.open(tar_img_path).convert("RGB")

        src_mask_path = os.path.join(self.mask_path, src_img_path.split("/")[-1])
        tar_mask_path = os.path.join(self.mask_path, tar_img_path.split("/")[-1])

        src_txt_path = os.path.join(self.text_path, src_img_path.split("/")[-1].split(".")[0] + ".txt")
        tar_txt_path = os.path.join(self.text_path, tar_img_path.split("/")[-1].split(".")[0] + ".txt")

        src_text_str = load_text_from_file(src_txt_path)
        tar_text_str = load_text_from_file(tar_txt_path)

        src_msk_img = Image.open(src_mask_path).convert("RGB")
        tar_msk_img = Image.open(tar_mask_path).convert("RGB")

        src_rgb_img, src_msk_img = synchronized_horizontal_flip_manual(src_rgb_img, src_msk_img)
        tar_rgb_img, tar_msk_img = synchronized_horizontal_flip_manual(tar_rgb_img, tar_msk_img)

        if self.t_transform is not None and self.s_transform is not None:
            img1_t = self.t_transform(src_rgb_img)
            img2_t = self.t_transform(tar_rgb_img)
            mask1_t = self.t_transform(src_msk_img)
            mask2_t = self.t_transform(tar_msk_img)
        return img1_t, img2_t, 1 - mask1_t, 1 - mask2_t, src_text_str, tar_text_str


class FaceImageDataset(Dataset):
    def __init__(
        self,
        db_path: str,
        t_transform: Callable | None = None,
        s_transform: Callable | None = None,
    ) -> None:
        self.img_list = get_img_list(os.path.join(db_path, "img"))
        self.mask_path = os.path.join(db_path, "mask")
        np.random.shuffle(self.img_list)
        self.t_transform = t_transform
        self.s_transform = s_transform
        self.num_sample = len(self.img_list)

    def __len__(self) -> int:
        return self.num_sample

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        src_idx = random.randint(0, self.num_sample - 1)
        tar_idx = random.randint(0, self.num_sample - 1)
        while src_idx == tar_idx:
            tar_idx = random.randint(0, self.num_sample - 1)

        src_img_path = self.img_list[src_idx]
        tar_img_path = self.img_list[tar_idx]

        src_rgb_img = Image.open(src_img_path).convert("RGB")
        tar_rgb_img = Image.open(tar_img_path).convert("RGB")

        src_mask_path = os.path.join(self.mask_path, src_img_path.split("/")[-1])
        tar_mask_path = os.path.join(self.mask_path, tar_img_path.split("/")[-1])

        src_msk_img = Image.open(src_mask_path).convert("RGB")
        tar_msk_img = Image.open(tar_mask_path).convert("RGB")

        src_rgb_img, src_msk_img = synchronized_horizontal_flip_manual(src_rgb_img, src_msk_img)
        tar_rgb_img, tar_msk_img = synchronized_horizontal_flip_manual(tar_rgb_img, tar_msk_img)

        if self.t_transform is not None and self.s_transform is not None:
            img1_t = self.t_transform(src_rgb_img)
            img2_t = self.t_transform(tar_rgb_img)
            mask1_t = self.t_transform(src_msk_img)
            mask2_t = self.t_transform(tar_msk_img)
        return img1_t, img2_t, 1 - mask1_t, 1 - mask2_t


def _make_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    t_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )
    s_transform = transforms.Compose(
        [
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Lambda(normalize_by_127_5),
        ]
    )
    return t_transform, s_transform


def get_dataloader(db_path: str, batch_size: int, num_workers: int = 4) -> DataLoader:
    t_transform, s_transform = _make_transforms()
    train_set = FaceImageDataset(db_path, t_transform=t_transform, s_transform=s_transform)
    return DataLoader(dataset=train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)


def get_dataloader_img_only(db_path: str, batch_size: int, num_workers: int = 4) -> DataLoader:
    t_transform, s_transform = _make_transforms()
    train_set = FaceImageDataset_ImageOnly(db_path, t_transform=t_transform, s_transform=s_transform)
    return DataLoader(dataset=train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)


def get_dataloader_clip(db_path: str, batch_size: int, num_workers: int = 4) -> DataLoader:
    t_transform, s_transform = _make_transforms()
    train_set = FaceImageDataset_CLIP(db_path, t_transform=t_transform, s_transform=s_transform)
    return DataLoader(dataset=train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)


def get_dataloader_fixed_src_tar(db_path: str, batch_size: int, num_workers: int = 4) -> DataLoader:
    t_transform, s_transform = _make_transforms()
    train_set = FaceImageDataset(db_path, t_transform=t_transform, s_transform=s_transform)
    return DataLoader(dataset=train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)


def get_dataloader_tmp(
    db_path: str,
    local_rank: int,
    batch_size: int,
    seed: int = 2048,
    num_workers: int = 2,
) -> DataLoaderX:
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )
    train_set = FaceImageDataset(db_path, transform)
    rank, world_size = get_dist_info()
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)
    init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)
    return DataLoaderX(
        local_rank=local_rank,
        dataset=train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=init_fn,
    )


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator: DataLoader, local_rank: int, max_prefetch: int = 6) -> None:
        super().__init__()
        self.queue: Queue.Queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self) -> None:
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self) -> object:
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self) -> object:
        return self.next()

    def __iter__(self) -> BackgroundGenerator:
        return self


class DataLoaderX(DataLoader):
    def __init__(self, local_rank: int, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self) -> DataLoaderX:
        self.iter = super().__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self) -> None:
        self.batch, _ = next(self.iter, None)
        if self.batch is None:
            return
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self) -> list[torch.Tensor]:
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch
