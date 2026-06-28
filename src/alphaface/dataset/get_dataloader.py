from __future__ import annotations

import os
import random
from collections.abc import Callable
from glob import glob

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


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


class FaceImageDatasetClip(Dataset):
    def __init__(
        self,
        db_path: str,
        t_transform: Callable | None = None,
        s_transform: Callable | None = None,
    ) -> None:
        packed_dir = os.path.join(db_path, "packed")
        if not os.path.isdir(packed_dir):
            raise FileNotFoundError(f"Packed dataset directory not found: {packed_dir}")

        self.img_list = get_img_list(packed_dir)
        if not self.img_list:
            raise FileNotFoundError(f"No packed PNG samples found in {packed_dir}")
        if len(self.img_list) < 2:
            raise ValueError(f"Packed training dataset needs at least two samples, found {len(self.img_list)}")
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

        src_text_str = src.caption
        tar_text_str = tar.caption

        def _to_tensor(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr.copy())

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
    return get_dataloader_clip(db_path, batch_size=batch_size, num_workers=num_workers)


def get_dataloader_img_only(db_path: str, batch_size: int, num_workers: int = 4) -> DataLoader:
    raise NotImplementedError("Image-only training datasets are no longer supported; use packed PNG samples")


def get_dataloader_clip(db_path: str, batch_size: int, num_workers: int = 4) -> DataLoader:
    t_transform, s_transform = _make_transforms()
    train_set = FaceImageDatasetClip(db_path, t_transform=t_transform, s_transform=s_transform)
    return DataLoader(dataset=train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)


def get_dataloader_fixed_src_tar(db_path: str, batch_size: int, num_workers: int = 4) -> DataLoader:
    return get_dataloader_clip(db_path, batch_size=batch_size, num_workers=num_workers)
