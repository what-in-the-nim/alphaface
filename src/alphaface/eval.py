from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from .dataset.get_dataloader import get_dataloader
from .models.swapper_alphaface import AlphaFace, build_AlphaFace
from .objectives.loss import *
from .utils.utils_config import get_config
from .utils.utils_distributed_sampler import setup_seed


def list_images(
    directory: str | Path,
    extensions: Iterable[str] | None = None,
    recursive: bool = False,
) -> list[Path]:
    """Return a sorted list of image files found in *directory*."""
    default_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".webp", ".ppm", ".pgm"}
    exts = default_exts if extensions is None else {
        (e.lower() if e.startswith(".") else f".{e.lower()}") for e in extensions
    }
    directory = Path(directory).expanduser().resolve()
    paths = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(p for p in paths if p.is_file() and p.suffix.lower() in exts)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def normalize_by_127_5(img: torch.Tensor) -> torch.Tensor:
    img = (img * 255.0).int()
    return (img / 127.5) - 1.0


def tensor2img(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor * 255.0).int()


def eval_alphaface(config: object, model: AlphaFace) -> None:
    writer = SummaryWriter(config.tb_dir)  # type: ignore[attr-defined]

    os.makedirs(config.output, exist_ok=True)  # type: ignore[attr-defined]

    print("Resuming from checkpoint...from %s" % config.model_path)  # type: ignore[attr-defined]
    dict_checkpoint = torch.load(config.model_path)  # type: ignore[attr-defined]
    model.Swapper.load_state_dict(dict_checkpoint["swapper"])

    model = model.cuda()
    print(f"Number of parameters in Swapper: {count_parameters(model.Swapper)}")
    model.Swapper.eval()
    model.Id_encoder.eval()

    t_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    s_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Lambda(normalize_by_127_5),
    ])

    src_img_list = list_images(config.src_path)  # type: ignore[attr-defined]
    tar_img_list = list_images(config.tar_path)  # type: ignore[attr-defined]

    for src_img_file in src_img_list:
        img1_s = s_transform(Image.open(src_img_file)).unsqueeze(0).cuda()

        for tar_img_file in tar_img_list:
            img2_t = t_transform(Image.open(tar_img_file)).unsqueeze(0).cuda()
            swapped_2_1 = model(img2_t, img1_s)

            if swapped_2_1.dim() == 4:
                swapped_2_1 = swapped_2_1[0]
            if swapped_2_1.shape[0] in {1, 3, 4}:
                swapped_2_1 = swapped_2_1.permute(1, 2, 0)
            if swapped_2_1.max() <= 1.0:
                swapped_2_1 = swapped_2_1 * 255.0

            swapped_2_1 = swapped_2_1.clamp(0, 255).byte()
            np_image = swapped_2_1.cpu().numpy()
            img = Image.fromarray(np_image)
            if np_image.shape[2] == 1:
                img = img.convert("L")

            save_name = src_img_file.stem + "_" + tar_img_file.name
            save_path = os.path.join(config.output, save_name)  # type: ignore[attr-defined]
            img.save(save_path)
            print(f"Saved image to {save_path}")


def main(args: argparse.Namespace) -> None:
    cfg = get_config(args.config)
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    if cfg.tensorboard:
        writer = SummaryWriter(cfg.tb_path)
    else:
        writer = None
    os.makedirs(cfg.log_dir, exist_ok=True)

    print("Preparing the student model")
    alphaface = build_AlphaFace(config=cfg).to("cuda")
    eval_alphaface(cfg, alphaface)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaFace evaluation")
    parser.add_argument("config", type=str, default="./configs/test_config", help="py config file")
    main(parser.parse_args())
