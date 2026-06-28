from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import hydra
import pytorch_lightning as L
import torch
import torchvision.transforms as transforms
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .lit_module import AlphaFaceLitModule
from .models.swapper_alphaface import remap_legacy_swapper_state_dict

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


def list_images(
    directory: str | Path,
    extensions: Iterable[str] | None = None,
    recursive: bool = False,
) -> list[Path]:
    """Return a sorted list of image files found in *directory*."""
    default_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".webp", ".ppm", ".pgm"}
    exts = (
        default_exts
        if extensions is None
        else {(e.lower() if e.startswith(".") else f".{e.lower()}") for e in extensions}
    )
    directory = Path(directory).expanduser().resolve()
    paths = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(p for p in paths if p.is_file() and p.suffix.lower() in exts)


def normalize_by_127_5(img: torch.Tensor) -> torch.Tensor:
    img = (img * 255.0).int()
    return (img / 127.5) - 1.0


class _SwapPairDataset(Dataset):
    """Yields (target, source) tensor pairs for every source x target combination."""

    def __init__(self, src_paths: list[Path], tar_paths: list[Path]) -> None:
        self.pairs = [(s, t) for s in src_paths for t in tar_paths]
        self.t_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        self.s_transform = transforms.Compose(
            [transforms.Resize((112, 112)), transforms.ToTensor(), transforms.Lambda(normalize_by_127_5)]
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        src_file, tar_file = self.pairs[idx]
        source = self.s_transform(Image.open(src_file).convert("RGB"))
        target = self.t_transform(Image.open(tar_file).convert("RGB"))
        return target, source

    def output_name(self, idx: int) -> str:
        src_file, tar_file = self.pairs[idx]
        return src_file.stem + "_" + tar_file.name


def _save_swapped(tensor: torch.Tensor, path: str) -> None:
    img = tensor
    if img.dim() == 4:
        img = img[0]
    if img.shape[0] in {1, 3, 4}:
        img = img.permute(1, 2, 0)
    if img.max() <= 1.0:
        img = img * 255.0
    np_image = img.clamp(0, 255).byte().cpu().numpy()
    out = Image.fromarray(np_image)
    if np_image.shape[2] == 1:
        out = out.convert("L")
    out.save(path)
    print(f"Saved image to {path}")


@hydra.main(config_path=CONFIG_DIR, config_name="eval", version_base=None)
def main(cfg: DictConfig) -> None:
    L.seed_everything(cfg.seed, workers=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.output, exist_ok=True)

    print("Preparing the AlphaFace model")
    model = AlphaFaceLitModule(cfg)

    print(f"Resuming from checkpoint... from {cfg.model_path}")
    ckpt = torch.load(cfg.model_path, map_location="cpu")
    state = ckpt.get("swapper", ckpt)
    model.model.swapper.load_state_dict(remap_legacy_swapper_state_dict(state))

    dataset = _SwapPairDataset(list_images(cfg.src_path), list_images(cfg.tar_path))
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    trainer = L.Trainer(accelerator="auto", devices=1, logger=False)
    outputs = trainer.predict(model, dataloaders=loader)

    for idx, swapped in enumerate(outputs or []):
        _save_swapped(swapped, os.path.join(cfg.output, dataset.output_name(idx)))


if __name__ == "__main__":
    main()
