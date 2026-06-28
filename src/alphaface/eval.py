from __future__ import annotations

import os
import sys
from collections.abc import Iterable
from pathlib import Path

import pytorch_lightning as L
import torch
import torchvision.transforms as transforms
from PIL import Image
from pytorch_lightning.cli import LightningCLI
from torch.utils.data import DataLoader, Dataset

from .lit_module import AlphaFaceLitModule
from .models.swapper_alphaface import remap_legacy_swapper_state_dict

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "eval.yaml"


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


class AlphaFaceEvalDataModule(L.LightningDataModule):
    def __init__(self, src_path: str, tar_path: str, output: str, batch_size: int = 1) -> None:
        super().__init__()
        self.src_path = src_path
        self.tar_path = tar_path
        self.output = output
        self.batch_size = batch_size
        self.dataset: _SwapPairDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        self.dataset = _SwapPairDataset(list_images(self.src_path), list_images(self.tar_path))

    def predict_dataloader(self) -> DataLoader:
        assert self.dataset is not None, "call setup() before predict_dataloader()"
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)


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


def _default_predict_args(args: list[str]) -> list[str]:
    return args or ["--config", str(DEFAULT_CONFIG)]


def main(args: list[str] | None = None) -> None:
    cli_args = _default_predict_args(list(sys.argv[1:] if args is None else args))
    if args is None:
        sys.argv = [sys.argv[0], *cli_args]
        cli_args = None

    cli = LightningCLI(
        AlphaFaceLitModule,
        AlphaFaceEvalDataModule,
        args=cli_args,
        run=False,
        save_config_callback=None,
        auto_configure_optimizers=False,
    )

    cfg = cli.model.cfg
    os.makedirs(getattr(cfg, "log_dir", "tmp"), exist_ok=True)
    os.makedirs(cli.datamodule.output, exist_ok=True)

    print("Preparing the AlphaFace model")

    print(f"Resuming from checkpoint... from {cfg.model_path}")
    ckpt = torch.load(cfg.model_path, map_location="cpu")
    state = ckpt.get("swapper", ckpt)
    cli.model.model.swapper.load_state_dict(remap_legacy_swapper_state_dict(state))

    cli.datamodule.setup("predict")
    assert cli.datamodule.dataset is not None
    outputs = cli.trainer.predict(cli.model, datamodule=cli.datamodule)

    for idx, swapped in enumerate(outputs or []):
        _save_swapped(swapped, os.path.join(cli.datamodule.output, cli.datamodule.dataset.output_name(idx)))


if __name__ == "__main__":
    main()
