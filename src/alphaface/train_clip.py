from __future__ import annotations

import sys
from pathlib import Path

import torch
from pytorch_lightning.cli import LightningCLI

from .data_module import AlphaFaceDataModule
from .lit_module import AlphaFaceLitModule

torch.set_float32_matmul_precision("medium")

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "train.yaml"
SUBCOMMANDS = {"fit", "validate", "test", "predict"}


class AlphaFaceCLI(LightningCLI):
    def before_fit(self) -> None:
        cfg = self.model.cfg
        Path(getattr(cfg, "log_dir", "tmp")).mkdir(parents=True, exist_ok=True)
        Path(getattr(cfg, "output", "./save/save_clip")).mkdir(parents=True, exist_ok=True)


def _default_fit_args(args: list[str]) -> list[str]:
    if not args:
        return ["fit", "--config", str(DEFAULT_CONFIG)]
    if args[0] in SUBCOMMANDS:
        return args
    return ["fit", *args]


def main(args: list[str] | None = None) -> None:
    cli_args = _default_fit_args(list(sys.argv[1:] if args is None else args))
    if args is None:
        sys.argv = [sys.argv[0], *cli_args]
        cli_args = None
    AlphaFaceCLI(
        AlphaFaceLitModule,
        AlphaFaceDataModule,
        args=cli_args,
        save_config_callback=None,
        auto_configure_optimizers=False,
    )


if __name__ == "__main__":
    main()
