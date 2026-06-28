from __future__ import annotations

import os
from pathlib import Path

import hydra
import pytorch_lightning as L
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from .data_module import AlphaFaceDataModule
from .lit_module import AlphaFaceLitModule

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


def _select_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@hydra.main(config_path=CONFIG_DIR, config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    L.seed_everything(cfg.seed, workers=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.output, exist_ok=True)

    device = _select_device()
    print(f"Using device: {device}")

    model = AlphaFaceLitModule(cfg, device=device)
    datamodule = AlphaFaceDataModule(cfg.db_path, cfg.batch_size, cfg.num_workers)

    if getattr(cfg, "using_wandb", False):
        logger: object = WandbLogger(project=cfg.wandb_project, entity=cfg.wandb_entity)
    else:
        logger = TensorBoardLogger(save_dir=cfg.tb_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.output,
        every_n_train_steps=cfg.save_interval,
        save_last=True,
        save_top_k=-1,
        filename="model_{step}",
    )

    trainer = L.Trainer(
        max_epochs=cfg.num_epoch,
        accelerator="auto",
        devices="auto",
        precision="16-mixed" if getattr(cfg, "use_amp", False) and device == "cuda" else "32-true",
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=100,
    )

    ckpt_path = cfg.model_path if getattr(cfg, "resume", False) and cfg.model_path else None
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
