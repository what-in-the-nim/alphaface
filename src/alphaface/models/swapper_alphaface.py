from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn

from .arcface_resnet import resnet50
from .swapper_units_adin import Decoder, Discriminator, Encoder, EncoderNoBNIN, VGGPerceptualLoss

# Repo-root models/ dir, resolved from this file so loading works from any cwd.
MODELS_DIR = Path(__file__).resolve().parents[3] / "models"


class Normalize(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self.mean
        x = x / self.std
        return x


class ResNetIdEncoder(nn.Module):
    def __init__(self, ema_path: str, device: str = "cpu") -> None:
        super().__init__()
        # NOTE: `.resnet` and `.ema` attribute names are load-bearing — they map to the
        # pretrained ArcFace checkpoint and emp.npy keys and must not be renamed.
        self.resnet = resnet50()
        self.ema = nn.Linear(512, 512)
        self.ema.weight.data = torch.nn.Parameter(torch.from_numpy(np.load(ema_path)).permute(1, 0).to(device))
        self.ema.bias.data.fill_(0.0)

    def processing_ema_only(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.ema(latent)
        return torch.div(x, torch.linalg.norm(x, dim=1, keepdim=True))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.resnet(inputs)
        x = self.ema(x)
        return torch.div(x, torch.linalg.norm(x, dim=1, keepdim=True))


class Swapper(nn.Module):
    def __init__(self, source_dim: int, exist_bn: bool = True, use_checkpoint: bool = False) -> None:
        super().__init__()
        self.source_dim = source_dim
        self.encoder = (
            Encoder(self.source_dim, use_checkpoint=use_checkpoint)
            if exist_bn
            else EncoderNoBNIN(self.source_dim, use_checkpoint=use_checkpoint)
        )
        self.decoder = Decoder(1024, 3)

    def forward(
        self, target: torch.Tensor, source: torch.Tensor, get_latent: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        output = self.encoder(target, source)
        if get_latent:
            return self.decoder(output, get_latent=True)
        return self.decoder(output)


class AlphaFace(nn.Module):
    def __init__(self, swapper: Swapper, id_encoder: nn.Module, fine_tune: bool = False, device: str = "cpu") -> None:
        super().__init__()
        self._device_str = device
        self.swapper = swapper.to(device)
        self.fine_tune = fine_tune
        self.id_encoder = id_encoder.to(device)
        self.discriminator: Discriminator | None = None
        self.train_adv: bool = False
        self.perceptual_loss: VGGPerceptualLoss | None = None

    def prepare_adversarial_learning(self, img_size: int, max_conv_size: int) -> None:
        self.discriminator = Discriminator(img_size=img_size, max_conv_dim=max_conv_size).to(self._device_str)
        self.train_adv = True

    def prepare_vgg_perceptual_loss(self) -> None:
        self.perceptual_loss = VGGPerceptualLoss(layer_ids=(3, 8, 15, 22)).to(self._device_str)

    def set_grads(self) -> None:
        for param in self.swapper.parameters():
            param.requires_grad = True
        for param in self.id_encoder.parameters():
            param.requires_grad = False
        if self.train_adv and self.discriminator is not None:
            for param in self.discriminator.parameters():
                param.requires_grad = True

    def get_id_code(self, source: torch.Tensor) -> torch.Tensor:
        return self.id_encoder(source).detach()

    def forward(
        self, target: torch.Tensor, source: torch.Tensor, get_latent: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if not self.fine_tune:
            source = self.id_encoder(source)
        return self.swapper(target, source, get_latent=get_latent)

    def add_batch_instance_norm_to_swapper(self) -> None:
        bn_layer = nn.BatchNorm2d(1024)
        in_layer = nn.InstanceNorm2d(1024, affine=True)
        self.swapper.encoder.Encoder["layer_3"] = nn.Sequential(
            self.swapper.encoder.Encoder["layer_3"],
            bn_layer,
            in_layer,
        )
        for i in range(5):
            setattr(self.swapper.encoder, f"batch_norm_{i}", nn.BatchNorm2d(1024))
            setattr(self.swapper.encoder, f"instance_norm_{i}", nn.InstanceNorm2d(1024, affine=True))


def remap_legacy_swapper_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remap old ``Swapper`` checkpoint keys (``E.``/``G.``) to the renamed attributes
    (``encoder.``/``decoder.``) so pre-refactor swapper weights still load."""
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("E."):
            key = "encoder." + key[len("E.") :]
        elif key.startswith("G."):
            key = "decoder." + key[len("G.") :]
        remapped[key] = value
    return remapped


def build_alpha_face(
    config: DictConfig | None = None,
    fine_tune: bool = False,
    adv_train: bool = True,
    device: str = "cpu",
) -> AlphaFace:
    swapper = Swapper(512, use_checkpoint=True)
    print("Loading pre-trained model for the ID encoder")
    id_encoder: nn.Module = ResNetIdEncoder(ema_path=str(MODELS_DIR / "emp.npy"), device=device)
    id_encoder.resnet.load_state_dict(  # type: ignore[attr-defined]
        torch.load(str(MODELS_DIR / "arcface_w600k_r50_pytorch.pt"), map_location=torch.device(device))
    )

    model = AlphaFace(swapper, id_encoder, fine_tune=fine_tune, device=device)
    if adv_train:
        model.prepare_adversarial_learning(img_size=256, max_conv_size=512)
        model.prepare_vgg_perceptual_loss()
    return model
