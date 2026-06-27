from __future__ import annotations

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch import nn

from ..backbones import get_model
from .arcface_resnet import resnet50
from .swapper_units_adin import Decoder, Discriminator, Encoder, Encoder_noBNIN, VGGPerceptualLoss

batch_size = 1


class Normalize(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self.mean
        x = x / self.std
        return x


class ResNet_ID_encoder(pl.LightningModule):
    def __init__(self, ema_path: str) -> None:
        super().__init__()
        self.resnet = resnet50()
        self.ema = nn.Linear(512, 512)
        self.ema.weight.data = torch.nn.Parameter(torch.from_numpy(np.load(ema_path)).permute(1, 0)).cuda()
        self.ema.bias.data.fill_(0.0)

    def processing_ema_only(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.ema(latent)
        return torch.div(x, torch.linalg.norm(x, dim=1, keepdim=True))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.resnet(inputs)
        x = self.ema(x)
        return torch.div(x, torch.linalg.norm(x, dim=1, keepdim=True))


class Swapper(nn.Module):
    def __init__(self, source_dim: int, exist_BN: bool = True) -> None:
        super().__init__()
        self.source_dim = source_dim
        self.E = Encoder(self.source_dim) if exist_BN else Encoder_noBNIN(self.source_dim)
        self.G = Decoder(1024, 3)
        self.dis: Discriminator | None = None
        self.train_adv: bool | None = None

    def forward(
        self, target: torch.Tensor, source: torch.Tensor, get_latent: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        output = self.E(target, source)
        if get_latent:
            return self.G(output, get_latent=True)
        return self.G(output)


class AlphaFace(pl.LightningModule):
    def __init__(self, swapper: Swapper, id_encoder: nn.Module, fine_tune: bool = False) -> None:
        super().__init__()
        self.Swapper = swapper.cuda()
        self.fine_tune = fine_tune
        self.Id_encoder = id_encoder.cuda()
        self.dis: Discriminator | None = None
        self.train_adv: bool = False
        self.feats_extractor: VGGPerceptualLoss | None = None

    def Prepareing_adversrial_learning(self, img_size: int, max_conv_size: int) -> None:
        self.dis = Discriminator(img_size=img_size, max_conv_dim=max_conv_size).cuda()
        self.train_adv = True

    def Preparing_VGG_percet_loss(self) -> None:
        self.feats_extractor = VGGPerceptualLoss(layer_ids=(3, 8, 15, 22)).cuda()

    def set_grads(self) -> None:
        for param in self.Swapper.parameters():
            param.requires_grad = True
        for param in self.Id_encoder.parameters():
            param.requires_grad = False
        if self.train_adv and self.dis is not None:
            for param in self.dis.parameters():
                param.requires_grad = True

    def save(self, fname: str, step: int) -> None:
        print(f"Saving checkpoint into {fname}...")
        torch.save(self.Swapper.state_dict(), f"{fname}_swapper_{step}.pt")
        torch.save(self.Swapper.state_dict(), f"{fname}_swapper_last.pt")
        torch.save(self.Id_encoder.state_dict(), f"{fname}_idc_{step}.pt")
        torch.save(self.Id_encoder.state_dict(), f"{fname}_idc_last.pt")
        if self.train_adv and self.dis is not None:
            torch.save(self.dis.state_dict(), f"{fname}_dis_{step}.pt")
            torch.save(self.dis.state_dict(), f"{fname}_dis_last.pt")

    def load(self, fname: str, step: int | None = None) -> None:
        print(f"Loading checkpoint from {fname}...")
        suffix = f"_{step}" if step is not None else "_last"
        self.Swapper.load_state_dict(torch.load(f"{fname}_swapper{suffix}.pt"))
        if self.dis is not None:
            self.dis.load_state_dict(torch.load(f"{fname}_dis{suffix}.pt"))
        if not self.fine_tune:
            self.Id_encoder.load_state_dict(torch.load(f"{fname}_idc{suffix}.pt"))

    def get_id_code(self, source: torch.Tensor) -> torch.Tensor:
        return self.Id_encoder(source).detach()

    def forward(
        self, target: torch.Tensor, source: torch.Tensor, get_latent: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if not self.fine_tune:
            source = self.Id_encoder(source)
        return self.Swapper(target, source, get_latent=get_latent)

    def add_batch_instant_norm2swapper(self) -> None:
        bn_layer = nn.BatchNorm2d(1024)
        in_layer = nn.InstanceNorm2d(1024, affine=True)
        self.Swapper.E.Encoder["layer_3"] = nn.Sequential(
            self.Swapper.E.Encoder["layer_3"],
            bn_layer,
            in_layer,
        )
        for i in range(5):
            setattr(self.Swapper.E, f"batch_norm_{i}", nn.BatchNorm2d(1024))
            setattr(self.Swapper.E, f"instance_norm_{i}", nn.InstanceNorm2d(1024, affine=True))


def build_AlphaFace(
    config: DictConfig | None = None,
    fine_tune: bool = False,
    adv_train: bool = True,
    new_id_model: bool = False,
) -> AlphaFace:
    swapper = Swapper(512)
    print("Loading pre-trained model for the ID encoder")
    if not new_id_model:
        id_encoder: nn.Module = ResNet_ID_encoder(ema_path="./models/emp.npy")
        id_encoder.resnet.load_state_dict(  # type: ignore[attr-defined]
            torch.load("./models/arcface_w600k_r50_pytorch.pt", map_location=torch.device("cuda"))
        )
    else:
        assert config is not None
        weight = torch.load(config.id_network_path)
        id_encoder = get_model(config.id_network, dropout=0, fp16=False).cuda()
        id_encoder.load_state_dict(weight)

    model = AlphaFace(swapper, id_encoder, fine_tune=fine_tune)
    if adv_train:
        model.Prepareing_adversrial_learning(img_size=256, max_conv_size=512)
        model.Preparing_VGG_percet_loss()
    return model
