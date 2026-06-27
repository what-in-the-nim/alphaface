from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvmodels


class Adaptive_instance_normalistaion(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def mu(self, x: torch.Tensor) -> torch.Tensor:
        """Takes a (n,c,h,w) tensor and returns the spatial mean as (n,c) tensor."""
        return torch.sum(x, (2, 3)) / (x.shape[2] * x.shape[3])

    def sigma(self, x: torch.Tensor) -> torch.Tensor:
        """Takes a (n,c,h,w) tensor and returns the spatial std as (n,c) tensor."""
        return torch.sqrt(
            (torch.sum((x.permute([2, 3, 0, 1]) - self.mu(x)).permute([2, 3, 0, 1]) ** 2, (2, 3)) + 0.000000023)
            / (x.shape[2] * x.shape[3])
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Transfers mean/std of style y onto content x (AdaIN)."""
        return (self.sigma(y) * ((x.permute([2, 3, 0, 1]) - self.mu(x)) / self.sigma(x)) + self.mu(y)).permute(
            [2, 3, 0, 1]
        )


class Target_Adaptive_Identify_Feature_Feeding_Block(nn.Module):
    def __init__(self, output_dim: int, z_id_size: int = 512) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.z_id_size = z_id_size
        self.fc = nn.Linear(self.z_id_size, self.output_dim)
        self.AdaIN = Adaptive_instance_normalistaion()

    def forward(self, z_id: torch.Tensor, z_target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.fc(z_id)
        out = torch.unsqueeze(out, 2)
        out = torch.unsqueeze(out, 3)
        first_1028 = out[:, 0 : int(self.output_dim / 2), :, :]
        second_1024 = out[:, int(self.output_dim / 2) : self.output_dim, :, :]
        first_1028 = (first_1028 + self.AdaIN(first_1028, z_target)) / 2.0
        second_1024 = (second_1024 + self.AdaIN(second_1024, z_target)) / 2.0
        return first_1028, second_1024


class Operation_Unit(nn.Module):
    def __init__(
        self,
        channel: int,
        identity_feature_out_dim: int,
        identity_feature_in_dim: int = 512,
        act_out: bool = True,
    ) -> None:
        super().__init__()
        self.channel = channel
        self.act_out = act_out
        self.id_feature_out_dim = identity_feature_out_dim
        self.id_feature_in_dim = identity_feature_in_dim
        self.Conv1 = nn.Conv2d(self.channel, self.channel, kernel_size=3, stride=1, padding=0)
        self.activation = nn.ReLU()
        self.IFF = Target_Adaptive_Identify_Feature_Feeding_Block(
            self.id_feature_out_dim, z_id_size=self.id_feature_in_dim
        )

    def forward(self, input_feature: torch.Tensor, id_feature: torch.Tensor) -> torch.Tensor:
        idf0_0, idf0_1 = self.IFF(id_feature, input_feature)
        x0 = F.pad(input_feature, (1, 1, 1, 1, 0, 0), mode="reflect")
        x0 = self.Conv1(x0)
        x0 = torch.sub(x0, torch.mean(x0, (2, 3), keepdim=True))

        x1 = torch.mul(x0, x0)
        x1 = torch.mean(x1, (2, 3), keepdim=True)
        x1 = torch.add(x1, 9.99999993922529e-9)
        x1 = torch.sqrt(x1)
        x1 = torch.div(1.0, x1)
        x2 = torch.mul(x0, x1)
        x2 = torch.mul(idf0_0, x2)
        x2 = torch.add(x2, idf0_1)
        if self.act_out:
            return self.activation(x2)
        return x2


class Cross_Adaptive_Identity_Injdection_Block(nn.Module):
    def __init__(self, channel: int, identity_feature_out_dim: int, identity_feature_in_dim: int) -> None:
        super().__init__()
        self.channel = channel
        self.identity_feature_out_dim = identity_feature_out_dim
        self.identity_feature_in_dim = identity_feature_in_dim

        self.OP1 = Operation_Unit(
            self.channel,
            self.identity_feature_out_dim,
            identity_feature_in_dim=self.identity_feature_in_dim,
        )
        self.OP2 = Operation_Unit(
            self.channel,
            self.identity_feature_out_dim,
            identity_feature_in_dim=self.identity_feature_in_dim,
            act_out=False,
        )

    def forward(self, input_feature: torch.Tensor, id_feature: torch.Tensor) -> torch.Tensor:
        x = self.OP1(input_feature, id_feature)
        x = self.OP2(x, id_feature)
        return input_feature + x


class Encoder_noBNIN(nn.Module):
    def __init__(self, id_feature_dim: int) -> None:
        super().__init__()
        self.id_feature_dim = id_feature_dim
        self.Encoder_channel = [3, 128, 256, 512, 1024]
        self.Encoder_kernel_size = [7, 3, 3, 3]
        self.pading_scale = [0, 1, 1, 1]
        self.stride_scale = [1, 1, 2, 2]
        self.Encoder = nn.ModuleDict(
            {
                f"layer_{i}": nn.Sequential(
                    nn.Conv2d(
                        self.Encoder_channel[i],
                        self.Encoder_channel[i + 1],
                        kernel_size=self.Encoder_kernel_size[i],
                        stride=self.stride_scale[i],
                        padding=self.pading_scale[i],
                    ),
                    nn.LeakyReLU(0.2),
                )
                for i in range(4)
            }
        )
        self.fusion_module = nn.ModuleDict(
            {f"fusion_layer_{i}": Cross_Adaptive_Identity_Injdection_Block(1024, 2048, 512) for i in range(6)}
        )

    def forward(self, x: torch.Tensor, id_feature: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (3, 3, 3, 3, 0, 0), mode="reflect")
        for i in range(4):
            x = self.Encoder[f"layer_{i}"](x)
        for i in range(6):
            x = self.fusion_module[f"fusion_layer_{i}"](x, id_feature)
        return x


class Encoder(nn.Module):
    def __init__(self, id_feature_dim: int) -> None:
        super().__init__()
        self.id_feature_dim = id_feature_dim
        self.Encoder_channel = [3, 128, 256, 512, 1024]
        self.Encoder_kernel_size = [7, 3, 3, 3]
        self.pading_scale = [0, 1, 1, 1]
        self.stride_scale = [1, 1, 2, 2]
        self.Encoder = nn.ModuleDict(
            {
                f"layer_{i}": nn.Sequential(
                    nn.Conv2d(
                        self.Encoder_channel[i],
                        self.Encoder_channel[i + 1],
                        kernel_size=self.Encoder_kernel_size[i],
                        stride=self.stride_scale[i],
                        padding=self.pading_scale[i],
                    ),
                    nn.LeakyReLU(0.2),
                )
                for i in range(4)
            }
        )
        self.fusion_module = nn.ModuleDict(
            {f"fusion_layer_{i}": Cross_Adaptive_Identity_Injdection_Block(1024, 2048, 512) for i in range(6)}
        )

    def forward(self, x: torch.Tensor, id_feature: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (3, 3, 3, 3, 0, 0), mode="reflect")
        for i in range(4):
            x = self.Encoder[f"layer_{i}"](x)
        for i in range(6):
            x = self.fusion_module[f"fusion_layer_{i}"](x, id_feature)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels: int = 1023, out_channels: int = 3) -> None:
        super().__init__()
        self.in_channel = in_channels
        self.out_channel = out_channels
        self.Upsample = nn.Upsample(scale_factor=2, align_corners=False, mode="bilinear")
        self.Conv1 = nn.Conv2d(self.in_channel, 512, kernel_size=3, stride=1, padding=1)
        self.Conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.Conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.Conv4_new = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.Conv5_new = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.Conv6_new = nn.Conv2d(128, self.out_channel, kernel_size=5, stride=1, padding=0)
        self.Activation_LeakyRelu = nn.LeakyReLU(0.2)
        self.Activation_Tanh = nn.Tanh()

    def forward(
        self, visual_feature: torch.Tensor, get_latent: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        out_latent: list[torch.Tensor] = []
        x = self.Upsample(visual_feature)
        out_latent.append(x)
        x = self.Activation_LeakyRelu(self.Conv1(x))
        out_latent.append(x)
        x = self.Upsample(x)
        x = self.Activation_LeakyRelu(self.Conv2(x))
        out_latent.append(x)
        x = self.Activation_LeakyRelu(self.Conv3(x))
        out_latent.append(x)
        x = self.Activation_LeakyRelu(self.Conv4_new(x))
        out_latent.append(x)
        x = self.Activation_LeakyRelu(self.Conv5_new(x))
        out_latent.append(x)
        x = F.pad(x, (3, 3, 3, 3, 0, 0), mode="reflect")
        x = self.Conv6_new(x)
        x = self.Activation_Tanh(x)
        x = torch.add(x, 1.0)
        x = torch.div(x, 2.0)
        if get_latent:
            return x, out_latent
        return x


class ResBlk(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        actv: nn.Module = nn.LeakyReLU(0.2),
        normalize: bool = False,
        downsample: bool = False,
    ) -> None:
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in: int, dim_out: int) -> None:
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x: torch.Tensor) -> torch.Tensor:
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self._shortcut(x) + self._residual(x)) / math.sqrt(2)


class Discriminator(nn.Module):
    def __init__(self, img_size: int = 384, max_conv_dim: int = 1024) -> None:
        super().__init__()
        dim_in = 2**14 // img_size
        num_domains = 1
        blocks: list[nn.Module] = [nn.Conv2d(3, dim_in, 3, 1, 1)]
        repeat_num = int(np.log2(img_size)) - 1
        for _l in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            if _l % 2 == 0:
                blocks += [ResBlk(dim_in, dim_out, downsample=False)]
            else:
                blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 1, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.main = nn.Sequential(*blocks)

    def get_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        feature_list: list[torch.Tensor] = []
        for _depth, block in enumerate(self.main):
            x = block(x)
            if _depth > 7 and _depth < 10:
                feature_list.append(x)
        return feature_list

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.main:
            x = block(x)
        return x.view(x.size(0), -1)


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using fixed VGG-16 feature activations."""

    def __init__(
        self,
        layer_ids: tuple[int, ...] = (3, 8, 15, 22),
        weights: tuple[float, ...] | None = None,
        criterion: str = "l1",
        resize: bool = True,
    ) -> None:
        super().__init__()
        vgg = tvmodels.vgg16(weights=tvmodels.VGG16_Weights.IMAGENET1K_V1).features
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.layer_ids = layer_ids
        self.weights = torch.ones(len(layer_ids)) if weights is None else torch.tensor(weights)

        if criterion == "l1":
            self.distance: nn.Module = nn.L1Loss(reduction="none")
        elif criterion == "l2":
            self.distance = nn.MSELoss(reduction="none")
        else:
            raise ValueError("criterion must be 'l1' or 'l2'")

        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _extract_feats(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats: list[torch.Tensor] = []
        out = x
        if self.resize:
            out = nn.functional.interpolate(out, size=(224, 224), mode="bilinear", align_corners=False)
        out = (out - self.mean) / self.std
        for i, layer in enumerate(self.vgg):
            out = layer(out)
            if i in self.layer_ids:
                feats.append(out)
            if i >= max(self.layer_ids):
                break
        return feats

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss. Both tensors must be in [0,1] range, shape (N,3,H,W)."""
        assert pred.shape == target.shape, "Input tensors must have the same shape."
        feats_pred = self._extract_feats(pred)
        feats_tgt = self._extract_feats(target)
        losses = [
            w * self.distance(f_p, f_t).mean(dim=[1, 2, 3]) for f_p, f_t, w in zip(feats_pred, feats_tgt, self.weights)
        ]
        return torch.stack(losses, dim=0).sum(dim=0).mean()
