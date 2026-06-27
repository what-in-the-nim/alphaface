from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_msssim import ssim


def identity_loss(embedding_generated: torch.Tensor, embedding_source: torch.Tensor) -> torch.Tensor:
    loss = 1 - F.cosine_similarity(embedding_source, embedding_generated, dim=1)
    return loss.mean()


def clip_text_loss(
    embedding_generated: torch.Tensor,
    embedding_source: torch.Tensor,
    clip_score: torch.Tensor,
) -> torch.Tensor:
    cos_sim = F.cosine_similarity(embedding_source, embedding_generated, dim=1)
    loss = clip_score - cos_sim
    loss = torch.where(0 < loss, loss, torch.zeros_like(loss))
    return loss.sum()


def identity_score(embedding_generated: torch.Tensor, embedding_source: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(embedding_source, embedding_generated, dim=1)


def reconstruction_loss(generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(generated, target)


def masked_reconstruction_loss(
    generated: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor
) -> torch.Tensor:
    return F.l1_loss(target_mask * generated, target_mask * target)


def structural_similarity_loss(generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1 - ssim(generated, target, data_range=1.0)


def structural_similarity_score(generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ssim(generated, target, data_range=1.0)


def multi_scale_adversarial_loss(discriminator_output: torch.Tensor, is_real: bool = True) -> torch.Tensor:
    """MSE adversarial loss for generator or discriminator."""
    target = torch.ones_like(discriminator_output) if is_real else torch.zeros_like(discriminator_output)
    return F.mse_loss(discriminator_output, target)


def adversarial_loss(discriminator_output: torch.Tensor, is_real: bool = True) -> torch.Tensor:
    """BCE adversarial loss for generator or discriminator."""
    target = torch.ones_like(discriminator_output) if is_real else torch.zeros_like(discriminator_output)
    return F.binary_cross_entropy_with_logits(discriminator_output, target)


def hinge_discriminator_loss(D_real: torch.Tensor, D_fake: torch.Tensor) -> torch.Tensor:
    """Hinge loss for discriminator."""
    return torch.mean(F.relu(1.0 - D_real)) + torch.mean(F.relu(1.0 + D_fake))


def hinge_generator_loss(D_fake: torch.Tensor) -> torch.Tensor:
    """Hinge loss for generator."""
    return -torch.mean(D_fake)


def gradient_penalty(
    discriminator: nn.Module,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: torch.device,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    batch_size = real_images.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = epsilon * real_images + (1 - epsilon) * fake_images
    interpolated.requires_grad_(True)
    d_interpolated = discriminator(interpolated)
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradient_norm = gradients.view(batch_size, -1).norm(2, dim=1)
    return lambda_gp * ((gradient_norm - 1) ** 2).mean()


def attribute_preservation_loss(
    features_generated: list[torch.Tensor], features_target: list[torch.Tensor]
) -> torch.Tensor:
    loss = torch.tensor(0.0)
    for feat_gen, feat_tgt in zip(features_generated, features_target):
        loss = loss + F.l1_loss(feat_gen, feat_tgt, reduction="mean")
    return loss


def masked_attribute_preservation_loss(
    features_generated: list[torch.Tensor],
    features_target: list[torch.Tensor],
    mask: torch.Tensor,
) -> torch.Tensor:
    loss = torch.tensor(0.0)
    for feat_gen, feat_tgt in zip(features_generated, features_target):
        _shape = feat_gen.shape
        reshaped_mask = F.interpolate(mask, (_shape[2], _shape[3]), mode="bilinear", align_corners=True)
        loss = loss + F.l1_loss(
            reshaped_mask[:, 0:1, :, :] * feat_gen,
            reshaped_mask[:, 0:1, :, :] * feat_tgt,
            reduction="mean",
        )
    return loss


class PerceptualLoss(nn.Module):
    def __init__(self, feature_extractor: nn.Module) -> None:
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = feature_extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        features_gen = self.feature_extractor(generated)
        features_tgt = self.feature_extractor(target)
        return F.l1_loss(features_gen, features_tgt)
