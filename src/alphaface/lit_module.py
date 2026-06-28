from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import clip
import pytorch_lightning as L
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import make_grid

from .models.swapper_alphaface import AlphaFace, build_alpha_face
from .objectives.loss import (
    clip_text_loss,
    identity_loss,
    identity_score,
    masked_reconstruction_loss,
    multi_scale_adversarial_loss,
    reconstruction_loss,
)


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _config_namespace(config: Any) -> Any:
    if isinstance(config, Mapping):
        return SimpleNamespace(**config)
    return config


class AlphaFaceLitModule(L.LightningModule):
    """Lightning wrapper around :class:`AlphaFace` implementing the CLIP-guided GAN loop.

    Uses manual optimization: the model has two optimizers (swapper + discriminator) and
    an adversarial phase switch at ``cfg.adv_sess``, so gradient accumulation and clipping
    are handled here rather than by the Trainer.
    """

    def __init__(self, config: dict[str, Any], device: str = "auto") -> None:
        super().__init__()
        if device == "auto":
            device = _select_device()
        self.automatic_optimization = False
        self.cfg = _config_namespace(config)
        self.save_hyperparameters(config)

        print(f"Using device: {device}")
        self.model: AlphaFace = build_alpha_face(config=self.cfg, device=device)

        clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        self.clip_model = clip_model

        self.clip_normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )
        self._fwd_iter = 0  # forward-pass counter for gradient accumulation
        self.model.set_grads()

    def train(self, mode: bool = True) -> AlphaFaceLitModule:
        """Keep the frozen encoders (ID, CLIP, VGG) in eval mode regardless of Trainer
        toggling train mode — their BatchNorm running stats must not be updated."""
        super().train(mode)
        self.clip_model.eval()
        self.model.id_encoder.eval()
        if self.model.perceptual_loss is not None:
            self.model.perceptual_loss.eval()
        return self

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _to_source(img_t: torch.Tensor) -> torch.Tensor:
        """112x112 source image normalized to [-1, 1] by 127.5."""
        return ((F.interpolate(img_t, (112, 112), mode="bilinear") * 255.0) / 127.5) - 1.0

    def _to_clip(self, img: torch.Tensor) -> torch.Tensor:
        return self.clip_normalize(F.interpolate(img, (224, 224), mode="bilinear") * 255.0)

    # ------------------------------------------------------------------ optimizers
    def configure_optimizers(self) -> Any:
        cfg = self.cfg
        swapper_params = [{"params": self.model.swapper.parameters()}]
        dis_params = [{"params": self.model.discriminator.parameters()}]

        if cfg.optimizer == "sgd":
            swapper_opt: torch.optim.Optimizer = torch.optim.SGD(
                swapper_params, lr=cfg.init_lr_swapper, momentum=0.9, weight_decay=cfg.weight_decay
            )
            dis_opt: torch.optim.Optimizer = torch.optim.SGD(
                dis_params, lr=cfg.init_lr_dis, momentum=0.9, weight_decay=cfg.weight_decay
            )
        elif cfg.optimizer == "adamw":
            swapper_opt = torch.optim.AdamW(
                swapper_params, lr=cfg.init_lr_swapper, betas=(0.0, 0.99), weight_decay=cfg.weight_decay
            )
            dis_opt = torch.optim.AdamW(
                dis_params, lr=cfg.init_lr_dis, betas=(0.0, 0.99), weight_decay=cfg.weight_decay
            )
        elif cfg.optimizer == "adam":
            swapper_opt = torch.optim.Adam(
                swapper_params, lr=cfg.init_lr_swapper, betas=(0.0, 0.99), weight_decay=cfg.weight_decay
            )
            dis_opt = torch.optim.Adam(dis_params, lr=cfg.init_lr_dis, betas=(0.0, 0.99), weight_decay=cfg.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

        lr_schedule_gamma = getattr(cfg, "lr_schedule_gamma", 0.97)
        scheduler_swapper = torch.optim.lr_scheduler.StepLR(
            swapper_opt, step_size=cfg.lr_schedule_step, gamma=lr_schedule_gamma
        )
        scheduler_dis = torch.optim.lr_scheduler.StepLR(
            dis_opt, step_size=cfg.lr_schedule_step, gamma=lr_schedule_gamma
        )
        return [swapper_opt, dis_opt], [scheduler_swapper, scheduler_dis]

    def on_train_epoch_end(self) -> None:
        for scheduler in self.lr_schedulers():
            scheduler.step()

    # ------------------------------------------------------------------ training
    def training_step(self, batch: Any, batch_idx: int) -> None:
        cfg = self.cfg
        model = self.model
        clip_model = self.clip_model
        grad_acc = max(1, getattr(cfg, "gradient_acc", 1))

        if len(batch) != 12:
            raise ValueError(f"AlphaFace training requires packed PNG samples; expected 12 fields, got {len(batch)}")

        (
            img1_t,
            img2_t,
            mask1_t,
            mask2_t,
            _txt1_t,
            _txt2_t,
            pre_clip_img1,
            pre_clip_img2,
            pre_clip_txt1,
            pre_clip_txt2,
            pre_id1,
            pre_id2,
        ) = batch

        img1_s = self._to_source(img1_t)
        img2_s = self._to_source(img2_t)

        with torch.no_grad():
            identity_code_1 = pre_id1.to(dtype=torch.float32)
            identity_code_2 = pre_id2.to(dtype=torch.float32)
            img1_features = pre_clip_img1.to(dtype=torch.float32)
            img2_features = pre_clip_img2.to(dtype=torch.float32)
            text1_features = pre_clip_txt1.to(dtype=torch.float32)
            text2_features = pre_clip_txt2.to(dtype=torch.float32)

        # Cross-swap — retained for cycle, perceptual, clip, and (later) disc.
        swapped_1_2 = model(img1_t, img2_s)
        swapped_2_1 = model(img2_t, img1_s)

        # ID loss: encode swapped faces at 112px, free the 112-sized tensors immediately.
        swapped_code_1_2 = model.id_encoder(self._to_source(swapped_1_2))
        swapped_code_2_1 = model.id_encoder(self._to_source(swapped_2_1))
        loss_id = identity_loss(swapped_code_2_1, identity_code_1) + identity_loss(swapped_code_1_2, identity_code_2)
        del swapped_code_1_2, swapped_code_2_1

        # Self-reconstruction: compute and immediately discard outputs.
        swapped_face_1_1 = model(img1_t, img1_s)
        swapped_face_2_2 = model(img2_t, img2_s)
        loss_self_rec = reconstruction_loss(img1_t, swapped_face_1_1) + reconstruction_loss(img2_t, swapped_face_2_2)
        del swapped_face_1_1, swapped_face_2_2

        # Perceptual loss — uses cross-swap outputs in-place.
        loss_percept = model.perceptual_loss(img1_t, swapped_1_2) + model.perceptual_loss(img2_t, swapped_2_1)

        # Cycle-consistency: feed cross-swap back through model, free cycle outputs.
        swapped_face_1_2_1 = model(swapped_1_2, img1_s)
        swapped_face_2_1_2 = model(swapped_2_1, img2_s)
        loss_2cycle_rec = reconstruction_loss(img1_t, swapped_face_1_2_1) + reconstruction_loss(
            img2_t, swapped_face_2_1_2
        )
        del swapped_face_1_2_1, swapped_face_2_1_2

        # Masked reconstruction — still uses cross-swap outputs.
        loss_masked_recon = masked_reconstruction_loss(img1_t, swapped_1_2, mask1_t) + masked_reconstruction_loss(
            img2_t, swapped_2_1, mask2_t
        )

        # CLIP losses — encode swapped faces, then free CLIP tensors.
        img1_2_features = clip_model.encode_image(self._to_clip(swapped_1_2))
        img2_1_features = clip_model.encode_image(self._to_clip(swapped_2_1))
        clip_t_loss_id = identity_loss(img2_1_features, img1_features) + identity_loss(img1_2_features, img2_features)
        clip_score_1 = identity_score(img1_features, text1_features)
        clip_score_2 = identity_score(img2_features, text2_features)
        clip_text2img_loss = clip_text_loss(img2_1_features, text2_features, clip_score_2) + clip_text_loss(
            img1_2_features, text1_features, clip_score_1
        )
        del img1_2_features, img2_1_features, clip_score_1, clip_score_2

        base_gen_loss = (
            cfg.w_id * loss_id
            + cfg.w_self_rec * loss_self_rec
            + cfg.w_percept * loss_percept
            + cfg.w_2cycle * loss_2cycle_rec
            + cfg.w_mask_rec * loss_masked_recon
            + cfg.w_clip_id * clip_t_loss_id
            + cfg.w_clip_text * clip_text2img_loss
        )

        swapper_opt, dis_opt = self.optimizers()
        adversarial = self.global_step > cfg.adv_sess
        total_disc_loss = None
        loss_adv_gen = None

        if not adversarial:
            total_gen_loss = base_gen_loss
            self.manual_backward(total_gen_loss / grad_acc)
            del swapped_1_2, swapped_2_1
        else:
            # Freeze disc params so the generator backward doesn't pollute disc grads.
            for p in model.discriminator.parameters():
                p.requires_grad_(False)
            disc_gen_output_1_2 = model.discriminator(swapped_1_2)
            disc_gen_output_2_1 = model.discriminator(swapped_2_1)
            loss_adv_gen = multi_scale_adversarial_loss(
                disc_gen_output_1_2, is_real=True
            ) + multi_scale_adversarial_loss(disc_gen_output_2_1, is_real=True)
            del disc_gen_output_1_2, disc_gen_output_2_1
            total_gen_loss = base_gen_loss + cfg.w_gen_adv * loss_adv_gen
            self.manual_backward(total_gen_loss / grad_acc)
            for p in model.discriminator.parameters():
                p.requires_grad_(True)

            loss_disc_real = multi_scale_adversarial_loss(
                model.discriminator(img1_t), is_real=True
            ) + multi_scale_adversarial_loss(model.discriminator(img2_t), is_real=True)
            loss_disc_fake = multi_scale_adversarial_loss(
                model.discriminator(swapped_1_2.detach()), is_real=False
            ) + multi_scale_adversarial_loss(model.discriminator(swapped_2_1.detach()), is_real=False)
            del swapped_1_2, swapped_2_1
            total_disc_loss = (loss_disc_real + loss_disc_fake) / 2
            self.manual_backward(total_disc_loss / grad_acc)

        self._fwd_iter += 1
        if self._fwd_iter % grad_acc == 0:
            self.clip_gradients(swapper_opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            swapper_opt.step()
            swapper_opt.zero_grad(set_to_none=True)

            if adversarial:
                self.clip_gradients(dis_opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
                dis_opt.step()
                dis_opt.zero_grad(set_to_none=True)

        # ---- logging ----
        self.log("train/total_gen_loss", total_gen_loss, on_step=True, prog_bar=True)
        self.log("train/identity_loss", loss_id, on_step=True)
        self.log("train/self_recon_loss", loss_self_rec, on_step=True)
        self.log("train/perceptual_loss", loss_percept, on_step=True)
        self.log("train/cycle_recon_loss", loss_2cycle_rec, on_step=True)
        self.log("train/masked_recon_loss", loss_masked_recon, on_step=True)
        self.log("train/clip_id_loss", clip_t_loss_id, on_step=True)
        self.log("train/clip_text_loss", clip_text2img_loss, on_step=True)
        self.log("train/lr_swapper", swapper_opt.param_groups[-1]["lr"], on_step=True)
        if adversarial and total_disc_loss is not None:
            self.log("train/adv_gen_loss", loss_adv_gen, on_step=True)
            self.log("train/adv_disc_loss", total_disc_loss, on_step=True)
            self.log("train/lr_discriminator", dis_opt.param_groups[-1]["lr"], on_step=True)

        if getattr(cfg, "visualize", False) and self.global_step % 100 == 0:
            exp = getattr(self.logger, "experiment", None)
            if exp is not None and hasattr(exp, "add_image"):
                exp.add_image("1_target_attribute_image", make_grid(img1_t), self.global_step)
                exp.add_image("2_target_mask", make_grid(mask1_t), self.global_step)
                exp.add_image("3_source_identity_image", make_grid(img2_t), self.global_step)
                exp.add_image("4_source_mask", make_grid(mask2_t), self.global_step)
                exp.add_image("5_swapped_result", make_grid(swapped_1_2), self.global_step)

    # ------------------------------------------------------------------ predict
    def predict_step(self, batch: Any, batch_idx: int = 0, dataloader_idx: int = 0) -> torch.Tensor:
        target, source = batch
        return self.model(target, source)
