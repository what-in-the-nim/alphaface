from __future__ import annotations

import os
from typing import Any

import clip
import hydra
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from .dataset.get_dataloader import get_dataloader_clip
from .models.swapper_alphaface import AlphaFace, build_AlphaFace
from .objectives.loss import (
    clip_text_loss,
    identity_loss,
    identity_score,
    masked_reconstruction_loss,
    multi_scale_adversarial_loss,
    reconstruction_loss,
)
from .utils.utils_distributed_sampler import setup_seed


def normalize_by_127_5(img: torch.Tensor) -> torch.Tensor:
    img = (img * 255.0).int()
    return (img / 127.5) - 1.0


def tensor2img(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor * 255.0).int()


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_with_clip(
    config: DictConfig,
    fs_model: AlphaFace,
    clip_model: Any,
    opts: list[Optimizer],
    dataloader: DataLoader,
    device: str = "cpu",
) -> None:
    start_epoch = 0
    global_step = 0
    writer = SummaryWriter(config.tb_dir)
    swapper_opt, dis_opt = opts[0], opts[1]

    scheduler_swapper = StepLR(swapper_opt, step_size=config.lr_schedule_step, gamma=0.97)
    scheduler_discriminator = StepLR(dis_opt, step_size=config.lr_schedule_step, gamma=0.97)

    os.makedirs(config.output, exist_ok=True)

    normalize_transform = transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )

    use_amp = getattr(config, "use_amp", False) and device == "cuda"
    grad_acc = max(1, getattr(config, "gradient_acc", 1))
    scaler_swapper = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler_dis = torch.amp.GradScaler("cuda", enabled=use_amp)

    if config.resume:
        print("Resuming from checkpoint...")
        path = config.model_path or os.path.join(config.output, "model_last.pt")
        print("from " + path)
        dict_checkpoint = torch.load(path, map_location=device)
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        fs_model.Swapper.load_state_dict(dict_checkpoint["swapper"])
        fs_model.dis.load_state_dict(dict_checkpoint["discriminator"])
        swapper_opt.load_state_dict(dict_checkpoint["swapper_optimizer"])
        dis_opt.load_state_dict(dict_checkpoint["discriminator_optimizer"])
        del dict_checkpoint

    fs_model = fs_model.to(device)
    fs_model.Swapper.train()
    fs_model.Id_encoder.eval()
    fs_model.set_grads()
    clip_model.eval()

    swapper_opt.zero_grad(set_to_none=True)
    dis_opt.zero_grad(set_to_none=True)

    _iter = 0  # forward-pass counter for gradient accumulation

    for epoch in range(start_epoch, config.num_epoch):
        for _, target_samples in enumerate(dataloader):
            # Packed datasets return a 12-tuple with pre-computed embeddings;
            # legacy datasets return a 6-tuple (embeddings are None).
            batch = target_samples
            if len(batch) == 12:
                (
                    img1_t,
                    img2_t,
                    mask1_t,
                    mask2_t,
                    txt1_t,
                    txt2_t,
                    pre_clip_img1,
                    pre_clip_img2,
                    pre_clip_txt1,
                    pre_clip_txt2,
                    pre_id1,
                    pre_id2,
                ) = batch
            else:
                img1_t, img2_t, mask1_t, mask2_t, txt1_t, txt2_t = batch
                pre_clip_img1 = pre_clip_img2 = pre_clip_txt1 = pre_clip_txt2 = None
                pre_id1 = pre_id2 = None

            img1_t = img1_t.to(device)
            img1_s = ((F.interpolate(img1_t, (112, 112), mode="bilinear") * 255.0) / 127.5) - 1.0
            img2_t = img2_t.to(device)
            img2_s = ((F.interpolate(img2_t, (112, 112), mode="bilinear") * 255.0) / 127.5) - 1.0
            mask1_t = mask1_t.to(device)
            mask2_t = mask2_t.to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                swapped_1_2 = fs_model(img1_t, img2_s)
                swapped_2_1 = fs_model(img2_t, img1_s)

                swapped112_1_2 = ((F.interpolate(swapped_1_2, (112, 112), mode="bilinear") * 255.0) / 127.5) - 1.0
                swapped112_2_1 = ((F.interpolate(swapped_2_1, (112, 112), mode="bilinear") * 255.0) / 127.5) - 1.0

                swapped_1_2_clip = normalize_transform(F.interpolate(swapped_1_2, (224, 224), mode="bilinear") * 255.0)
                swapped_2_1_clip = normalize_transform(F.interpolate(swapped_2_1, (224, 224), mode="bilinear") * 255.0)

                img1_2_features = clip_model.encode_image(swapped_1_2_clip)
                img2_1_features = clip_model.encode_image(swapped_2_1_clip)

                with torch.no_grad():
                    # Use pre-computed embeddings when available (packed dataset),
                    # otherwise run the frozen encoders as usual.
                    def _has(t):
                        return t is not None and t[0] is not None

                    if _has(pre_id1):
                        identity_code_1 = pre_id1.to(device, dtype=torch.float32)
                        identity_code_2 = pre_id2.to(device, dtype=torch.float32)
                    else:
                        identity_code_1 = fs_model.Id_encoder(img1_s)
                        identity_code_2 = fs_model.Id_encoder(img2_s)

                    if _has(pre_clip_img1):
                        img1_features = pre_clip_img1.to(device, dtype=torch.float32)
                        img2_features = pre_clip_img2.to(device, dtype=torch.float32)
                    else:
                        img1_t_clip = normalize_transform(F.interpolate(img1_t, (224, 224), mode="bilinear") * 255.0)
                        img2_t_clip = normalize_transform(F.interpolate(img2_t, (224, 224), mode="bilinear") * 255.0)
                        img1_features = clip_model.encode_image(img1_t_clip)
                        img2_features = clip_model.encode_image(img2_t_clip)

                    if _has(pre_clip_txt1):
                        text1_features = pre_clip_txt1.to(device, dtype=torch.float32)
                        text2_features = pre_clip_txt2.to(device, dtype=torch.float32)
                    else:
                        tokenized1 = clip.tokenize(txt1_t, context_length=77, truncate=True).to(device)
                        tokenized2 = clip.tokenize(txt2_t, context_length=77, truncate=True).to(device)
                        text1_features = clip_model.encode_text(tokenized1)
                        text2_features = clip_model.encode_text(tokenized2)

                swapped_code_1_2 = fs_model.Id_encoder(swapped112_1_2)
                swapped_code_2_1 = fs_model.Id_encoder(swapped112_2_1)

                loss_id = identity_loss(swapped_code_2_1, identity_code_1) + identity_loss(
                    swapped_code_1_2, identity_code_2
                )

                swapped_face_1_1 = fs_model(img1_t, img1_s)
                swapped_face_2_2 = fs_model(img2_t, img2_s)
                loss_self_rec = reconstruction_loss(img1_t, swapped_face_1_1) + reconstruction_loss(
                    img2_t, swapped_face_2_2
                )

                loss_percept = fs_model.feats_extractor(img1_t, swapped_1_2) + fs_model.feats_extractor(
                    img2_t, swapped_2_1
                )

                swapped_face_1_2_1 = fs_model(swapped_1_2, img1_s)
                swapped_face_2_1_2 = fs_model(swapped_2_1, img2_s)
                loss_2cycle_rec = reconstruction_loss(img1_t, swapped_face_1_2_1) + reconstruction_loss(
                    img2_t, swapped_face_2_1_2
                )

                loss_masked_recon = masked_reconstruction_loss(
                    img1_t, swapped_1_2, mask1_t
                ) + masked_reconstruction_loss(img2_t, swapped_2_1, mask2_t)

                clip_t_loss_id = identity_loss(img2_1_features, img1_features) + identity_loss(
                    img1_2_features, img2_features
                )
                clip_score_1 = identity_score(img1_features, text1_features)
                clip_score_2 = identity_score(img2_features, text2_features)
                clip_text2img_loss = clip_text_loss(img2_1_features, text2_features, clip_score_2) + clip_text_loss(
                    img1_2_features, text1_features, clip_score_1
                )

                base_gen_loss = (
                    config.w_id * loss_id
                    + config.w_self_rec * loss_self_rec
                    + config.w_percept * loss_percept
                    + config.w_2cycle * loss_2cycle_rec
                    + config.w_mask_rec * loss_masked_recon
                    + config.w_clip_id * clip_t_loss_id
                    + config.w_clip_text * clip_text2img_loss
                )

            if global_step <= config.adv_sess:
                total_gen_loss = base_gen_loss
                # Generator backward — disc is not yet active so no contamination concern
                scaler_swapper.scale(total_gen_loss / grad_acc).backward()
            else:
                # Adversarial phase: freeze disc params so gen backward doesn't pollute disc grads
                for p in fs_model.dis.parameters():
                    p.requires_grad_(False)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    disc_gen_output_1_2 = fs_model.dis(swapped_1_2)
                    disc_gen_output_2_1 = fs_model.dis(swapped_2_1)
                    loss_adv_gen = multi_scale_adversarial_loss(
                        disc_gen_output_1_2, is_real=True
                    ) + multi_scale_adversarial_loss(disc_gen_output_2_1, is_real=True)
                    total_gen_loss = base_gen_loss + config.w_gen_adv * loss_adv_gen

                scaler_swapper.scale(total_gen_loss / grad_acc).backward()

                for p in fs_model.dis.parameters():
                    p.requires_grad_(True)

                # Discriminator backward — accumulates cleanly across grad_acc steps
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss_disc_real = multi_scale_adversarial_loss(
                        fs_model.dis(img1_t), is_real=True
                    ) + multi_scale_adversarial_loss(fs_model.dis(img2_t), is_real=True)
                    loss_disc_fake = multi_scale_adversarial_loss(
                        fs_model.dis(swapped_1_2.detach()), is_real=False
                    ) + multi_scale_adversarial_loss(fs_model.dis(swapped_2_1.detach()), is_real=False)
                    total_disc_loss = (loss_disc_real + loss_disc_fake) / 2

                scaler_dis.scale(total_disc_loss / grad_acc).backward()

            _iter += 1
            if _iter % grad_acc != 0:
                continue

            # ---- optimizer step ----
            scaler_swapper.unscale_(swapper_opt)
            torch.nn.utils.clip_grad_norm_(fs_model.Swapper.parameters(), max_norm=1.0)
            scaler_swapper.step(swapper_opt)
            scaler_swapper.update()
            swapper_opt.zero_grad(set_to_none=True)
            scheduler_swapper.step()

            if global_step > config.adv_sess:
                scaler_dis.unscale_(dis_opt)
                torch.nn.utils.clip_grad_norm_(fs_model.dis.parameters(), max_norm=1.0)
                scaler_dis.step(dis_opt)
                scaler_dis.update()
                dis_opt.zero_grad(set_to_none=True)
                scheduler_discriminator.step()

            if global_step % 100 == 0:
                dis_lr = dis_opt.param_groups[-1]["lr"]
                gen_lr = swapper_opt.param_groups[-1]["lr"]
                if global_step <= config.adv_sess:
                    print(
                        f"global step {global_step}, Generator Loss: {total_gen_loss.item()}, Discriminator Loss: Not applied"
                    )
                else:
                    print(
                        f"global step {global_step}, Generator Loss: {total_gen_loss.item()}, Discriminator Loss: {total_disc_loss.item()}"
                    )
                writer.add_scalar("0.Total_loss_for_swapper/train", total_gen_loss, global_step)
                writer.add_scalar("1.Identity_loss/train", loss_id, global_step)
                writer.add_scalar("2.Self_recon_loss/train", loss_self_rec, global_step)
                writer.add_scalar("3.Perceptual_loss/train", loss_percept, global_step)
                writer.add_scalar("4.2Cyclic_recon_loss/train", loss_2cycle_rec, global_step)
                writer.add_scalar("5.Masked_recon_loss/train", loss_masked_recon, global_step)
                writer.add_scalar("8.[Clip-learning] CLIP_ID Similarity loss/train", clip_t_loss_id, global_step)
                writer.add_scalar(
                    "9.[Clip-learning] CLIP_img-text_contrastive_learning_loss/train", clip_text2img_loss, global_step
                )
                writer.add_scalar("[-].Learning_rate_swapper/train", gen_lr, global_step)
                if global_step > config.adv_sess:
                    writer.add_scalar("8.Adv_gen_Loss/train", loss_adv_gen, global_step)
                    writer.add_scalar("9.Adv_disc_Loss/train", total_disc_loss, global_step)
                    writer.add_scalar("[-].Discriminator_learning_rate/train", dis_lr, global_step)

                if config.visualize:
                    writer.add_image("1_Target_attribute_image", make_grid(img1_t), global_step)
                    writer.add_image("2_Target_mask", make_grid(mask1_t), global_step)
                    writer.add_image("3_Source_identity_image", make_grid(img2_t), global_step)
                    writer.add_image("4_source_mask", make_grid(mask2_t), global_step)
                    writer.add_image("5_Swapped_results", make_grid(swapped_1_2), global_step)

            if global_step % config.save_interval == 0:
                gen_lr = swapper_opt.param_groups[-1]["lr"]
                dis_lr = dis_opt.param_groups[-1]["lr"]
                checkpoint = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "swapper_lr": gen_lr,
                    "dis_lr": dis_lr,
                    "swapper": fs_model.Swapper.state_dict(),
                    "discriminator": fs_model.dis.state_dict(),
                    "swapper_optimizer": swapper_opt.state_dict(),
                    "discriminator_optimizer": dis_opt.state_dict(),
                }
                torch.save(checkpoint, os.path.join(config.output, f"model_{global_step}.pt"))
                torch.save(checkpoint, os.path.join(config.output, "model_last.pt"))

            global_step += 1


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    device = _get_device()
    print(f"Using device: {device}")
    setup_seed(seed=cfg.seed, cuda_deterministic=False)
    os.makedirs(cfg.log_dir, exist_ok=True)

    print(cfg.num_workers)
    train_loader = get_dataloader_clip(cfg.db_path, cfg.batch_size, cfg.num_workers)

    print("Preparing the AlphaFace model")
    alpha_face = build_AlphaFace(config=cfg, new_id_model=False, device=device)

    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)

    use_fused = device == "cuda" and torch.__version__ >= "2.0"

    if cfg.optimizer == "sgd":
        swapper_opt: Optimizer = torch.optim.SGD(
            [{"params": alpha_face.Swapper.parameters()}],
            lr=cfg.init_lr_swapper,
            momentum=0.9,
            weight_decay=cfg.weight_decay,
        )
        dis_opt: Optimizer = torch.optim.SGD(
            [{"params": alpha_face.dis.parameters()}],
            lr=cfg.init_lr_dis,
            momentum=0.9,
            weight_decay=cfg.weight_decay,
        )
    elif cfg.optimizer == "adamw":
        _adamw_kwargs: dict = dict(betas=(0.0, 0.99), weight_decay=cfg.weight_decay)
        if use_fused:
            _adamw_kwargs["fused"] = True
        swapper_opt = torch.optim.AdamW(
            [{"params": alpha_face.Swapper.parameters()}],
            lr=cfg.init_lr_swapper,
            **_adamw_kwargs,
        )
        dis_opt = torch.optim.AdamW(
            [{"params": alpha_face.dis.parameters()}],
            lr=cfg.init_lr_dis,
            **_adamw_kwargs,
        )
    elif cfg.optimizer == "adam":
        swapper_opt = torch.optim.Adam(
            [{"params": alpha_face.Swapper.parameters()}],
            lr=cfg.init_lr_swapper,
            betas=(0.0, 0.99),
            weight_decay=cfg.weight_decay,
        )
        dis_opt = torch.optim.Adam(
            [{"params": alpha_face.dis.parameters()}],
            lr=cfg.init_lr_dis,
            betas=(0.0, 0.99),
            weight_decay=cfg.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    use_compile = device == "cuda" and hasattr(torch, "compile") and not getattr(cfg, "use_checkpoint", False)
    if use_compile:
        print("Compiling Swapper and Discriminator with torch.compile...")
        alpha_face.Swapper = torch.compile(alpha_face.Swapper, mode="reduce-overhead")
        alpha_face.dis = torch.compile(alpha_face.dis, mode="reduce-overhead")

    train_with_clip(cfg, alpha_face, clip_model, [swapper_opt, dis_opt], train_loader, device=device)


if __name__ == "__main__":
    main()
