from __future__ import annotations

import argparse
import os
from typing import Any

import clip
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from dataset.get_dataloader import get_dataloader_clip
from models.swapper_alphaface import AlphaFace, build_AlphaFace
from objectives.loss import (
    clip_text_loss,
    identity_loss,
    identity_score,
    masked_reconstruction_loss,
    multi_scale_adversarial_loss,
    reconstruction_loss,
)
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed


def normalize_by_127_5(img: torch.Tensor) -> torch.Tensor:
    img = (img * 255.0).int()
    return (img / 127.5) - 1.0


def tensor2img(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor * 255.0).int()


def train_with_clip(
    config: Any,
    fs_model: AlphaFace,
    clip_model: Any,
    opts: list[Optimizer],
    dataloader: DataLoader,
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

    if config.resume:
        print("Resuming from checkpoint...")
        path = config.model_path or os.path.join(config.output, "model_last.pt")
        print("from " + path)
        dict_checkpoint = torch.load(path)
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        fs_model.Swapper.load_state_dict(dict_checkpoint["swapper"])
        fs_model.dis.load_state_dict(dict_checkpoint["discriminator"])
        swapper_opt.load_state_dict(dict_checkpoint["swapper_optimizer"])
        dis_opt.load_state_dict(dict_checkpoint["discriminator_optimizer"])
        del dict_checkpoint

    fs_model = fs_model.cuda()
    fs_model.Swapper.train()
    fs_model.Id_encoder.eval()
    fs_model.set_grads()
    clip_model.eval()

    for epoch in range(start_epoch, config.num_epoch):
        for _, target_samples in enumerate(dataloader):
            img1_t, img2_t, mask1_t, mask2_t, txt1_t, txt2_t = target_samples
            img1_t = img1_t.cuda()
            img1_s = ((F.interpolate(img1_t, (112, 112), mode="bilinear") * 255.0) / 127.5) - 1.0
            img2_t = img2_t.cuda()
            img2_s = ((F.interpolate(img2_t, (112, 112), mode="bilinear") * 255.0) / 127.5) - 1.0
            mask1_t = mask1_t.cuda()
            mask2_t = mask2_t.cuda()

            swapped_1_2 = fs_model(img1_t, img2_s)
            swapped_2_1 = fs_model(img2_t, img1_s)

            swapped112_1_2 = ((F.interpolate(swapped_1_2, (112, 112), mode="bilinear") * 255.0) / 127.5) - 1.0
            swapped112_2_1 = ((F.interpolate(swapped_2_1, (112, 112), mode="bilinear") * 255.0) / 127.5) - 1.0

            tokenized1 = clip.tokenize(txt1_t, context_length=77, truncate=True).cuda()
            tokenized2 = clip.tokenize(txt2_t, context_length=77, truncate=True).cuda()

            img1_t_clip = normalize_transform(F.interpolate(img1_t, (224, 224), mode="bilinear") * 255.0)
            img2_t_clip = normalize_transform(F.interpolate(img2_t, (224, 224), mode="bilinear") * 255.0)
            swapped_1_2_clip = normalize_transform(F.interpolate(swapped_1_2, (224, 224), mode="bilinear") * 255.0)
            swapped_2_1_clip = normalize_transform(F.interpolate(swapped_2_1, (224, 224), mode="bilinear") * 255.0)

            img1_2_features = clip_model.encode_image(swapped_1_2_clip)
            img2_1_features = clip_model.encode_image(swapped_2_1_clip)

            with torch.no_grad():
                identity_code_1 = fs_model.Id_encoder(img1_s)
                identity_code_2 = fs_model.Id_encoder(img2_s)
                img1_features = clip_model.encode_image(img1_t_clip)
                img2_features = clip_model.encode_image(img2_t_clip)
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
            clip_text2img_loss = clip_text_loss(
                img2_1_features, text2_features, clip_score_2
            ) + clip_text_loss(img1_2_features, text1_features, clip_score_1)

            if global_step <= config.adv_sess:
                total_gen_loss = (
                    config.w_id * loss_id
                    + config.w_self_rec * loss_self_rec
                    + config.w_percept * loss_percept
                    + config.w_2cycle * loss_2cycle_rec
                    + config.w_mask_rec * loss_masked_recon
                    + config.w_clip_id * clip_t_loss_id
                    + config.w_clip_text * clip_text2img_loss
                )
                swapper_opt.zero_grad()
                total_gen_loss.backward()
                torch.nn.utils.clip_grad_norm_(fs_model.Swapper.parameters(), max_norm=1.0)
                swapper_opt.step()
                scheduler_swapper.step()
            else:
                disc_gen_output_1_2 = fs_model.dis(swapped_1_2)
                disc_gen_output_2_1 = fs_model.dis(swapped_2_1)
                loss_adv_gen = multi_scale_adversarial_loss(
                    disc_gen_output_1_2, is_real=True
                ) + multi_scale_adversarial_loss(disc_gen_output_2_1, is_real=True)

                total_gen_loss = (
                    config.w_id * loss_id
                    + config.w_self_rec * loss_self_rec
                    + config.w_percept * loss_percept
                    + config.w_2cycle * loss_2cycle_rec
                    + config.w_mask_rec * loss_masked_recon
                    + config.w_clip_id * clip_t_loss_id
                    + config.w_clip_text * clip_text2img_loss
                    + config.w_gen_adv * loss_adv_gen
                )
                swapper_opt.zero_grad()
                total_gen_loss.backward()
                torch.nn.utils.clip_grad_norm_(fs_model.Swapper.parameters(), max_norm=1.0)
                swapper_opt.step()

                loss_disc_real = multi_scale_adversarial_loss(
                    fs_model.dis(img1_t), is_real=True
                ) + multi_scale_adversarial_loss(fs_model.dis(img2_t), is_real=True)
                loss_disc_fake = multi_scale_adversarial_loss(
                    fs_model.dis(swapped_1_2.detach()), is_real=False
                ) + multi_scale_adversarial_loss(fs_model.dis(swapped_2_1.detach()), is_real=False)
                total_disc_loss = (loss_disc_real + loss_disc_fake) / 2

                dis_opt.zero_grad()
                total_disc_loss.backward()
                torch.nn.utils.clip_grad_norm_(fs_model.dis.parameters(), max_norm=1.0)
                dis_opt.step()
                scheduler_swapper.step()
                scheduler_discriminator.step()

            if global_step % 100 == 0:
                dis_lr = dis_opt.param_groups[-1]["lr"]
                gen_lr = swapper_opt.param_groups[-1]["lr"]
                if global_step <= config.adv_sess:
                    print(f"global step {global_step}, Generator Loss: {total_gen_loss.item()}, Discriminator Loss: Not applied")
                else:
                    print(f"global step {global_step}, Generator Loss: {total_gen_loss.item()}, Discriminator Loss: {total_disc_loss.item()}")
                writer.add_scalar("0.Total_loss_for_swapper/train", total_gen_loss, global_step)
                writer.add_scalar("1.Identity_loss/train", loss_id, global_step)
                writer.add_scalar("2.Self_recon_loss/train", loss_self_rec, global_step)
                writer.add_scalar("3.Perceptual_loss/train", loss_percept, global_step)
                writer.add_scalar("4.2Cyclic_recon_loss/train", loss_2cycle_rec, global_step)
                writer.add_scalar("5.Masked_recon_loss/train", loss_masked_recon, global_step)
                writer.add_scalar("8.[Clip-learning] CLIP_ID Similarity loss/train", clip_t_loss_id, global_step)
                writer.add_scalar("9.[Clip-learning] CLIP_img-text_contrastive_learning_loss/train", clip_text2img_loss, global_step)
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


def main(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_config(args.config)
    setup_seed(seed=cfg.seed, cuda_deterministic=False)
    os.makedirs(cfg.log_dir, exist_ok=True)

    print(cfg.num_workers)
    train_loader = get_dataloader_clip(cfg.db_path, cfg.batch_size, cfg.num_workers)

    print("Preparing the AlphaFace model")
    alpha_face = build_AlphaFace(config=cfg).cuda()

    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)

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
        swapper_opt = torch.optim.AdamW(
            [{"params": alpha_face.Swapper.parameters()}],
            lr=cfg.init_lr_swapper,
            betas=(0.0, 0.99),
            weight_decay=cfg.weight_decay,
        )
        dis_opt = torch.optim.AdamW(
            [{"params": alpha_face.dis.parameters()}],
            lr=cfg.init_lr_dis,
            betas=(0.0, 0.99),
            weight_decay=cfg.weight_decay,
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

    train_with_clip(cfg, alpha_face, clip_model, [swapper_opt, dis_opt], train_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaFace CLIP training")
    parser.add_argument("config", type=str, default="./configs/test_config", help="py config file")
    main(parser.parse_args())
