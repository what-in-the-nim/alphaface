"""Per-stage pipeline functions for the alphaface dataset CLI."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .align import AlignedFace, FaceAligner
from .caption import FaceCaptioner
from .mask import FaceMasker
from .pack_png import update_png_alpha, update_png_chunks

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
log = logging.getLogger(__name__)


def _find_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in _IMAGE_EXTENSIONS)


def _output_stem(source_root: Path, image_path: Path, index: int) -> str:
    rel_parts = image_path.relative_to(source_root).with_suffix("").parts
    flat = "_".join(rel_parts)
    return f"{index:06d}_{flat}"


def run_extract(
    input_dir: Path,
    output_dir: Path,
    device: str = "cpu",
    output_size: int = 256,
    largest_only: bool = True,
    force: bool = False,
) -> None:
    images = _find_images(input_dir)
    if not images:
        sys.exit(f"No images found under {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Found %d source images", len(images))

    aligner: FaceAligner | None = None
    skipped = written = 0
    global_idx = 0

    for image_path in tqdm(images, desc="Extracting", unit="img"):
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            log.warning("Cannot read %s — skipped", image_path)
            skipped += 1
            continue

        flat = "_".join(image_path.relative_to(input_dir).with_suffix("").parts)
        existing = sorted(output_dir.glob(f"??????_{flat}.png"))
        if existing and not force:
            global_idx += len(existing)
            continue

        if aligner is None:
            aligner = FaceAligner(device=device)

        aligned_faces = (
            [aligner.align_largest(bgr, output_size)] if largest_only else aligner.align_all(bgr, output_size)
        )
        aligned_faces = [f for f in aligned_faces if f is not None]

        if not aligned_faces:
            log.debug("No face detected in %s", image_path)
            skipped += 1
            continue

        for af in aligned_faces:
            stem = _output_stem(input_dir, image_path, global_idx)
            global_idx += 1
            img_rgb = cv2.cvtColor(af.image, cv2.COLOR_BGR2RGB)
            Image.fromarray(img_rgb, "RGB").save(str(output_dir / f"{stem}.png"), optimize=False)
            written += 1

    log.info("Extract complete — written: %d  skipped: %d", written, skipped)


def run_mask(
    folder: Path,
    device: str = "cpu",
    mask_model: str = "weights/resnet34.onnx",
    force: bool = False,
) -> None:
    pngs = sorted(folder.glob("*.png"))
    if not pngs:
        sys.exit(f"No PNGs found in {folder}")

    masker = FaceMasker(model_path=mask_model, device=device)
    done = skipped = 0

    for path in tqdm(pngs, desc="Masking", unit="img"):
        pil = Image.open(str(path))
        if pil.mode == "RGBA" and not force:
            skipped += 1
            continue
        rgb = np.array(pil.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        af = AlignedFace(image=bgr, landmarks_68=np.zeros((68, 2), dtype=np.float32))
        mask = masker(af)
        update_png_alpha(path, mask)
        done += 1

    log.info("Mask complete — done: %d  skipped (already RGBA): %d", done, skipped)


def run_caption(
    folder: Path,
    caption_url: str = "http://localhost:11434/v1",
    caption_model: str = "llava:7b",
    caption_api_key: str = "none",
    caption_concurrency: int = 4,
    caption_timeout: float = 60.0,
    force: bool = False,
) -> None:
    pngs = sorted(folder.glob("*.png"))
    if not pngs:
        sys.exit(f"No PNGs found in {folder}")

    pending = []
    for path in tqdm(pngs, desc="Scanning captions", unit="img"):
        pil = Image.open(str(path))
        if not force and pil.text.get("alphaface_caption", "").strip():
            continue
        pending.append(path)

    if not pending:
        log.info("All PNGs already have captions. Use --force to recompute.")
        return

    captioner = FaceCaptioner(
        base_url=caption_url,
        model=caption_model,
        api_key=caption_api_key,
        concurrency=caption_concurrency,
        timeout=caption_timeout,
    )

    log.info("Captioning %d faces …", len(pending))
    bgr_images = []
    for path in pending:
        rgb = np.array(Image.open(str(path)).convert("RGB"))
        bgr_images.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    captions = captioner.caption_many(bgr_images, progress=True)

    written = empty = verify_fail = 0
    for path, cap in tqdm(list(zip(pending, captions)), desc="Writing captions", unit="img"):
        cap = cap.strip()
        if not cap:
            log.warning("Empty caption returned for %s — skipping (re-run to retry)", path.name)
            empty += 1
            continue
        update_png_chunks(path, force=True, caption=cap)
        saved = Image.open(str(path)).text.get("alphaface_caption", "").strip()
        if saved != cap:
            log.error("Write verification failed for %s (got %r)", path.name, saved[:40] if saved else "")
            verify_fail += 1
        else:
            written += 1

    log.info("Caption complete — %d written, %d empty, %d verify failures", written, empty, verify_fail)
    if verify_fail:
        log.error("%d files failed write verification — check filesystem/permissions", verify_fail)
    if empty:
        log.warning("%d images have no caption — re-run 'alphaface caption' to retry them", empty)


def run_embed(
    folder: Path,
    device: str = "cpu",
    batch_size: int = 64,
    force: bool = False,
) -> None:
    import clip as _clip

    pngs = sorted(folder.glob("*.png"))
    if not pngs:
        sys.exit(f"No PNGs found in {folder}")

    pending = []  # (path, PIL image, caption, needs_img, needs_txt)
    error_count = 0

    for path in tqdm(pngs, desc="Scanning", unit="img"):
        pil = Image.open(str(path))
        meta = pil.text
        caption = meta.get("alphaface_caption", "").strip()
        if not caption:
            log.error("No caption in %s — run 'alphaface caption' first", path.name)
            error_count += 1
            continue
        needs_img = force or not meta.get("alphaface_clip_img", "").strip()
        needs_txt = force or not meta.get("alphaface_clip_txt", "").strip()
        if needs_img or needs_txt:
            pending.append((path, pil, caption, needs_img, needs_txt))

    if error_count:
        log.warning("%d PNGs skipped — missing caption", error_count)
    if not pending:
        log.info("All PNGs already have CLIP embeddings. Use --force to recompute.")
        return

    log.info("Loading CLIP for %d samples …", len(pending))
    clip_model, clip_preprocess = _clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()

    for i in tqdm(range(0, len(pending), batch_size), desc="CLIP embeddings", unit="batch"):
        batch = pending[i : i + batch_size]
        paths = [x[0] for x in batch]
        pils = [x[1] for x in batch]
        caps = [x[2] for x in batch]
        need_img = [x[3] for x in batch]
        need_txt = [x[4] for x in batch]

        img_embs: np.ndarray | None = None
        if any(need_img):
            clip_input = torch.stack([clip_preprocess(p.convert("RGB")) for p in pils]).to(device)
            with torch.no_grad():
                img_embs = clip_model.encode_image(clip_input).cpu().float().numpy()

        txt_embs: np.ndarray | None = None
        if any(need_txt):
            tokens = _clip.tokenize(caps, context_length=77, truncate=True).to(device)
            with torch.no_grad():
                txt_embs = clip_model.encode_text(tokens).cpu().float().numpy()

        for j, path in enumerate(paths):
            update_png_chunks(
                path,
                force=True,
                clip_img_emb=img_embs[j] if need_img[j] and img_embs is not None else None,
                clip_txt_emb=txt_embs[j] if need_txt[j] and txt_embs is not None else None,
            )

    log.info("Embed complete — %d PNGs processed", len(pending))


def run_identify(
    folder: Path,
    checkpoint: str,
    device: str = "cpu",
    batch_size: int = 64,
    force: bool = False,
) -> None:
    pngs = sorted(folder.glob("*.png"))
    if not pngs:
        sys.exit(f"No PNGs found in {folder}")

    pending = []
    for path in tqdm(pngs, desc="Scanning", unit="img"):
        pil = Image.open(str(path))
        if not force and pil.text.get("alphaface_id_emb", "").strip():
            continue
        pending.append(path)

    if not pending:
        log.info("All PNGs already have ArcFace embeddings. Use --force to recompute.")
        return

    from ..models.swapper_alphaface import build_alpha_face

    log.info("Loading ArcFace from %s …", checkpoint)
    model = build_alpha_face(device=device)
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("swapper", ckpt)
    model.id_encoder.load_state_dict(state, strict=False)
    id_encoder = model.id_encoder.eval().to(device)

    for i in tqdm(range(0, len(pending), batch_size), desc="ArcFace embeddings", unit="batch"):
        batch_paths = pending[i : i + batch_size]
        id_imgs = []
        for path in batch_paths:
            rgb = np.array(Image.open(str(path)).convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            face_112 = cv2.resize(bgr, (112, 112))
            face_t = torch.from_numpy(face_112).permute(2, 0, 1).float()
            face_t = (face_t / 127.5) - 1.0
            id_imgs.append(face_t)
        id_batch = torch.stack(id_imgs).to(device)
        with torch.no_grad():
            id_embs = id_encoder(id_batch).cpu().float().numpy()
        for j, path in enumerate(batch_paths):
            update_png_chunks(path, force=True, id_emb=id_embs[j])

    log.info("Identify complete — %d PNGs processed", len(pending))
