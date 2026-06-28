"""Build an AlphaFace training dataset from a folder of raw images.

Pipeline per image
------------------
1. Detect + align face(s) to 256×256  →  <output>/img/<stem>.png
2. Generate binary face mask           →  <output>/mask/<stem>.png
3. Generate VLM face caption           →  <output>/txt/<stem>.txt
4. Compute CLIP + ID embeddings and
   pack everything into a single RGBA  →  <output>/packed/<stem>.png

Entry point
-----------
    alphaface-prepare-dataset --input /raw/images --output /dataset/custom

Or directly:
    python scripts/prepare_dataset.py --input /raw/images --output /dataset/custom

Flags
-----
    --skip-pack               Skip Phase 4 (keep 3-file layout only)
    --delete-originals        Remove img/ mask/ txt/ after packing
    --id-encoder-checkpoint   Path to ArcFace .pt checkpoint for ID embeddings
    --embed-batch-size N      Mini-batch size for embedding inference (default 64)

Standalone packer (for existing 3-file datasets)
-------------------------------------------------
    alphaface-pack-dataset --dataset /path/to/dataset
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .align import FaceAligner
from .caption import FaceCaptioner
from .mask import FaceMasker
from .pack_png import pack_png

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

log = logging.getLogger(__name__)


def _find_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in _IMAGE_EXTENSIONS)


def _output_stem(source_root: Path, image_path: Path, index: int) -> str:
    """Stable, collision-free output stem: index + flattened relative path."""
    rel_parts = image_path.relative_to(source_root).with_suffix("").parts
    flat = "_".join(rel_parts)
    return f"{index:06d}_{flat}"


# ---------------------------------------------------------------------------
# Phase 4 helpers
# ---------------------------------------------------------------------------


def _load_id_encoder(checkpoint_path: str, device: str):
    """Load a standalone ArcFace ResNet-50 checkpoint for embedding inference."""
    from ..models.swapper_alphaface import build_AlphaFace  # lazy import

    model = build_AlphaFace(device=device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("swapper", ckpt)
    model.Id_encoder.load_state_dict(state, strict=False)
    model.Id_encoder.eval().to(device)
    return model.Id_encoder


def _compute_embeddings(
    pending: list[tuple[str, object]],
    captions: dict[str, str],
    clip_model,
    clip_preprocess,
    id_model,
    device: str,
    batch_size: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Return {stem: {clip_img, clip_txt, id_emb?}} with CPU float32 arrays."""
    import clip as _clip
    from PIL import Image as _Image

    stems = [s for s, _ in pending]
    faces = [af for _, af in pending]
    caps = [captions.get(s, "") for s in stems]

    result: dict[str, dict] = {}

    for i in range(0, len(stems), batch_size):
        batch_stems = stems[i : i + batch_size]
        batch_faces = faces[i : i + batch_size]
        batch_caps = caps[i : i + batch_size]

        # --- CLIP image ---
        pil_imgs = [_Image.fromarray(cv2.cvtColor(af.image, cv2.COLOR_BGR2RGB)) for af in batch_faces]
        clip_img_batch = torch.stack([clip_preprocess(p) for p in pil_imgs]).to(device)
        with torch.no_grad():
            clip_img_embs = clip_model.encode_image(clip_img_batch).cpu().float().numpy()

        # --- CLIP text ---
        tokens = _clip.tokenize(batch_caps, context_length=77, truncate=True).to(device)
        with torch.no_grad():
            clip_txt_embs = clip_model.encode_text(tokens).cpu().float().numpy()

        # --- ArcFace ID (optional) ---
        id_embs = [None] * len(batch_stems)
        if id_model is not None:
            id_imgs = []
            for af in batch_faces:
                face_112 = cv2.resize(af.image, (112, 112))  # BGR uint8
                face_t = torch.from_numpy(face_112).permute(2, 0, 1).float()
                face_t = (face_t / 127.5) - 1.0  # [-1, 1]
                id_imgs.append(face_t)
            id_batch = torch.stack(id_imgs).to(device)
            with torch.no_grad():
                id_embs_t = id_model(id_batch).cpu().float().numpy()
            id_embs = list(id_embs_t)

        for j, stem in enumerate(batch_stems):
            result[stem] = {
                "clip_img_emb": clip_img_embs[j],
                "clip_txt_emb": clip_txt_embs[j],
                "id_emb": id_embs[j],
            }

    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(
    input_dir: Path,
    output_dir: Path,
    device: str = "cpu",
    output_size: int = 256,
    largest_only: bool = True,
    skip_mask: bool = False,
    skip_caption: bool = False,
    mask_model: str = "weights/resnet34.onnx",
    caption_url: str = "http://localhost:11434/v1",
    caption_model: str = "llava:7b",
    caption_api_key: str = "none",
    caption_concurrency: int = 4,
    caption_timeout: float = 60.0,
    skip_pack: bool = False,
    id_encoder_checkpoint: str | None = None,
    embed_batch_size: int = 64,
    delete_originals: bool = False,
) -> None:
    images = _find_images(input_dir)
    if not images:
        sys.exit(f"No images found under {input_dir}")
    log.info("Found %d source images", len(images))

    img_dir = output_dir / "img"
    mask_dir = output_dir / "mask"
    txt_dir = output_dir / "txt"
    img_dir.mkdir(parents=True, exist_ok=True)
    if not skip_mask:
        mask_dir.mkdir(parents=True, exist_ok=True)
    if not skip_caption:
        txt_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading models on %s …", device)
    aligner = FaceAligner(device=device)
    masker = FaceMasker(model_path=mask_model, device=device) if not skip_mask else None
    captioner = (
        FaceCaptioner(
            base_url=caption_url,
            model=caption_model,
            api_key=caption_api_key,
            concurrency=caption_concurrency,
            timeout=caption_timeout,
        )
        if not skip_caption
        else None
    )

    # Phase 1: align + mask, collecting faces for batch captioning.
    pending: list[tuple[str, object]] = []  # (stem, aligned_face)
    skipped = 0
    global_idx = 0

    for image_path in tqdm(images, desc="Aligning", unit="img"):
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            log.warning("Cannot read %s — skipped", image_path)
            skipped += 1
            continue

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

            cv2.imwrite(str(img_dir / f"{stem}.png"), af.image)

            if masker is not None:
                cv2.imwrite(str(mask_dir / f"{stem}.png"), masker(af))

            pending.append((stem, af))

    # Phase 2: concurrent captioning over all collected faces.
    if captioner is not None and pending:
        log.info("Captioning %d faces concurrently (concurrency=%d) …", len(pending), caption_concurrency)
        stems, faces = zip(*pending)
        captions_list = captioner.caption_many([f.image for f in faces])
        for stem, caption in zip(stems, captions_list):
            (txt_dir / f"{stem}.txt").write_text(caption + "\n", encoding="utf-8")

    written = len(pending)
    log.info("Finished phases 1-2 — written: %d  skipped: %d", written, skipped)

    # Phase 3: compute embeddings and pack into single RGBA PNG files.
    if not skip_pack and pending:
        _run_pack(
            pending=pending,
            txt_dir=txt_dir,
            mask_dir=mask_dir,
            output_dir=output_dir,
            device=device,
            id_encoder_checkpoint=id_encoder_checkpoint,
            embed_batch_size=embed_batch_size,
            skip_mask=skip_mask,
            delete_originals=delete_originals,
        )

    log.info("Dataset saved to %s", output_dir)


def _run_pack(
    pending: list[tuple[str, object]],
    txt_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    device: str,
    id_encoder_checkpoint: str | None,
    embed_batch_size: int,
    skip_mask: bool,
    delete_originals: bool,
) -> None:
    import clip as _clip

    log.info("Phase 3: loading CLIP for embedding computation …")
    clip_model, clip_preprocess = _clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()

    id_model = None
    if id_encoder_checkpoint:
        log.info("Loading ID encoder from %s …", id_encoder_checkpoint)
        id_model = _load_id_encoder(id_encoder_checkpoint, device)

    captions: dict[str, str] = {}
    for stem, _ in pending:
        txt_file = txt_dir / f"{stem}.txt"
        captions[stem] = txt_file.read_text(encoding="utf-8").strip() if txt_file.exists() else ""

    log.info("Computing embeddings for %d faces (batch_size=%d) …", len(pending), embed_batch_size)
    emb_map = _compute_embeddings(pending, captions, clip_model, clip_preprocess, id_model, device, embed_batch_size)

    packed_dir = output_dir / "packed"
    packed_dir.mkdir(exist_ok=True)

    for stem, af in tqdm(pending, desc="Packing", unit="sample"):
        img_rgb = cv2.cvtColor(af.image, cv2.COLOR_BGR2RGB)

        if not skip_mask:
            mask_path = mask_dir / f"{stem}.png"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
        else:
            mask = None

        embs = emb_map[stem]
        pack_png(
            img_rgb=img_rgb,
            mask=mask,
            caption=captions[stem],
            clip_img_emb=embs["clip_img_emb"],
            clip_txt_emb=embs["clip_txt_emb"],
            id_emb=embs["id_emb"],
            out_path=packed_dir / f"{stem}.png",
        )

    log.info("Packed %d samples → %s", len(pending), packed_dir)

    if delete_originals:
        for d in (output_dir / "img", output_dir / "mask", output_dir / "txt"):
            if d.exists():
                shutil.rmtree(d)
        log.info("Deleted original img/ mask/ txt/ directories")


# ---------------------------------------------------------------------------
# Standalone packer for existing 3-file datasets
# ---------------------------------------------------------------------------


def pack_existing(
    dataset_dir: Path,
    device: str = "cpu",
    id_encoder_checkpoint: str | None = None,
    embed_batch_size: int = 64,
    delete_originals: bool = False,
) -> None:
    """Pack an already-prepared 3-file dataset into the packed/ layout."""
    from glob import glob

    img_dir = dataset_dir / "img"
    mask_dir = dataset_dir / "mask"
    txt_dir = dataset_dir / "txt"

    if not img_dir.is_dir():
        sys.exit(f"No img/ directory found under {dataset_dir}")

    img_paths = sorted(glob(str(img_dir / "*.png")))
    if not img_paths:
        sys.exit(f"No PNGs found in {img_dir}")

    log.info("Found %d images in %s", len(img_paths), img_dir)

    class _FakeAF:
        """Minimal stand-in for AlignedFace holding a BGR image."""

        def __init__(self, bgr: np.ndarray):
            self.image = bgr

    pending = []
    for p in img_paths:
        stem = Path(p).stem
        bgr = cv2.imread(p)
        if bgr is not None:
            pending.append((stem, _FakeAF(bgr)))

    _run_pack(
        pending=pending,
        txt_dir=txt_dir,
        mask_dir=mask_dir,
        output_dir=dataset_dir,
        device=device,
        id_encoder_checkpoint=id_encoder_checkpoint,
        embed_batch_size=embed_batch_size,
        skip_mask=not mask_dir.is_dir(),
        delete_originals=delete_originals,
    )


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Prepare AlphaFace training dataset from raw images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        metavar="DIR",
        help="Root folder of raw images (searched recursively)",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
        metavar="DIR",
        help="Output dataset root (img/, mask/, txt/, packed/ created here)",
    )
    parser.add_argument(
        "--device", default=None, metavar="DEVICE", help="cuda or cpu (default: cuda if available, else cpu)"
    )
    parser.add_argument("--size", type=int, default=256, metavar="N", help="Output image size in pixels")
    parser.add_argument(
        "--all-faces", action="store_true", help="Keep every detected face per image instead of just the largest"
    )
    parser.add_argument("--no-mask", action="store_true", help="Skip face mask generation")
    parser.add_argument(
        "--mask-model",
        default="weights/resnet34.onnx",
        metavar="PATH",
        help="Path to BiSeNet ONNX model (resnet18.onnx or resnet34.onnx)",
    )
    parser.add_argument("--no-caption", action="store_true", help="Skip caption generation")
    parser.add_argument(
        "--caption-url",
        default="http://localhost:11434/v1",
        metavar="URL",
        help="Base URL of the OpenAI-compatible vision server",
    )
    parser.add_argument(
        "--caption-model", default="llava:7b", metavar="MODEL", help="Model name to request from the server"
    )
    parser.add_argument(
        "--caption-api-key", default="none", metavar="KEY", help="API key for the server (use 'none' for local servers)"
    )
    parser.add_argument(
        "--caption-concurrency", type=int, default=4, metavar="N", help="Max simultaneous caption requests"
    )
    parser.add_argument(
        "--caption-timeout", type=float, default=60.0, metavar="SEC", help="Per-request timeout in seconds"
    )
    # Phase 3 flags
    parser.add_argument(
        "--skip-pack", action="store_true", help="Skip Phase 3: do not compute embeddings or create packed/ PNGs"
    )
    parser.add_argument(
        "--id-encoder-checkpoint",
        default=None,
        metavar="PATH",
        help="Path to ArcFace checkpoint for ID embedding pre-computation",
    )
    parser.add_argument(
        "--embed-batch-size", type=int, default=64, metavar="N", help="Mini-batch size for embedding inference"
    )
    parser.add_argument(
        "--delete-originals", action="store_true", help="Delete img/ mask/ txt/ directories after packing"
    )
    args = parser.parse_args(argv)

    if not args.input.is_dir():
        sys.exit(f"Input path is not a directory: {args.input}")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    device = args.device or _detect_device()

    run(
        input_dir=args.input,
        output_dir=args.output,
        device=device,
        output_size=args.size,
        largest_only=not args.all_faces,
        skip_mask=args.no_mask,
        skip_caption=args.no_caption,
        mask_model=args.mask_model,
        caption_url=args.caption_url,
        caption_model=args.caption_model,
        caption_api_key=args.caption_api_key,
        caption_concurrency=args.caption_concurrency,
        caption_timeout=args.caption_timeout,
        skip_pack=args.skip_pack,
        id_encoder_checkpoint=args.id_encoder_checkpoint,
        embed_batch_size=args.embed_batch_size,
        delete_originals=args.delete_originals,
    )


def pack_main(argv: list[str] | None = None) -> None:
    """Entry point for alphaface-pack-dataset (existing 3-file datasets)."""
    parser = argparse.ArgumentParser(
        description="Pack an existing AlphaFace 3-file dataset (img/mask/txt) into packed/ PNGs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        "-d",
        required=True,
        type=Path,
        metavar="DIR",
        help="Dataset root containing img/, mask/, txt/ subdirectories",
    )
    parser.add_argument(
        "--device", default=None, metavar="DEVICE", help="cuda or cpu (default: cuda if available, else cpu)"
    )
    parser.add_argument(
        "--id-encoder-checkpoint",
        default=None,
        metavar="PATH",
        help="Path to ArcFace checkpoint for ID embedding pre-computation",
    )
    parser.add_argument(
        "--embed-batch-size", type=int, default=64, metavar="N", help="Mini-batch size for embedding inference"
    )
    parser.add_argument(
        "--delete-originals", action="store_true", help="Delete img/ mask/ txt/ directories after packing"
    )
    args = parser.parse_args(argv)

    if not args.dataset.is_dir():
        sys.exit(f"Dataset path is not a directory: {args.dataset}")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    device = args.device or _detect_device()

    pack_existing(
        dataset_dir=args.dataset,
        device=device,
        id_encoder_checkpoint=args.id_encoder_checkpoint,
        embed_batch_size=args.embed_batch_size,
        delete_originals=args.delete_originals,
    )


def _detect_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


if __name__ == "__main__":
    main()
