"""Build an AlphaFace training dataset from a folder of raw images.

Pipeline per image
------------------
1. Detect + align face(s) to 256×256  →  <output>/img/<stem>.png
2. Generate binary face mask           →  <output>/mask/<stem>.png
3. Generate BLIP face caption          →  <output>/txt/<stem>.txt

Entry point
-----------
    alphaface-prepare-dataset --input /raw/images --output /dataset/custom

Or directly:
    python scripts/prepare_dataset.py --input /raw/images --output /dataset/custom
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

from .align import FaceAligner
from .caption import FaceCaptioner
from .mask import FaceMasker

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

log = logging.getLogger(__name__)


def _find_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in _IMAGE_EXTENSIONS)


def _output_stem(source_root: Path, image_path: Path, index: int) -> str:
    """Stable, collision-free output stem: index + flattened relative path."""
    rel_parts = image_path.relative_to(source_root).with_suffix("").parts
    flat = "_".join(rel_parts)
    return f"{index:06d}_{flat}"


def run(
    input_dir: Path,
    output_dir: Path,
    device: str = "cpu",
    output_size: int = 256,
    largest_only: bool = True,
    skip_mask: bool = False,
    skip_caption: bool = False,
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
    masker = FaceMasker() if not skip_mask else None
    captioner = FaceCaptioner(device=device) if not skip_caption else None

    written = skipped = 0
    global_idx = 0

    for image_path in tqdm(images, desc="Preprocessing", unit="img"):
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

            if captioner is not None:
                caption = captioner(af.image)
                (txt_dir / f"{stem}.txt").write_text(caption + "\n", encoding="utf-8")

            written += 1

    log.info("Finished — written: %d  skipped (no face / unreadable): %d", written, skipped)
    log.info("Dataset saved to %s", output_dir)


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
        help="Output dataset root (img/, mask/, txt/ created here)",
    )
    parser.add_argument(
        "--device", default=None, metavar="DEVICE", help="cuda or cpu (default: cuda if available, else cpu)"
    )
    parser.add_argument("--size", type=int, default=256, metavar="N", help="Output image size in pixels")
    parser.add_argument(
        "--all-faces", action="store_true", help="Keep every detected face per image instead of just the largest"
    )
    parser.add_argument("--no-mask", action="store_true", help="Skip face mask generation")
    parser.add_argument("--no-caption", action="store_true", help="Skip BLIP caption generation")
    args = parser.parse_args(argv)

    if not args.input.is_dir():
        sys.exit(f"Input path is not a directory: {args.input}")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    device = args.device or (_detect_device())

    run(
        input_dir=args.input,
        output_dir=args.output,
        device=device,
        output_size=args.size,
        largest_only=not args.all_faces,
        skip_mask=args.no_mask,
        skip_caption=args.no_caption,
    )


def _detect_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


if __name__ == "__main__":
    main()
