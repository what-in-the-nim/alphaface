"""Unified entry point for all alphaface commands."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _detect_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _add_device(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", default=None, metavar="DEVICE", help="cuda or cpu (default: auto-detect)")


def _add_force(p: argparse.ArgumentParser) -> None:
    p.add_argument("--force", action="store_true", help="Overwrite existing data instead of skipping")


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="alphaface",
        description="AlphaFace — face swapping toolkit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ── extract ────────────────────────────────────────────────────────────────
    p_extract = sub.add_parser("extract", help="Align faces from raw images into a dataset folder")
    p_extract.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        metavar="DIR",
        help="Root folder of raw images (searched recursively)",
    )
    p_extract.add_argument(
        "--output", "-o", required=True, type=Path, metavar="DIR", help="Output folder; aligned PNGs written here"
    )
    _add_device(p_extract)
    p_extract.add_argument("--size", type=int, default=256, metavar="N", help="Output face size in pixels")
    p_extract.add_argument(
        "--all-faces", action="store_true", help="Keep every detected face per image, not just the largest"
    )
    _add_force(p_extract)

    # ── mask ───────────────────────────────────────────────────────────────────
    p_mask = sub.add_parser("mask", help="Add BiSeNet face mask as alpha channel to dataset PNGs")
    p_mask.add_argument(
        "--folder", "-f", required=True, type=Path, metavar="DIR", help="Dataset folder produced by 'extract'"
    )
    _add_device(p_mask)
    p_mask.add_argument(
        "--mask-model", default="weights/resnet34.onnx", metavar="PATH", help="Path to BiSeNet ONNX model"
    )
    _add_force(p_mask)

    # ── caption ────────────────────────────────────────────────────────────────
    p_caption = sub.add_parser("caption", help="Add VLM captions to dataset PNGs")
    p_caption.add_argument(
        "--folder", "-f", required=True, type=Path, metavar="DIR", help="Dataset folder produced by 'extract'"
    )
    p_caption.add_argument(
        "--caption-url",
        default="http://localhost:11434/v1",
        metavar="URL",
        help="Base URL of the OpenAI-compatible vision server",
    )
    p_caption.add_argument(
        "--caption-model", default="llava:7b", metavar="MODEL", help="Model name to request from the server"
    )
    p_caption.add_argument(
        "--caption-api-key", default="none", metavar="KEY", help="API key (use 'none' for local servers)"
    )
    p_caption.add_argument("--concurrency", type=int, default=4, metavar="N", help="Max simultaneous caption requests")
    p_caption.add_argument("--timeout", type=float, default=60.0, metavar="SEC", help="Per-request timeout in seconds")
    _add_force(p_caption)

    # ── embed ──────────────────────────────────────────────────────────────────
    p_embed = sub.add_parser("embed", help="Add CLIP image and text embeddings to dataset PNGs")
    p_embed.add_argument(
        "--folder",
        "-f",
        required=True,
        type=Path,
        metavar="DIR",
        help="Dataset folder (PNGs must already have captions)",
    )
    _add_device(p_embed)
    p_embed.add_argument("--batch-size", type=int, default=64, metavar="N", help="Mini-batch size for CLIP inference")
    _add_force(p_embed)

    # ── identify ───────────────────────────────────────────────────────────────
    p_identify = sub.add_parser("identify", help="Add ArcFace identity embeddings to dataset PNGs")
    p_identify.add_argument(
        "--folder", "-f", required=True, type=Path, metavar="DIR", help="Dataset folder produced by 'extract'"
    )
    p_identify.add_argument(
        "--checkpoint", required=True, metavar="PATH", help="Path to ArcFace model checkpoint (.pt)"
    )
    _add_device(p_identify)
    p_identify.add_argument(
        "--batch-size", type=int, default=64, metavar="N", help="Mini-batch size for ArcFace inference"
    )
    _add_force(p_identify)

    # ── train / eval passthrough ───────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Train the AlphaFace model (all args forwarded to LightningCLI)")
    p_train.add_argument("args", nargs=argparse.REMAINDER)

    p_eval = sub.add_parser("eval", help="Evaluate the AlphaFace model (all args forwarded to LightningCLI)")
    p_eval.add_argument("args", nargs=argparse.REMAINDER)

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    # ── passthrough subcommands ────────────────────────────────────────────────
    if args.command == "train":
        sys.argv = ["alphaface-train"] + list(args.args)
        from .train_clip import main as _train

        _train()
        return

    if args.command == "eval":
        sys.argv = ["alphaface-eval"] + list(args.args)
        from .eval import main as _eval

        _eval()
        return

    # ── dataset stages ─────────────────────────────────────────────────────────
    device = getattr(args, "device", None) or _detect_device()

    from .preprocess.stages import run_caption, run_embed, run_extract, run_identify, run_mask

    if args.command == "extract":
        if not args.input.is_dir():
            sys.exit(f"Input directory does not exist: {args.input}")
        run_extract(
            input_dir=args.input,
            output_dir=args.output,
            device=device,
            output_size=args.size,
            largest_only=not args.all_faces,
            force=args.force,
        )

    elif args.command == "mask":
        if not args.folder.is_dir():
            sys.exit(f"Folder does not exist: {args.folder}")
        run_mask(
            folder=args.folder,
            device=device,
            mask_model=args.mask_model,
            force=args.force,
        )

    elif args.command == "caption":
        if not args.folder.is_dir():
            sys.exit(f"Folder does not exist: {args.folder}")
        run_caption(
            folder=args.folder,
            caption_url=args.caption_url,
            caption_model=args.caption_model,
            caption_api_key=args.caption_api_key,
            caption_concurrency=args.concurrency,
            caption_timeout=args.timeout,
            force=args.force,
        )

    elif args.command == "embed":
        if not args.folder.is_dir():
            sys.exit(f"Folder does not exist: {args.folder}")
        run_embed(
            folder=args.folder,
            device=device,
            batch_size=args.batch_size,
            force=args.force,
        )

    elif args.command == "identify":
        if not args.folder.is_dir():
            sys.exit(f"Folder does not exist: {args.folder}")
        run_identify(
            folder=args.folder,
            checkpoint=args.checkpoint,
            device=device,
            batch_size=args.batch_size,
            force=args.force,
        )
