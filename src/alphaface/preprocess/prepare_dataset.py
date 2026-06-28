"""Build an AlphaFace training dataset from a folder of raw images.

Pipeline per image
------------------
1. Detect + align face(s) to 256×256  →  <output>/img/<stem>.png
2. Generate binary face mask           →  <output>/mask/<stem>.png
3. Generate VLM face caption           →  <output>/txt/<stem>.txt
4. Compute CLIP + ArcFace embeddings and
   pack everything into a single RGBA  →  <output>/packed/<stem>.png

Use 'alphaface --help' for the current staged CLI.
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .align import AlignedFace, FaceAligner
from .caption import FaceCaptioner
from .mask import FaceMasker
from .pack_png import PackedSample, pack_png, unpack_png

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
_REBUILD_FIELDS = frozenset({"caption", "clip_image", "clip_text", "arcface"})
_REBUILD_ALIASES = {
    "clip_img": "clip_image",
    "clip-image": "clip_image",
    "clip_image": "clip_image",
    "image": "clip_image",
    "clip_txt": "clip_text",
    "clip-text": "clip_text",
    "clip_text": "clip_text",
    "text": "clip_text",
    "id": "arcface",
    "id_emb": "arcface",
    "arcface": "arcface",
}

log = logging.getLogger(__name__)


def _normalize_rebuild_fields(rebuild: list[str] | tuple[str, ...] | None, rebuild_all: bool = False) -> set[str]:
    if rebuild_all:
        return set(_REBUILD_FIELDS)

    fields: set[str] = set()
    for raw in rebuild or []:
        key = raw.strip().lower()
        if not key:
            continue
        canonical = _REBUILD_ALIASES.get(key, key)
        if canonical not in _REBUILD_FIELDS:
            valid = ", ".join(sorted(_REBUILD_FIELDS))
            raise ValueError(f"Unknown rebuild field {raw!r}; expected one of: {valid}")
        fields.add(canonical)
    if "caption" in fields:
        fields.add("clip_text")
    return fields


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
    from ..models.swapper_alphaface import build_alpha_face  # lazy import

    model = build_alpha_face(device=device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("swapper", ckpt)
    model.id_encoder.load_state_dict(state, strict=False)
    model.id_encoder.eval().to(device)
    return model.id_encoder


def _load_existing_packed(packed_dir: Path) -> dict[str, PackedSample]:
    if not packed_dir.is_dir():
        return {}

    existing: dict[str, PackedSample] = {}
    packed_paths = sorted(packed_dir.glob("*.png"))
    for path in tqdm(packed_paths, desc="Reading packed cache", unit="png"):
        try:
            existing[path.stem] = unpack_png(path, require_complete=False)
        except Exception as exc:
            log.warning("Ignoring unreadable packed cache %s: %s", path, exc)
    return existing


def _existing_caption(
    stem: str,
    txt_dir: Path,
    existing: dict[str, PackedSample],
    rebuild_fields: set[str],
    *,
    allow_txt: bool = True,
) -> str:
    txt_file = txt_dir / f"{stem}.txt"
    if allow_txt and txt_file.exists():
        return txt_file.read_text(encoding="utf-8").strip()
    if "caption" in rebuild_fields:
        return ""
    cached = existing.get(stem)
    return (cached.caption or "").strip() if cached is not None else ""


def _compute_embeddings(
    pending: list[tuple[str, object]],
    captions: dict[str, str],
    existing: dict[str, PackedSample],
    rebuild_fields: set[str],
    clip_model,
    clip_preprocess,
    id_model,
    device: str,
    batch_size: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Return {stem: {clip_img, clip_txt, id_emb}} with CPU float32 arrays."""
    from PIL import Image as _Image

    result: dict[str, dict[str, np.ndarray | None]] = {}
    missing_clip_img: list[tuple[str, object]] = []
    missing_clip_txt: list[tuple[str, object]] = []
    missing_id: list[tuple[str, object]] = []

    for stem, af in tqdm(pending, desc="Planning embeddings", unit="sample"):
        cached = existing.get(stem)
        clip_img_emb = cached.clip_img_emb if cached is not None else None
        clip_txt_emb = cached.clip_txt_emb if cached is not None else None
        id_emb = cached.id_emb if cached is not None else None
        if "clip_image" in rebuild_fields:
            clip_img_emb = None
        if "clip_text" in rebuild_fields:
            clip_txt_emb = None
        if "arcface" in rebuild_fields:
            id_emb = None

        result[stem] = {
            "clip_img_emb": clip_img_emb,
            "clip_txt_emb": clip_txt_emb,
            "id_emb": id_emb,
        }
        if clip_img_emb is None:
            missing_clip_img.append((stem, af))
        if clip_txt_emb is None:
            missing_clip_txt.append((stem, af))
        if id_emb is None:
            missing_id.append((stem, af))

    log.info(
        "Embedding cache hits: clip_img=%d/%d clip_txt=%d/%d id=%d/%d",
        len(pending) - len(missing_clip_img),
        len(pending),
        len(pending) - len(missing_clip_txt),
        len(pending),
        len(pending) - len(missing_id),
        len(pending),
    )

    for i in tqdm(range(0, len(missing_clip_img), batch_size), desc="CLIP image embeddings", unit="batch"):
        batch = missing_clip_img[i : i + batch_size]
        batch_stems = [s for s, _ in batch]
        batch_faces = [af for _, af in batch]
        pil_imgs = [_Image.fromarray(cv2.cvtColor(af.image, cv2.COLOR_BGR2RGB)) for af in batch_faces]
        clip_img_batch = torch.stack([clip_preprocess(p) for p in pil_imgs]).to(device)
        with torch.no_grad():
            clip_img_embs = clip_model.encode_image(clip_img_batch).cpu().float().numpy()
        for j, stem in enumerate(batch_stems):
            result[stem]["clip_img_emb"] = clip_img_embs[j]

    for i in tqdm(range(0, len(missing_clip_txt), batch_size), desc="CLIP text embeddings", unit="batch"):
        import clip as _clip

        batch = missing_clip_txt[i : i + batch_size]
        batch_stems = [s for s, _ in batch]
        batch_caps = [captions.get(s, "") for s in batch_stems]
        tokens = _clip.tokenize(batch_caps, context_length=77, truncate=True).to(device)
        with torch.no_grad():
            clip_txt_embs = clip_model.encode_text(tokens).cpu().float().numpy()
        for j, stem in enumerate(batch_stems):
            result[stem]["clip_txt_emb"] = clip_txt_embs[j]

    for i in tqdm(range(0, len(missing_id), batch_size), desc="ArcFace embeddings", unit="batch"):
        batch = missing_id[i : i + batch_size]
        batch_stems = [s for s, _ in batch]
        batch_faces = [af for _, af in batch]
        id_imgs = []
        for af in batch_faces:
            face_112 = cv2.resize(af.image, (112, 112))  # BGR uint8
            face_t = torch.from_numpy(face_112).permute(2, 0, 1).float()
            face_t = (face_t / 127.5) - 1.0  # [-1, 1]
            id_imgs.append(face_t)
        id_batch = torch.stack(id_imgs).to(device)
        with torch.no_grad():
            id_embs_t = id_model(id_batch).cpu().float().numpy()
        for j, stem in enumerate(batch_stems):
            result[stem]["id_emb"] = id_embs_t[j]

    return result  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def _write_packed_stub(
    stem: str,
    bgr: np.ndarray,
    mask_dir: Path,
    packed_dir: Path,
    skip_mask: bool,
) -> None:
    """Write image+mask to packed/{stem}.png if no packed file exists yet.

    Skipped when a packed PNG is already present so we never downgrade a
    complete packed sample back to a stub.
    """
    packed_path = packed_dir / f"{stem}.png"
    if packed_path.exists():
        return
    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mask_path = mask_dir / f"{stem}.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if not skip_mask and mask_path.exists() else None
    pack_png(
        img_rgb=img_rgb,
        mask=mask,
        caption=None,
        clip_img_emb=None,
        clip_txt_emb=None,
        id_emb=None,
        out_path=packed_path,
    )


def run(
    input_dir: Path,
    output_dir: Path,
    device: str = "cpu",
    output_size: int = 256,
    largest_only: bool = True,
    skip_mask: bool = False,
    mask_model: str = "weights/resnet34.onnx",
    caption_url: str = "http://localhost:11434/v1",
    caption_model: str = "llava:7b",
    caption_api_key: str = "none",
    caption_concurrency: int = 4,
    caption_timeout: float = 60.0,
    id_encoder_checkpoint: str | None = None,
    embed_batch_size: int = 64,
    delete_originals: bool = False,
    rebuild: list[str] | tuple[str, ...] | None = None,
    rebuild_all: bool = False,
) -> None:
    rebuild_fields = _normalize_rebuild_fields(rebuild, rebuild_all)
    images = _find_images(input_dir)
    if not images:
        sys.exit(f"No images found under {input_dir}")
    if not id_encoder_checkpoint:
        sys.exit("--id-encoder-checkpoint is required so packed PNGs contain ArcFace embeddings")
    log.info("Found %d source images", len(images))

    img_dir = output_dir / "img"
    mask_dir = output_dir / "mask"
    txt_dir = output_dir / "txt"
    packed_dir = output_dir / "packed"
    img_dir.mkdir(parents=True, exist_ok=True)
    if not skip_mask:
        mask_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)
    packed_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading models on %s …", device)
    aligner: FaceAligner | None = None
    masker = FaceMasker(model_path=mask_model, device=device) if not skip_mask else None
    captioner = FaceCaptioner(
        base_url=caption_url,
        model=caption_model,
        api_key=caption_api_key,
        concurrency=caption_concurrency,
        timeout=caption_timeout,
    )

    # Phase 1: align + mask, collecting faces for batch captioning.
    pending: list[tuple[str, object]] = []  # (stem, aligned_face)
    skipped = 0
    global_idx = 0

    align_hits = 0
    for image_path in tqdm(images, desc="Aligning", unit="img"):
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            log.warning("Cannot read %s — skipped", image_path)
            skipped += 1
            continue

        flat = "_".join(image_path.relative_to(input_dir).with_suffix("").parts)
        cached_stems = sorted(img_dir.glob(f"??????_{flat}.png"))
        first_cached_idx = int(cached_stems[0].stem[:6]) if cached_stems else -1

        if cached_stems and first_cached_idx == global_idx:
            for cf in cached_stems:
                stem = cf.stem
                cached_bgr = cv2.imread(str(cf))
                if cached_bgr is None:
                    continue
                if masker is not None and not (mask_dir / f"{stem}.png").exists():
                    af_tmp = AlignedFace(image=cached_bgr, landmarks_68=np.zeros((68, 2), dtype=np.float32))
                    cv2.imwrite(str(mask_dir / f"{stem}.png"), masker(af_tmp))
                _write_packed_stub(stem, cached_bgr, mask_dir, packed_dir, skip_mask)
                pending.append((stem, AlignedFace(image=cached_bgr, landmarks_68=np.zeros((68, 2), dtype=np.float32))))
                global_idx += 1
                align_hits += 1
            continue

        if aligner is None:
            log.info("Cache miss — loading FaceAligner on %s …", device)
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

            cv2.imwrite(str(img_dir / f"{stem}.png"), af.image)

            if masker is not None:
                cv2.imwrite(str(mask_dir / f"{stem}.png"), masker(af))

            _write_packed_stub(stem, af.image, mask_dir, packed_dir, skip_mask)
            pending.append((stem, af))

    log.info("Alignment cache hits: %d/%d", align_hits, len(images) - skipped)

    existing = _load_existing_packed(packed_dir)

    # Phase 2: concurrent captioning over faces without a cached/file caption.
    caption_pending = [
        (stem, af)
        for stem, af in tqdm(pending, desc="Checking captions", unit="sample")
        if not _existing_caption(stem, txt_dir, existing, rebuild_fields, allow_txt="caption" not in rebuild_fields)
    ]
    if caption_pending:
        log.info("Captioning %d faces concurrently (concurrency=%d) …", len(caption_pending), caption_concurrency)
        stems, faces = zip(*caption_pending)
        captions_list = captioner.caption_many([f.image for f in faces], progress=True)
        for stem, caption in tqdm(list(zip(stems, captions_list)), desc="Writing captions", unit="txt"):
            (txt_dir / f"{stem}.txt").write_text(caption + "\n", encoding="utf-8")
    else:
        log.info("Caption cache hits: %d/%d", len(pending), len(pending))

    written = len(pending)
    log.info("Finished phases 1-2 — written: %d  skipped: %d", written, skipped)

    # Phase 3: compute embeddings and pack into single RGBA PNG files.
    if pending:
        _run_pack(
            pending=pending,
            txt_dir=txt_dir,
            mask_dir=mask_dir,
            output_dir=output_dir,
            existing=existing,
            rebuild_fields=rebuild_fields,
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
    existing: dict[str, PackedSample] | None,
    rebuild_fields: set[str],
    device: str,
    id_encoder_checkpoint: str | None,
    embed_batch_size: int,
    skip_mask: bool,
    delete_originals: bool,
) -> None:
    existing = existing or _load_existing_packed(output_dir / "packed")

    captions: dict[str, str] = {}
    for stem, _ in tqdm(pending, desc="Loading captions", unit="sample"):
        captions[stem] = _existing_caption(stem, txt_dir, existing, rebuild_fields)

    missing_clip = []
    missing_id = []
    for stem, _ in tqdm(pending, desc="Checking embedding cache", unit="sample"):
        cached = existing.get(stem)
        if (
            "clip_image" in rebuild_fields
            or "clip_text" in rebuild_fields
            or cached is None
            or cached.clip_img_emb is None
            or cached.clip_txt_emb is None
        ):
            missing_clip.append(stem)
        if "arcface" in rebuild_fields or cached is None or cached.id_emb is None:
            missing_id.append(stem)

    clip_model = None
    clip_preprocess = None
    if missing_clip:
        import clip as _clip

        log.info("Phase 3: loading CLIP for %d samples with missing CLIP metadata …", len(missing_clip))
        clip_model, clip_preprocess = _clip.load("ViT-B/32", device=device, jit=False)
        clip_model.eval()

    if not id_encoder_checkpoint:
        raise ValueError("id_encoder_checkpoint is required")
    id_model = None
    if missing_id:
        log.info(
            "Loading ID encoder from %s for %d samples with missing ArcFace metadata …",
            id_encoder_checkpoint,
            len(missing_id),
        )
        id_model = _load_id_encoder(id_encoder_checkpoint, device)

    log.info("Computing embeddings for %d faces (batch_size=%d) …", len(pending), embed_batch_size)
    emb_map = _compute_embeddings(
        pending,
        captions,
        existing,
        rebuild_fields,
        clip_model,
        clip_preprocess,
        id_model,
        device,
        embed_batch_size,
    )

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
# Standalone converter for existing img/mask/txt datasets
# ---------------------------------------------------------------------------


def pack_existing(
    dataset_dir: Path,
    device: str = "cpu",
    id_encoder_checkpoint: str | None = None,
    embed_batch_size: int = 64,
    delete_originals: bool = False,
    rebuild: list[str] | tuple[str, ...] | None = None,
    rebuild_all: bool = False,
) -> None:
    """Convert an already-prepared img/mask/txt dataset into the packed/ layout."""
    from glob import glob

    rebuild_fields = _normalize_rebuild_fields(rebuild, rebuild_all)

    img_dir = dataset_dir / "img"
    mask_dir = dataset_dir / "mask"
    txt_dir = dataset_dir / "txt"

    if not img_dir.is_dir():
        sys.exit(f"No img/ directory found under {dataset_dir}")
    if not txt_dir.is_dir():
        sys.exit(f"No txt/ directory found under {dataset_dir}; packed PNGs require captions")
    if not id_encoder_checkpoint:
        sys.exit("--id-encoder-checkpoint is required so packed PNGs contain ArcFace embeddings")

    img_paths = sorted(glob(str(img_dir / "*.png")))
    if not img_paths:
        sys.exit(f"No PNGs found in {img_dir}")

    log.info("Found %d images in %s", len(img_paths), img_dir)

    class _FakeAF:
        """Minimal stand-in for AlignedFace holding a BGR image."""

        def __init__(self, bgr: np.ndarray):
            self.image = bgr

    pending = []
    for p in tqdm(img_paths, desc="Loading prepared images", unit="img"):
        stem = Path(p).stem
        bgr = cv2.imread(p)
        if bgr is not None:
            pending.append((stem, _FakeAF(bgr)))

    _run_pack(
        pending=pending,
        txt_dir=txt_dir,
        mask_dir=mask_dir,
        output_dir=dataset_dir,
        existing=_load_existing_packed(dataset_dir / "packed"),
        rebuild_fields=rebuild_fields,
        device=device,
        id_encoder_checkpoint=id_encoder_checkpoint,
        embed_batch_size=embed_batch_size,
        skip_mask=not mask_dir.is_dir(),
        delete_originals=delete_originals,
    )
