"""Pack/unpack a single RGBA PNG that stores image + mask + metadata.

Layout
------
- Channels 0-2 : RGB face image (uint8)
- Channel 3    : grayscale mask (0 = face, 255 = background)
- iTXt chunks  : caption and base64-encoded float16 embeddings

Keys written by pack_png
------------------------
    alphaface_caption   plain text caption
    alphaface_clip_img  CLIP ViT-B/32 image embedding  (512 × float16, base64)
    alphaface_clip_txt  CLIP ViT-B/32 text embedding   (512 × float16, base64)
    alphaface_id_emb    ArcFace identity embedding      (512 × float16, base64)
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin

_EMB_DTYPE = np.float16
_EMB_DIM = 512


@dataclass
class PackedSample:
    img_rgb: np.ndarray  # uint8 (H, W, 3)
    mask: np.ndarray | None  # uint8 (H, W) or None when absent
    caption: str | None
    clip_img_emb: np.ndarray | None  # float16 (512,)
    clip_txt_emb: np.ndarray | None  # float16 (512,)
    id_emb: np.ndarray | None  # float16 (512,)


def _enc(arr: np.ndarray) -> str:
    return base64.b64encode(arr.astype(_EMB_DTYPE).tobytes()).decode()


def _dec(b64: str) -> np.ndarray:
    # .copy() is mandatory — frombuffer returns a read-only view of the byte buffer
    return np.frombuffer(base64.b64decode(b64), dtype=_EMB_DTYPE).copy()


def _require_embedding(name: str, arr: np.ndarray | None) -> np.ndarray:
    if arr is None:
        raise ValueError(f"Packed PNG is missing required {name}")
    arr = np.asarray(arr)
    if arr.shape != (_EMB_DIM,):
        raise ValueError(f"Packed PNG {name} must have shape ({_EMB_DIM},), got {arr.shape}")
    return arr


def _optional_embedding(meta: dict[str, str], key: str) -> np.ndarray | None:
    if key not in meta or not meta[key]:
        return None
    arr = _dec(meta[key])
    return arr if arr.shape == (_EMB_DIM,) else None


def pack_png(
    img_rgb: np.ndarray,
    mask: np.ndarray | None,
    caption: str,
    clip_img_emb: np.ndarray,
    clip_txt_emb: np.ndarray,
    id_emb: np.ndarray,
    out_path: str | Path,
) -> None:
    """Write a packed RGBA PNG to *out_path*.

    Args:
        img_rgb:      uint8 (H, W, 3) RGB image.
        mask:         uint8 (H, W) mask (0=face, 255=bg). Stored as alpha channel.
                      When None a fully-opaque alpha (255) is written.
        caption:      Free-text caption string from the captioner.
        clip_img_emb: float32/16 (512,) CLIP image embedding.
        clip_txt_emb: float32/16 (512,) CLIP text embedding.
        id_emb:       float32/16 (512,) ArcFace embedding.
        out_path:     Destination path (created / overwritten).
    """
    caption = caption.strip()
    if not caption:
        raise ValueError("Packed PNG requires a non-empty caption")
    clip_img_emb = _require_embedding("alphaface_clip_img", clip_img_emb)
    clip_txt_emb = _require_embedding("alphaface_clip_txt", clip_txt_emb)
    id_emb = _require_embedding("alphaface_id_emb", id_emb)

    if mask is None:
        alpha = np.full(img_rgb.shape[:2], 255, dtype=np.uint8)
    else:
        alpha = mask.astype(np.uint8)

    rgba = np.dstack([img_rgb.astype(np.uint8), alpha])
    pil = Image.fromarray(rgba, "RGBA")

    info = PngImagePlugin.PngInfo()
    info.add_itxt("alphaface_caption", caption)
    info.add_itxt("alphaface_clip_img", _enc(clip_img_emb))
    info.add_itxt("alphaface_clip_txt", _enc(clip_txt_emb))
    info.add_itxt("alphaface_id_emb", _enc(id_emb))

    pil.save(str(out_path), pnginfo=info, optimize=False)


def unpack_png(path: str | Path, *, require_complete: bool = True) -> PackedSample:
    """Load a packed PNG and return a :class:`PackedSample`.

    Packed PNGs must be RGBA and include caption, CLIP image embedding, CLIP
    text embedding, and ArcFace embedding iTXt metadata unless
    ``require_complete`` is false.
    """
    pil = Image.open(str(path))
    meta = pil.text  # dict populated from iTXt/tEXt chunks

    required = ("alphaface_caption", "alphaface_clip_img", "alphaface_clip_txt", "alphaface_id_emb")
    missing = [key for key in required if key not in meta or not meta[key]]
    if pil.mode != "RGBA" or (require_complete and missing):
        raise ValueError(f"{path} is not a valid packed AlphaFace PNG; missing={missing}, mode={pil.mode}")

    rgba = np.array(pil)
    img_rgb = rgba[:, :, :3]
    mask = rgba[:, :, 3]

    caption = meta.get("alphaface_caption") or None
    clip_img_emb = _optional_embedding(meta, "alphaface_clip_img")
    clip_txt_emb = _optional_embedding(meta, "alphaface_clip_txt")
    id_emb = _optional_embedding(meta, "alphaface_id_emb")

    if require_complete:
        clip_img_emb = _require_embedding("alphaface_clip_img", clip_img_emb)
        clip_txt_emb = _require_embedding("alphaface_clip_txt", clip_txt_emb)
        id_emb = _require_embedding("alphaface_id_emb", id_emb)

    return PackedSample(
        img_rgb=img_rgb,
        mask=mask,
        caption=caption,
        clip_img_emb=clip_img_emb,
        clip_txt_emb=clip_txt_emb,
        id_emb=id_emb,
    )
