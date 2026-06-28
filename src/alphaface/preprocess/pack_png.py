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
    alphaface_id_emb    ArcFace identity embedding      (512 × float16, base64) — optional
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


def pack_png(
    img_rgb: np.ndarray,
    mask: np.ndarray | None,
    caption: str | None,
    clip_img_emb: np.ndarray,
    clip_txt_emb: np.ndarray,
    id_emb: np.ndarray | None,
    out_path: str | Path,
) -> None:
    """Write a packed RGBA PNG to *out_path*.

    Args:
        img_rgb:      uint8 (H, W, 3) RGB image.
        mask:         uint8 (H, W) mask (0=face, 255=bg). Stored as alpha channel.
                      When None a fully-opaque alpha (255) is written.
        caption:      Free-text caption string.
        clip_img_emb: float32/16 (512,) CLIP image embedding.
        clip_txt_emb: float32/16 (512,) CLIP text embedding.
        id_emb:       float32/16 (512,) ArcFace embedding, or None to omit.
        out_path:     Destination path (created / overwritten).
    """
    if mask is None:
        alpha = np.full(img_rgb.shape[:2], 255, dtype=np.uint8)
    else:
        alpha = mask.astype(np.uint8)

    rgba = np.dstack([img_rgb.astype(np.uint8), alpha])
    pil = Image.fromarray(rgba, "RGBA")

    info = PngImagePlugin.PngInfo()
    info.add_itxt("alphaface_caption", caption or "")
    info.add_itxt("alphaface_clip_img", _enc(clip_img_emb))
    info.add_itxt("alphaface_clip_txt", _enc(clip_txt_emb))
    if id_emb is not None:
        info.add_itxt("alphaface_id_emb", _enc(id_emb))

    pil.save(str(out_path), pnginfo=info, optimize=False)


def unpack_png(path: str | Path) -> PackedSample:
    """Load a packed or legacy PNG and return a :class:`PackedSample`.

    A PNG is considered *packed* when:
    - Its mode is ``'RGBA'``, AND
    - The iTXt key ``alphaface_caption`` is present.

    For legacy RGB PNGs (old 3-file layout) the embedding/mask/caption fields
    are returned as ``None`` so callers can gracefully fall back.
    """
    pil = Image.open(str(path))
    meta = pil.text  # dict populated from iTXt/tEXt chunks

    if pil.mode != "RGBA" or "alphaface_caption" not in meta:
        return PackedSample(
            img_rgb=np.array(pil.convert("RGB")),
            mask=None,
            caption=None,
            clip_img_emb=None,
            clip_txt_emb=None,
            id_emb=None,
        )

    rgba = np.array(pil)
    img_rgb = rgba[:, :, :3]
    mask = rgba[:, :, 3]

    return PackedSample(
        img_rgb=img_rgb,
        mask=mask,
        caption=meta.get("alphaface_caption") or None,
        clip_img_emb=_dec(meta["alphaface_clip_img"]) if "alphaface_clip_img" in meta else None,
        clip_txt_emb=_dec(meta["alphaface_clip_txt"]) if "alphaface_clip_txt" in meta else None,
        id_emb=_dec(meta["alphaface_id_emb"]) if "alphaface_id_emb" in meta else None,
    )
