from __future__ import annotations

import cv2
import numpy as np

from .align import AlignedFace

# 68-point landmark layout (dlib/ibug convention, as returned by insightface buffalo_l):
#   0-16  jaw contour (left outer → chin → right outer)
#   17-21 left eyebrow  (inner → outer)
#   22-26 right eyebrow (inner → outer)
#   27-30 nose bridge
#   31-35 nose base
#   36-41 left eye
#   42-47 right eye
#   48-67 mouth


def _face_polygon(lm: np.ndarray, size: int) -> np.ndarray:
    """Build a face-covering polygon from 68 landmarks.

    The polygon follows the jaw contour along the bottom and sides, then
    connects across an estimated forehead boundary above the eyebrows.
    """
    jaw = lm[0:17]  # left outer → chin (8) → right outer

    left_brow = lm[17:22]  # inner → outer
    right_brow = lm[22:27]  # inner → outer

    # Push brows upward to cover forehead.
    # Use vertical distance from chin to nose-bridge as reference.
    chin_y = lm[8, 1]
    bridge_y = lm[27, 1]
    dy = max((chin_y - bridge_y) * 0.45, 20.0)

    # Forehead arc: right outer → right inner (reversed brow) then
    # left inner → left outer, all shifted upward.
    right_top = right_brow[::-1].copy()  # outer(26) → inner(22)
    right_top[:, 1] -= dy

    left_top = left_brow.copy()  # inner(17) → outer(21)
    left_top[:, 1] -= dy

    polygon = np.vstack([jaw, right_top, left_top])
    return np.clip(polygon, 0, size - 1).astype(np.int32)


class FaceMasker:
    """Generate a binary face mask from an AlignedFace.

    Uses the 68-point landmarks already computed during alignment —
    no extra model or library needed beyond insightface.

    Output convention: face pixels = 0, background = 255.
    This matches the dataloader which applies ``1 - mask`` before loss.
    """

    def __call__(self, face: AlignedFace) -> np.ndarray:
        h, w = face.image.shape[:2]
        polygon = _face_polygon(face.landmarks_68, min(h, w))
        mask = np.full((h, w), 255, dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 0)
        return mask
