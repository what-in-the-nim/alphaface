from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from insightface.app import FaceAnalysis

# FFHQ 5-point reference landmarks at 512×512, scaled to target size at call time.
# Order: left-eye, right-eye, nose-tip, left-mouth, right-mouth.
_FFHQ_REF_512 = np.array(
    [
        [192.98138, 239.94708],
        [318.90277, 240.19366],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ],
    dtype=np.float32,
)


@dataclass
class AlignedFace:
    """An FFHQ-aligned face crop together with its 68-point landmarks.

    ``landmarks_68`` are already transformed into the aligned image's
    pixel coordinate space, so they can be used directly for masking.
    """

    image: np.ndarray  # BGR, shape (size, size, 3)
    landmarks_68: np.ndarray  # float32, shape (68, 2)


def _affine_align(
    image: np.ndarray,
    landmarks_5: np.ndarray,
    landmarks_68: np.ndarray,
    size: int,
) -> AlignedFace | None:
    ref = _FFHQ_REF_512 * (size / 512.0)
    M, _ = cv2.estimateAffinePartial2D(landmarks_5, ref, method=cv2.LMEDS)
    if M is None:
        return None
    aligned_img = cv2.warpAffine(image, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # Transform 68 landmarks (xy part only) into aligned coordinate space.
    pts_2d = landmarks_68[:, :2].astype(np.float32)
    aligned_lm = cv2.transform(pts_2d.reshape(-1, 1, 2), M).reshape(-1, 2)
    return AlignedFace(image=aligned_img, landmarks_68=aligned_lm)


class FaceAligner:
    """Detect faces and produce FFHQ-aligned crops via InsightFace.

    Requires: pip install insightface onnxruntime  (or onnxruntime-gpu)
    Model weights (~280 MB) are downloaded automatically on first use.
    The buffalo_l pack includes detection, 68-point 3D landmarks, and
    recognition — no extra downloads needed.
    """

    def __init__(
        self,
        det_size: tuple[int, int] = (640, 640),
        device: str = "cpu",
    ) -> None:
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "mps":
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.app = FaceAnalysis(providers=providers)
        self.app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=det_size)

    def align_all(self, image_bgr: np.ndarray, size: int = 256) -> list[AlignedFace]:
        """Return one AlignedFace per detected face."""
        results = []
        for face in self.app.get(image_bgr):
            af = _affine_align(image_bgr, face.kps, face.landmark_3d_68, size)
            if af is not None:
                results.append(af)
        return results

    def align_largest(self, image_bgr: np.ndarray, size: int = 256) -> AlignedFace | None:
        """Return only the largest detected face, or None if none found."""
        faces = self.app.get(image_bgr)
        if not faces:
            return None
        largest = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        return _affine_align(image_bgr, largest.kps, largest.landmark_3d_68, size)
