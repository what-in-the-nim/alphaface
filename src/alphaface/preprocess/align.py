from __future__ import annotations

import cv2
import numpy as np

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


def _affine_align(image: np.ndarray, landmarks_5: np.ndarray, size: int) -> np.ndarray | None:
    ref = _FFHQ_REF_512 * (size / 512.0)
    M, _ = cv2.estimateAffinePartial2D(landmarks_5, ref, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(image, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


class FaceAligner:
    """Detect faces and produce FFHQ-aligned crops via InsightFace.

    Requires: pip install insightface onnxruntime  (or onnxruntime-gpu)
    Model weights are downloaded automatically on first use (~200 MB).
    """

    def __init__(
        self,
        det_size: tuple[int, int] = (640, 640),
        device: str = "cpu",
    ) -> None:
        from insightface.app import FaceAnalysis

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self.app = FaceAnalysis(providers=providers)
        self.app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=det_size)

    def align_all(self, image_bgr: np.ndarray, size: int = 256) -> list[np.ndarray]:
        """Return one aligned BGR crop per detected face."""
        faces = self.app.get(image_bgr)
        results = []
        for face in faces:
            aligned = _affine_align(image_bgr, face.kps, size)
            if aligned is not None:
                results.append(aligned)
        return results

    def align_largest(self, image_bgr: np.ndarray, size: int = 256) -> np.ndarray | None:
        """Return only the largest detected face, or None if none found."""
        faces = self.app.get(image_bgr)
        if not faces:
            return None
        largest = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        return _affine_align(image_bgr, largest.kps, size)
