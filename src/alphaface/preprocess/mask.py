from __future__ import annotations

import cv2
import numpy as np

# MediaPipe FaceMesh face-oval contour indices (subset of the 468-point mesh).
_FACE_OVAL = [
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
]


class FaceMasker:
    """Generate a binary face mask from an aligned face image using MediaPipe FaceMesh.

    Output convention: face pixels = 0, background = 255.
    This matches the dataloader which applies ``1 - mask`` before computing loss.

    Requires: pip install mediapipe
    No external checkpoint download needed.
    """

    def __init__(self) -> None:
        import mediapipe as mp

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

    def __call__(self, image_bgr: np.ndarray) -> np.ndarray | None:
        """Return a uint8 HW mask or None if no face is detected."""
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None

        lm = result.multi_face_landmarks[0].landmark
        oval_pts = np.array(
            [[int(lm[i].x * w), int(lm[i].y * h)] for i in _FACE_OVAL],
            dtype=np.int32,
        )
        mask = np.full((h, w), 255, dtype=np.uint8)
        cv2.fillPoly(mask, [oval_pts], 0)
        return mask

    def close(self) -> None:
        self._face_mesh.close()

    def __enter__(self) -> FaceMasker:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
