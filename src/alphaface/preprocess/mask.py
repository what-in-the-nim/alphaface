from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

from .align import AlignedFace

# BiSeNet 19-class labels (index = class id):
#   0: background,  1: skin,    2: l_brow,  3: r_brow,  4: l_eye,
#   5: r_eye,       6: eye_g,   7: l_ear,   8: r_ear,   9: ear_r,
#  10: nose,       11: mouth,  12: u_lip,  13: l_lip,  14: neck,
#  15: neck_l,     16: cloth,  17: hair,   18: hat

# Classes treated as "face" (masked to 0); everything else → 255.
_FACE_CLASSES = frozenset([1, 2, 3, 4, 5, 10, 11, 12, 13])

_INPUT_SIZE = (512, 512)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess(image_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, _INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    normalized = (resized.astype(np.float32) / 255.0 - _MEAN) / _STD
    return np.expand_dims(normalized.transpose(2, 0, 1), 0)  # BCHW


def _postprocess(output: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
    seg = output.squeeze(0).argmax(0).astype(np.uint8)
    return cv2.resize(seg, original_size, interpolation=cv2.INTER_NEAREST)


class FaceMasker:
    """Generate a binary face mask using BiSeNet semantic segmentation (ONNX).

    Output convention: face pixels = 0, background = 255.
    This matches the dataloader which applies ``1 - mask`` before loss.

    Args:
        model_path: Path to resnet18.onnx or resnet34.onnx from yakhyo/face-parsing.
        device: ``"cuda"`` to prefer CUDA, ``"cpu"`` for CPU-only.
    """

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda" and ort.get_device() == "GPU"
            else ["CPUExecutionProvider"]
        )
        self._session = ort.InferenceSession(model_path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]

    def __call__(self, face: AlignedFace) -> np.ndarray:
        h, w = face.image.shape[:2]
        tensor = _preprocess(face.image)
        outputs = self._session.run(self._output_names, {self._input_name: tensor})
        seg = _postprocess(outputs[0], (w, h))

        face_mask = np.isin(seg, list(_FACE_CLASSES))
        result = np.full((h, w), 255, dtype=np.uint8)
        result[face_mask] = 0
        return result
