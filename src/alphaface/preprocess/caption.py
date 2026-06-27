from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

# Conditional prompt steers BLIP toward face-attribute descriptions that
# match the style expected by the CLIP text-loss in training.
_PROMPT = "a face photo of a person who"


class FaceCaptioner:
    """Generate a face-focused text description using BLIP.

    Produces captions in the style of the training data:
    gaze direction, background, accessories, facial hair, occlusions.

    Requires: pip install transformers
    Model weights (~1 GB) are downloaded from HuggingFace on first use.
    """

    def __init__(self, device: str | None = None) -> None:
        from transformers import BlipForConditionalGeneration, BlipProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large",
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, image_bgr: np.ndarray) -> str:
        pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        inputs = self.processor(pil_img, text=_PROMPT, return_tensors="pt").to(self.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=80,
            num_beams=4,
            repetition_penalty=1.3,
        )
        caption = self.processor.decode(output[0], skip_special_tokens=True)
        # Strip the prompt prefix that BLIP echoes back
        if caption.lower().startswith(_PROMPT.lower()):
            caption = (_PROMPT + caption[len(_PROMPT) :]).strip()
        return caption
