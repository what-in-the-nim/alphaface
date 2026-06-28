from __future__ import annotations

import asyncio
import base64

import cv2
import numpy as np

_PROMPT = (
    "Describe pose, back-ground, facial accessories, and all obstacles covering"
    " the face area in the given face image. Only 70 words are allowed."
)


class FaceCaptioner:
    """Generate face descriptions via an OpenAI-compatible vision API.

    Fires all requests concurrently (up to ``concurrency`` at a time) using
    ``caption_many``; ``__call__`` is a convenience wrapper for a single image.

    Args:
        base_url: Base URL of the OpenAI-compatible server
            (e.g. ``"http://localhost:11434/v1"``).
        model: Model name to request (e.g. ``"llava:7b"``).
        api_key: API key. Use ``"none"`` for local servers that don't require one.
        concurrency: Max simultaneous in-flight requests.
        timeout: Per-request timeout in seconds.

    Requires: ``pip install aiohttp``
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "llava:7b",
        api_key: str = "none",
        concurrency: int = 4,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.concurrency = concurrency
        self.timeout = timeout

    def _encode(self, image_bgr: np.ndarray) -> str:
        _, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return base64.b64encode(buf.tobytes()).decode()

    async def _request(self, session, sem: asyncio.Semaphore, image_bgr: np.ndarray) -> str:
        import aiohttp

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{self._encode(image_bgr)}"},
                        },
                        {"type": "text", "text": _PROMPT},
                    ],
                }
            ],
            "max_tokens": 3000,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with sem:
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
        return data["choices"][0]["message"]["content"].strip()

    async def _run(self, images: list[np.ndarray], *, progress: bool = False) -> list[str]:
        import aiohttp
        from tqdm import tqdm

        sem = asyncio.Semaphore(self.concurrency)

        async def _indexed_request(session, idx: int, image: np.ndarray) -> tuple[int, str]:
            return idx, await self._request(session, sem, image)

        async with aiohttp.ClientSession() as session:
            tasks = [asyncio.create_task(_indexed_request(session, idx, img)) for idx, img in enumerate(images)]
            if not progress:
                return [caption for _, caption in sorted(await asyncio.gather(*tasks))]

            results: list[str | None] = [None] * len(tasks)
            with tqdm(total=len(tasks), desc="Captioning", unit="img") as pbar:
                for task in asyncio.as_completed(tasks):
                    idx, caption = await task
                    results[idx] = caption
                    pbar.update(1)
            return [r or "" for r in results]

    def caption_many(self, images: list[np.ndarray], *, progress: bool = False) -> list[str]:
        """Caption a batch of images with concurrent requests."""
        if not images:
            return []
        return asyncio.run(self._run(images, progress=progress))

    def __call__(self, image_bgr: np.ndarray) -> str:
        return self.caption_many([image_bgr])[0]
