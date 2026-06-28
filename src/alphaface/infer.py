from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as L
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .lit_module import AlphaFaceLitModule
from .models.swapper_alphaface import remap_legacy_swapper_state_dict


def _normalize_127_5(img: torch.Tensor) -> torch.Tensor:
    return ((img * 255.0).int() / 127.5) - 1.0


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    if t.dim() == 4:
        t = t[0]
    t = t.permute(1, 2, 0)
    t = (t * 255.0).clamp(0, 255).byte()
    return Image.fromarray(t.cpu().numpy())


class _SinglePairDataset(Dataset):
    def __init__(self, source: str | Path, target: str | Path) -> None:
        t_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        s_transform = transforms.Compose(
            [transforms.Resize((112, 112)), transforms.ToTensor(), transforms.Lambda(_normalize_127_5)]
        )
        self.source = s_transform(Image.open(source).convert("RGB"))
        self.target = t_transform(Image.open(target).convert("RGB"))

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.target, self.source


def run(
    source: str | Path,
    target: str | Path,
    checkpoint: str | Path,
    output: str | Path,
    device: str | None = None,
) -> Path:
    model = AlphaFaceLitModule({"use_checkpoint": False}, device=device or "auto")

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("swapper", ckpt)
    model.model.swapper.load_state_dict(remap_legacy_swapper_state_dict(state))

    loader = DataLoader(_SinglePairDataset(source, target), batch_size=1, shuffle=False)
    trainer = L.Trainer(accelerator="auto", devices=1, logger=False)
    swapped = trainer.predict(model, dataloaders=loader)[0]

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _tensor_to_pil(swapped).save(output)
    print(f"Saved: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="AlphaFace inference — swap the identity of source onto target")
    parser.add_argument("--source", required=True, help="Source image (identity to transfer)")
    parser.add_argument("--target", required=True, help="Target image (attributes to keep)")
    parser.add_argument("--checkpoint", default="./models/vit_b_fr_pgair.pt", help="Swapper checkpoint (.pt)")
    parser.add_argument("--output", default="./output/swapped.png", help="Output image path")
    parser.add_argument("--device", default=None, help="cpu | cuda | mps (auto-detected if omitted)")
    args = parser.parse_args()

    run(args.source, args.target, args.checkpoint, args.output, args.device)


if __name__ == "__main__":
    main()
