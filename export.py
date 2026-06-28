from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from alphaface.lit_module import AlphaFaceLitModule
from alphaface.models.swapper_alphaface import remap_legacy_swapper_state_dict


def _select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _strip_prefix(state_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}


def _extract_swapper_state(checkpoint: Any) -> dict[str, Any]:
    """Support the checkpoint shapes used by demo, raw export, and Lightning training."""
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint to be a dict, got {type(checkpoint)!r}")

    if "swapper" in checkpoint and isinstance(checkpoint["swapper"], dict):
        return checkpoint["swapper"]

    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        state_dict = checkpoint["state_dict"]
        for prefix in ("model.swapper.", "swapper."):
            stripped = _strip_prefix(state_dict, prefix)
            if stripped:
                return stripped
        return state_dict

    return checkpoint


def export_onnx(
    checkpoint_path: Path,
    output_path: Path,
    device: str,
    opset: int,
    batch_size: int,
    dynamic_batch: bool,
    verify: bool,
) -> None:
    device = _select_device(device)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lit = AlphaFaceLitModule({"use_checkpoint": False}, device=device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    swapper_state = remap_legacy_swapper_state_dict(_extract_swapper_state(checkpoint))
    lit.model.swapper.load_state_dict(swapper_state)
    lit.model.eval().to(device)

    target = torch.randn(batch_size, 3, 256, 256, device=device)
    source = torch.randn(batch_size, 3, 112, 112, device=device)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "target": {0: "batch"},
            "source": {0: "batch"},
            "swapped": {0: "batch"},
        }

    with torch.no_grad():
        torch.onnx.export(
            lit.model,
            (target, source),
            str(output_path),
            input_names=["target", "source"],
            output_names=["swapped"],
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
        )

    if verify:
        import onnx

        model = onnx.load(str(output_path))
        onnx.checker.check_model(model)

    print(f"Exported ONNX model: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export AlphaFace inference model to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("./models/alphaface_demo.pt"),
        help="Path to AlphaFace checkpoint (.pt or Lightning .ckpt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./models/alphaface_swapper.onnx"),
        help="Destination ONNX file",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Device used for export tracing",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--batch-size", type=int, default=1, help="Dummy batch size used during tracing")
    parser.add_argument(
        "--static-batch",
        action="store_true",
        help="Do not mark the batch dimension as dynamic",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run onnx.checker.check_model after export. Requires the onnx package.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        device=args.device,
        opset=args.opset,
        batch_size=args.batch_size,
        dynamic_batch=not args.static_batch,
        verify=args.verify,
    )


if __name__ == "__main__":
    main()
