"""Export the DINOv2 encoder to ONNX, so it runs without PyTorch.

Patch tokens only, the same slice encoder.py takes, with a dynamic batch.
Writes weights/dinov2s_224.onnx (~88 MB, gitignored).

  python scripts/export_dinov2_onnx.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "ai"))

from app.encoder import IMAGE_SIZE, ONNX_PATH, load_model  # noqa: E402


class PatchTokens(torch.nn.Module):
    """The timm model, returning patch tokens with the CLS token dropped."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.skip = model.num_prefix_tokens

    def forward(self, images):
        return self.model.forward_features(images)[:, self.skip :, :]


def main() -> None:
    # the exporter prints emoji, which a Windows code page console can't
    sys.stdout.reconfigure(encoding="utf-8")
    model = PatchTokens(load_model()).eval()
    example = torch.zeros(2, 3, IMAGE_SIZE, IMAGE_SIZE)

    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (example,),
        str(ONNX_PATH),
        input_names=["images"],
        output_names=["tokens"],
        # the batch size changes per organ (one image per slice)
        dynamic_shapes={"images": {0: torch.export.Dim("batch", min=1, max=64)}},
        external_data=False,
    )

    digest = hashlib.sha256(ONNX_PATH.read_bytes()).hexdigest()
    print(f"wrote {ONNX_PATH.relative_to(ROOT)}, {ONNX_PATH.stat().st_size / 1e6:.1f} MB")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
