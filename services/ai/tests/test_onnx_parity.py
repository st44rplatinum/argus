"""The exported ONNX encoder gives the same patch tokens as PyTorch."""

from __future__ import annotations

import numpy as np
import pytest

from app import encoder

pytestmark = pytest.mark.skipif(not encoder.ONNX_PATH.is_file(), reason="run scripts/export_dinov2_onnx.py")


def test_normalisation_matches_the_models_own():
    cfg = encoder.load_model().pretrained_cfg
    assert tuple(cfg["mean"]) == encoder.MEAN and tuple(cfg["std"]) == encoder.STD


# 2 is the size it was exported with; other sizes catch dynamic-batch bugs
@pytest.mark.parametrize("batch", [1, 8])
def test_onnx_matches_torch(batch):
    rng = np.random.default_rng(batch)
    x = encoder.network_input(rng.random((batch, 224, 224), dtype=np.float32))

    torch_tokens = encoder.patch_tokens(x, "torch")
    onnx_tokens = encoder.patch_tokens(x, "onnx")

    assert onnx_tokens.shape == (batch, 256, 384)
    # tight on purpose: a loose 0.99 would pass a wrong normalisation
    assert np.abs(onnx_tokens - torch_tokens).max() < 1e-3
    a = onnx_tokens / np.linalg.norm(onnx_tokens, axis=-1, keepdims=True)
    b = torch_tokens / np.linalg.norm(torch_tokens, axis=-1, keepdims=True)
    assert (a * b).sum(-1).min() > 0.9999
