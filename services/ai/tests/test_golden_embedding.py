"""Each step of the embedding still gives the saved answer (golden/embedding)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app import encoder

GOLDEN = Path(__file__).parent / "golden" / "embedding"
MANIFEST = json.loads((GOLDEN / "manifest.json").read_text(encoding="utf-8"))


def load():
    ct = np.load(GOLDEN / "ct.npy").astype(np.float32)
    return ct, np.load(GOLDEN / "mask.npy"), tuple(MANIFEST["spacing"])


def test_manifest_is_for_this_encoder():
    assert MANIFEST["encoder_id"] == encoder.ENCODER_ID
    assert MANIFEST["preprocess_version"] == encoder.PREPROCESS_VERSION


def test_stages_2_to_5_match():
    ct, mask, spacing = load()
    images, weights = encoder._prepare(ct, mask.astype(bool), spacing)

    np.testing.assert_array_equal(encoder.select_slices(mask.astype(bool)), np.load(GOLDEN / "slices.npy"))
    np.testing.assert_allclose(np.stack(images), np.load(GOLDEN / "cropped.npy"), atol=1e-6)
    np.testing.assert_allclose(encoder.resize(images[:2]), np.load(GOLDEN / "resized.npy"), atol=1e-6)
    np.testing.assert_allclose(np.stack(weights), np.load(GOLDEN / "weights.npy"), atol=1e-6)


@pytest.mark.parametrize("backend", ["torch", "onnx"])
def test_embedding_matches(backend):
    if backend == "onnx" and not encoder.ONNX_PATH.is_file():
        pytest.skip("run scripts/export_dinov2_onnx.py")
    ct, mask, spacing = load()
    vector = encoder.organ_embedding(ct, mask, spacing, backend=backend)
    # the onnx row is the one that matters: the file can replace torch
    assert float(vector @ np.load(GOLDEN / "embedding.npy")) > 0.9999
