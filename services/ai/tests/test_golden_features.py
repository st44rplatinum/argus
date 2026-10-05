"""The 11 features still give the saved answers (golden/features)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.features import FEATURE_VERSION, NAMES, organ_features

GOLDEN = Path(__file__).parent / "golden" / "features"
MANIFEST = json.loads((GOLDEN / "manifest.json").read_text(encoding="utf-8"))


def test_manifest_is_for_this_feature_version():
    assert MANIFEST["feature_version"] == FEATURE_VERSION, "features changed: regenerate deliberately"
    assert MANIFEST["names"] == list(NAMES)


@pytest.mark.parametrize("name", sorted(MANIFEST["cases"]))
def test_case_matches(name):
    case = MANIFEST["cases"][name]
    ct = np.load(GOLDEN / case["ct"]).astype(np.float32)
    mask = np.load(GOLDEN / case["mask"])
    values = organ_features(ct, mask, tuple(case["spacing"]))

    if case["expected"] is None:
        assert values is None
    else:
        # if this fails and the version didn't change, a formula changed
        # without bumping FEATURE_VERSION
        np.testing.assert_allclose(values, case["expected"], rtol=1e-6, atol=1e-9)
