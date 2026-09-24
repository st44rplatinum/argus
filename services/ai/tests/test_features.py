"""Shape and density features, against solids whose answers are known.

The point of hand-written features is that you can say what each one means.
These tests are that claim, checked: a sphere is round, a rod is not, and a
noisy organ reads busier than a uniform one.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.features import LEN, NAMES, organ_features

SPACING = (1.0, 1.0, 1.0)


def index(name: str) -> int:
    return NAMES.index(name)


def ball(radius: float, shape=(40, 40, 40)) -> np.ndarray:
    z, y, x = np.mgrid[: shape[0], : shape[1], : shape[2]]
    centre = np.array(shape) / 2
    return ((z - centre[0]) ** 2 + (y - centre[1]) ** 2 + (x - centre[2]) ** 2) <= radius**2


def test_vector_has_the_advertised_length():
    mask = ball(8)
    features = organ_features(np.full(mask.shape, 50.0, np.float32), mask, SPACING)

    assert features is not None
    assert features.shape == (LEN,)


def test_a_sphere_is_round():
    mask = ball(10)
    features = organ_features(np.full(mask.shape, 50.0, np.float32), mask, SPACING)

    # All three principal axes equal, so both ratios approach 1.
    assert features[index("elongation")] == pytest.approx(1.0, abs=0.05)
    assert features[index("flatness")] == pytest.approx(1.0, abs=0.05)


def test_a_rod_is_not_round():
    mask = np.zeros((60, 30, 30), dtype=bool)
    mask[5:55, 14:17, 14:17] = True
    features = organ_features(np.full(mask.shape, 50.0, np.float32), mask, SPACING)

    # Long in one direction only — the aorta comes out like this.
    assert features[index("elongation")] < 0.2
    assert features[index("flatness")] < 0.2


def test_a_disc_is_flat_but_not_elongated():
    mask = np.zeros((30, 60, 60), dtype=bool)
    mask[14:17, 10:50, 10:50] = True
    features = organ_features(np.full(mask.shape, 50.0, np.float32), mask, SPACING)

    assert features[index("elongation")] > 0.7   # wide in two directions
    assert features[index("flatness")] < 0.2     # thin in the third


def test_volume_follows_spacing():
    mask = ball(10)
    voxels = int(mask.sum())

    fine = organ_features(np.zeros(mask.shape, np.float32), mask, (1.0, 1.0, 1.0))
    coarse = organ_features(np.zeros(mask.shape, np.float32), mask, (2.0, 1.0, 1.0))

    assert fine[index("volume_ml")] == pytest.approx(voxels / 1000.0)
    # Twice the slice thickness is twice the volume for the same voxel count.
    assert coarse[index("volume_ml")] == pytest.approx(2 * fine[index("volume_ml")])


def test_density_is_read_from_inside_the_mask_only():
    mask = ball(10)
    volume = np.full(mask.shape, -1000.0, np.float32)   # air everywhere
    volume[mask] = 60.0                                  # soft tissue inside

    features = organ_features(volume, mask, SPACING)

    assert features[index("mean_hu")] == pytest.approx(60.0)
    assert features[index("std_hu")] == pytest.approx(0.0)


def test_noise_reads_as_busier():
    mask = ball(12)
    rng = np.random.default_rng(0)

    uniform = np.full(mask.shape, 60.0, np.float32)
    noisy = uniform + rng.normal(0, 25, mask.shape).astype(np.float32)

    quiet = organ_features(uniform, mask, SPACING)
    busy = organ_features(noisy, mask, SPACING)

    assert busy[index("gradient_mean")] > quiet[index("gradient_mean")] * 10
    assert busy[index("heterogeneity")] > quiet[index("heterogeneity")]


def test_a_sliver_is_not_an_organ():
    mask = np.zeros((20, 20, 20), dtype=bool)
    mask[0, 0, :5] = True  # five voxels at the edge of the field of view

    assert organ_features(np.zeros(mask.shape, np.float32), mask, SPACING) is None
