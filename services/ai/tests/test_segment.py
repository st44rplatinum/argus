"""Thresholding, checked against a volume whose answer is known by construction."""

from __future__ import annotations

import numpy as np
import pytest

from app.segment import calculate_bounding_box, threshold_segment


class FakeVolume:
    def __init__(self, array):
        self.array = array


def volume_with_blob() -> FakeVolume:
    """Soft tissue everywhere, one bone-density cube at a known place."""
    array = np.full((10, 20, 20), 40.0, dtype=np.float32)
    array[2:5, 6:9, 11:15] = 700.0
    return FakeVolume(array)


def test_threshold_finds_exactly_the_blob():
    mask = threshold_segment(volume_with_blob(), 300, 3000)

    assert mask.dtype == np.uint8
    assert int(mask.sum()) == 3 * 3 * 4


def test_bounding_box_matches_the_blob():
    mask = threshold_segment(volume_with_blob(), 300, 3000)

    assert calculate_bounding_box(mask) == {
        # Inclusive indices, so the upper bound is one less than the slice stop.
        "z_min": 2,
        "z_max": 4,
        "row_min": 6,
        "row_max": 8,
        "col_min": 11,
        "col_max": 14,
    }


def test_bounds_are_inclusive():
    """An empty range still selects the voxels sitting exactly on it."""
    array = np.array([[[100.0, 200.0, 300.0]]], dtype=np.float32)
    mask = threshold_segment(FakeVolume(array), 200, 200)

    assert mask.tolist() == [[[0, 1, 0]]]


def test_no_match_has_no_bounding_box():
    mask = threshold_segment(volume_with_blob(), 5000, 6000)

    assert int(mask.sum()) == 0
    assert calculate_bounding_box(mask) is None


def test_non_3d_is_rejected():
    with pytest.raises(ValueError, match="3D"):
        threshold_segment(FakeVolume(np.zeros((4, 4), dtype=np.float32)), 0, 1)
