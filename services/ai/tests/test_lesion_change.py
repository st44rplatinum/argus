"""Lesion measurement: right volume sharp or blurred, and air refused."""

from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk

from app.tasks.lesion_change import measure_lesion


def liver_with_lesion(blur_mm=0.0):
    """Liver at 30 HU with a 9 mm radius sphere at -10 HU, 1 mm voxels."""
    z, y, x = np.indices((40, 40, 40))
    sphere = (z - 20) ** 2 + (y - 20) ** 2 + (x - 20) ** 2 <= 81
    image = sitk.GetImageFromArray(np.where(sphere, -10.0, 30.0).astype(np.float32))
    if blur_mm:
        image = sitk.DiscreteGaussian(image, variance=blur_mm**2)
    return image, float(sphere.sum())


@pytest.mark.parametrize("blur_mm", [0.0, 1.0])
def test_half_way_threshold_keeps_the_volume(blur_mm):
    # a blurred edge counts half in, half out; the curve still costs a
    # little (the radius shrinks by about blur^2 / radius), so small lesions
    # under heavy blur read small
    image, truth = liver_with_lesion(blur_mm)
    measured = measure_lesion(image, (20.0, 20.0, 20.0))
    assert measured["volume_mm3"] == pytest.approx(truth, rel=0.08)


def test_air_is_refused():
    image = sitk.GetImageFromArray(np.full((20, 20, 20), -950.0, np.float32))
    with pytest.raises(ValueError, match="air or gas"):
        measure_lesion(image, (10.0, 10.0, 10.0))


def test_a_seed_near_the_edge_still_measures_the_whole_lesion():
    # 7 mm off the centre of a 9 mm lesion: the seed steps inward first
    image, truth = liver_with_lesion(1.0)
    measured = measure_lesion(image, (27.0, 20.0, 20.0))
    assert measured["volume_mm3"] == pytest.approx(truth, rel=0.08)


def scene():
    """Liver at 30 HU with a dark lesion (-10), a bright one (70) and a block of fat (-100)."""
    z, y, x = np.indices((40, 80, 80))
    ball = lambda cy, cx: (z - 20) ** 2 + (y - cy) ** 2 + (x - cx) ** 2 <= 49
    array = np.full((40, 80, 80), 30.0, np.float32)
    array[ball(20, 20)] = -10.0
    array[ball(20, 55)] = 70.0
    array[:, 60:, :] = -100.0
    return sitk.GetImageFromArray(array)


def test_tracking_accepts_the_same_kind_of_lesion():
    from app.tasks.lesion_change import measure_lesion

    measured = measure_lesion(scene(), (20.0, 20.0, 20.0), like=-40.0)
    assert measured["contrast_hu"] == pytest.approx(-40, abs=3)


@pytest.mark.parametrize(
    "point, why",
    [
        ((38.0, 38.0, 20.0), "plain liver"),
        ((55.0, 20.0, 20.0), "a bright lesion when tracking a dark one"),
        ((40.0, 57.0, 20.0), "fat next to the liver"),
    ],
)
def test_tracking_refuses_what_is_not_the_lesion(point, why):
    from app.tasks.lesion_change import NotSeen, measure_lesion

    with pytest.raises(NotSeen):
        measure_lesion(scene(), point, like=-40.0)
