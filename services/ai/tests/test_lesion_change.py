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
