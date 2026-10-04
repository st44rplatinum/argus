"""Registration recovers a known move, in the direction the jobs rely on."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from app.register import register_rigid


def phantom():
    """A body (0 HU) in air with a few bright blobs, 2 mm voxels."""
    z, y, x = np.indices((48, 64, 64))
    body = ((y - 32) ** 2 + (x - 32) ** 2 < 26**2) & (z > 4) & (z < 44)
    array = np.where(body, 0.0, -1000.0).astype(np.float32)
    for cz, cy, cx in ((15, 25, 20), (30, 40, 30), (24, 22, 44), (35, 28, 38)):
        array[(z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2 < 16] = 300.0
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((2.0, 2.0, 2.0))
    return image


def test_rigid_maps_fixed_points_to_moving_points():
    fixed = phantom()
    move = sitk.Euler3DTransform()
    move.SetCenter(fixed.TransformContinuousIndexToPhysicalPoint([32, 32, 24]))
    move.SetRotation(0.03, -0.02, 0.04)
    move.SetTranslation((6.0, -4.0, 3.0))
    # moving(p) = fixed(T(p)), so what sits at p in fixed sits at T^-1(p) in moving
    moving = sitk.Resample(fixed, fixed, move, sitk.sitkLinear, -1000.0)
    expected = move.GetInverse()

    found = register_rigid(fixed, moving).transform

    for point in ((40.0, 50.0, 30.0), (70.0, 60.0, 60.0), (55.0, 80.0, 45.0)):
        error = np.linalg.norm(np.subtract(found.TransformPoint(point), expected.TransformPoint(point)))
        assert error < 1.0
