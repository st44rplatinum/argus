"""A written series is stored the way viewers expect, without moving anything."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from nifti_to_dicom import write_series  # noqa: E402


def test_rows_start_at_the_front_and_voxels_stay_put(tmp_path):
    # stored like a RAS NIfTI: index x runs to the patient's right, y to the front
    array = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.5, 2.0, 3.0))
    image.SetDirection((-1, 0, 0, 0, -1, 0, 0, 0, 1))
    image.SetOrigin((10.0, 20.0, 30.0))

    write_series(image, tmp_path, "test", "TEST")
    slices = sorted((pydicom.dcmread(p) for p in tmp_path.glob("*.dcm")), key=lambda d: d.ImagePositionPatient[2])

    assert [float(v) for v in slices[0].ImageOrientationPatient] == [1, 0, 0, 0, 1, 0]

    # every voxel is still at the same place in the body
    for index in [(0, 0, 0), (5, 4, 3), (2, 1, 2)]:
        point = np.array(image.TransformIndexToPhysicalPoint(index))
        value = array[index[2], index[1], index[0]]
        ds = min(slices, key=lambda d: abs(d.ImagePositionPatient[2] - point[2]))
        offset = point - np.array(ds.ImagePositionPatient, float)
        row = round(offset[1] / float(ds.PixelSpacing[0]))
        col = round(offset[0] / float(ds.PixelSpacing[1]))
        assert ds.pixel_array[row, col] == value
