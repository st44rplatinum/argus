"""Volume to NIfTI, for models that only speak NIfTI.

The inverse of scripts/nifti_to_dicom.py, and the same warning applies: get
the direction wrong and the segmentation comes back mirrored, which on a
roughly symmetric abdomen looks entirely reasonable.

SimpleITK is handed LPS and writes RAS, so the conversion is its problem
rather than ours.
"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk


def volume_to_sitk(volume) -> sitk.Image:
    """A Volume as a SimpleITK image, geometry intact."""
    image = sitk.GetImageFromArray(volume.array.astype(np.float32))

    # Volume.spacing is (z, y, x); sitk wants (x, y, z).
    z_spacing, row_spacing, column_spacing = volume.spacing
    image.SetSpacing((float(column_spacing), float(row_spacing), float(z_spacing)))
    image.SetOrigin(tuple(float(v) for v in volume.origin))

    # ImageOrientationPatient is [direction of increasing column index,
    # direction of increasing row index]. The slice axis is their cross
    # product — the same normal the slice ordering already relies on.
    orientation = np.array(volume.orientation, dtype=np.float64)
    axis_x = orientation[0:3]
    axis_y = orientation[3:6]
    axis_z = np.cross(axis_x, axis_y)

    # sitk wants the 3x3 row-major, with column k the direction of axis k.
    direction = np.column_stack([axis_x, axis_y, axis_z])
    image.SetDirection(tuple(direction.flatten()))

    return image


def sitk_like(reference: sitk.Image, array: np.ndarray) -> sitk.Image:
    """An array wearing another image's geometry."""
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(reference)
    return image
