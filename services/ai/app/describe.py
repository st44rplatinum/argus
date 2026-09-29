from __future__ import annotations

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to, resample_to_output

from .config import ARGUS_THREADS
from .encoder import organ_embedding
from .features import organ_features


_TARGET_VOXEL_SIZE = 1.5
_VOXEL_TOLERANCE = 1e-3


def describe(ct: nib.Nifti1Image, mask: nib.Nifti1Image):
    """Canonical, 1.5 mm, then organ_features and organ_embedding.

    Returns (features, embedding). Either can be None when the organ is
    too small, same as the index builders.
    """
    # The learned embedding is orientation-sensitive, so canonicalise both
    # images before converting them to arrays.
    ct = nib.as_closest_canonical(ct)
    mask = nib.as_closest_canonical(mask)

    # Avoid interpolation when the CT is already at the exact target
    # resolution (within the tolerance used by the index pipeline).
    spacing = np.asarray(ct.header.get_zooms()[:3], dtype=np.float64)

    if not np.all(
        np.abs(spacing - _TARGET_VOXEL_SIZE) <= _VOXEL_TOLERANCE
    ):
        # Linear interpolation is appropriate for CT/HU values.
        ct = resample_to_output(
            ct,
            voxel_sizes=_TARGET_VOXEL_SIZE,
            order=1,
        )

        # Resample the mask onto the CT's new grid. Nearest-neighbour keeps
        # it binary and guarantees that CT and mask have identical geometry.
        mask = resample_from_to(
            mask,
            ct,
            order=0,
        )

    # Both index builders use the NIfTI array in [x, y, z] order.
    volume = np.asarray(ct.dataobj, dtype=np.float32)
    mask_array = np.asarray(mask.dataobj) > 0

    # A mask from another grid would describe the wrong voxels silently.
    if mask_array.shape != volume.shape:
        raise ValueError(f"mask {mask_array.shape} does not match CT {volume.shape}")
    spacing = tuple(float(x) for x in ct.header.get_zooms()[:3])

    # The two functions share the same >=100 voxel cutoff, so both return
    # None for an organ that is too small to describe.
    features = organ_features(
        volume,
        mask_array,
        spacing,
    )

    embedding = organ_embedding(
        volume,
        mask_array,
        spacing,
        threads=ARGUS_THREADS,
    )

    return features, embedding
