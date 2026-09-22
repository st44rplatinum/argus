"""Geometry tests for series assembly.

These build datasets by hand rather than touching a PACS. The cases are the
ones that produce plausible-looking, silently wrong volumes: reversed slice
order, spacing that disagrees with SliceThickness, and two reconstructions
sharing one series UID.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian

from app.dicomweb import assemble, slice_normal

STUDY = "1.2.3"
SERIES = "1.2.3.4"


def make_slice(
    index: int,
    z: float,
    value: int,
    *,
    orientation: tuple[float, ...] = (1, 0, 0, 0, 1, 0),
    thickness: float = 5.0,
) -> Dataset:
    """One axial slice whose pixels are all `value`, sitting at depth `z`."""
    ds = Dataset()
    ds.SOPInstanceUID = f"{SERIES}.{index}"
    ds.InstanceNumber = index
    ds.Modality = "CT"
    ds.ImageOrientationPatient = list(orientation)
    ds.ImagePositionPatient = [0.0, 0.0, z]
    ds.PixelSpacing = [0.5, 0.75]
    ds.SliceThickness = thickness
    ds.RescaleSlope = 1
    ds.RescaleIntercept = -1024

    # Real pixel data rather than a stubbed array, so the decode path that
    # assemble() actually uses is the one under test.
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = CTImageStorage
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds.Rows = 4
    ds.Columns = 4
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = np.full((4, 4), value, dtype=np.uint16).tobytes()
    return ds


def test_normal_is_perpendicular_to_both_axes():
    normal = slice_normal([1, 0, 0, 0, 1, 0])
    assert np.allclose(normal, [0, 0, 1])


def test_slices_are_ordered_by_position_not_instance_number():
    # InstanceNumber ascends while position descends — the two disagree, and
    # geometry has to win.
    instances = [
        make_slice(index=1, z=20.0, value=1124),
        make_slice(index=2, z=10.0, value=1074),
        make_slice(index=3, z=0.0, value=1024),
    ]

    volume = assemble(instances, STUDY, SERIES)

    # Ascending depth means the z=0 slice (0 HU) comes first.
    assert [round(float(v)) for v in volume.array[:, 0, 0]] == [0, 50, 100]
    assert volume.sop_uids == (f"{SERIES}.3", f"{SERIES}.2", f"{SERIES}.1")


def test_spacing_comes_from_positions_not_slice_thickness():
    # Overlapping reconstruction: 5 mm thick slices every 2 mm.
    instances = [
        make_slice(index=i, z=i * 2.0, value=1024, thickness=5.0) for i in range(4)
    ]

    volume = assemble(instances, STUDY, SERIES)

    assert volume.spacing == pytest.approx((2.0, 0.5, 0.75))


def test_duplicate_positions_are_rejected():
    # Two reconstructions filed under one series UID.
    instances = [
        make_slice(index=1, z=0.0, value=1024),
        make_slice(index=2, z=0.0, value=1030),
        make_slice(index=3, z=2.0, value=1024),
        make_slice(index=4, z=2.0, value=1030),
    ]

    with pytest.raises(ValueError, match="duplicate positions"):
        assemble(instances, STUDY, SERIES)


def test_irregular_spacing_is_rejected():
    instances = [
        make_slice(index=0, z=0.0, value=1024),
        make_slice(index=1, z=2.0, value=1024),
        make_slice(index=2, z=9.0, value=1024),  # gap
    ]

    with pytest.raises(ValueError, match="irregular slice spacing"):
        assemble(instances, STUDY, SERIES)


def test_rescale_is_applied():
    instances = [make_slice(index=0, z=0.0, value=1024)]

    volume = assemble(instances, STUDY, SERIES)

    # Stored 1024 with intercept -1024 is water, 0 HU.
    assert float(volume.array[0, 0, 0]) == pytest.approx(0.0)


def test_empty_series_is_an_error():
    with pytest.raises(ValueError, match="no instances"):
        assemble([], STUDY, SERIES)
