"""The SEG has to line up with the mask.

highdicom pairs frame i of the mask with source image i. The server returns
instances in arbitrary order — measured against this Orthanc, all 64 of the
phantom's slices came back out of geometric order — so handing it the raw
retrieve_series result puts every frame on the wrong slice, with the correct
shape and voxel count and no error raised.

These tests pin the reordering, because the failure is invisible.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.dicom_seg import build_segmentation

SERIES = "1.2.3.4"


class FakeDataset:
    """Just enough for the reordering logic; no pixel data required."""

    def __init__(self, index: int):
        self.SOPInstanceUID = f"{SERIES}.{index}"
        self.index = index


def test_sources_are_reordered_to_match_the_mask(monkeypatch):
    captured = {}

    def fake_segmentation(**kwargs):
        captured.update(kwargs)
        return "segmentation"

    monkeypatch.setattr("app.dicom_seg.hd.seg.Segmentation", fake_segmentation)

    # The server hands them back shuffled.
    shuffled = [FakeDataset(i) for i in (2, 0, 3, 1)]
    # The volume knows the geometric order.
    sop_uids = [f"{SERIES}.{i}" for i in range(4)]
    mask = np.zeros((4, 2, 2), dtype=np.uint8)

    build_segmentation(shuffled, mask, sop_uids)

    passed = [d.index for d in captured["source_images"]]
    assert passed == [0, 1, 2, 3], "sources must follow sop_uids, not server order"


def test_missing_slice_is_an_error(monkeypatch):
    monkeypatch.setattr("app.dicom_seg.hd.seg.Segmentation", lambda **kwargs: None)

    sources = [FakeDataset(i) for i in range(3)]
    sop_uids = [f"{SERIES}.{i}" for i in range(4)]
    mask = np.zeros((4, 2, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match="absent from the source datasets"):
        build_segmentation(sources, mask, sop_uids)


def test_depth_mismatch_is_an_error(monkeypatch):
    monkeypatch.setattr("app.dicom_seg.hd.seg.Segmentation", lambda **kwargs: None)

    sources = [FakeDataset(i) for i in range(4)]
    sop_uids = [f"{SERIES}.{i}" for i in range(4)]
    mask = np.zeros((3, 2, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match="does not match"):
        build_segmentation(sources, mask, sop_uids)
