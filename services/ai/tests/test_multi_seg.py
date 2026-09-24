"""Label renumbering for multi-structure segmentations.

A model that can emit 117 structures emits a handful per scan, and highdicom
needs segment numbers 1..N with no gaps. Getting the remap wrong relabels
organs — a liver described as a spleen, with the right voxels.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.dicom_seg import build_multi_segmentation

SERIES = "1.2.3.4"
NAMES = {1: "spleen", 5: "liver", 17: "thyroid_gland", 99: "autochthon_left"}


class FakeDataset:
    def __init__(self, index: int):
        self.SOPInstanceUID = f"{SERIES}.{index}"


@pytest.fixture
def captured(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "app.dicom_seg.hd.seg.Segmentation", lambda **kwargs: seen.update(kwargs)
    )
    return seen


def labelmap_with(values):
    """A 4-slice label map, one structure per slice."""
    array = np.zeros((4, 3, 3), dtype=np.uint16)
    for slice_index, value in enumerate(values):
        array[slice_index, 1, 1] = value
    return array


def test_sparse_labels_become_consecutive(captured):
    # The model's own numbers: 1, 5, 17, 99 — gaps everywhere.
    labelmap = labelmap_with([1, 5, 17, 99])
    sources = [FakeDataset(i) for i in range(4)]
    sop_uids = [f"{SERIES}.{i}" for i in range(4)]

    build_multi_segmentation(sources, labelmap, sop_uids, NAMES)

    numbers = [d.SegmentNumber for d in captured["segment_descriptions"]]
    assert numbers == [1, 2, 3, 4]

    written = captured["pixel_array"]
    assert sorted(int(v) for v in np.unique(written)) == [0, 1, 2, 3, 4]


def test_labels_keep_their_names_through_the_remap(captured):
    labelmap = labelmap_with([1, 5, 17, 99])
    sources = [FakeDataset(i) for i in range(4)]
    sop_uids = [f"{SERIES}.{i}" for i in range(4)]

    build_multi_segmentation(sources, labelmap, sop_uids, NAMES)

    labels = [d.SegmentLabel for d in captured["segment_descriptions"]]
    # Ascending original label order, so liver stays liver.
    assert labels == ["spleen", "liver", "thyroid gland", "autochthon left"]


def test_absent_structures_are_not_described(captured):
    # Only two of the four known structures appear.
    labelmap = labelmap_with([0, 5, 0, 17])
    sources = [FakeDataset(i) for i in range(4)]
    sop_uids = [f"{SERIES}.{i}" for i in range(4)]

    build_multi_segmentation(sources, labelmap, sop_uids, NAMES)

    labels = [d.SegmentLabel for d in captured["segment_descriptions"]]
    assert labels == ["liver", "thyroid gland"]


def test_unknown_label_still_gets_a_segment(captured):
    labelmap = labelmap_with([0, 0, 0, 404])
    sources = [FakeDataset(i) for i in range(4)]
    sop_uids = [f"{SERIES}.{i}" for i in range(4)]

    build_multi_segmentation(sources, labelmap, sop_uids, NAMES)

    assert captured["segment_descriptions"][0].SegmentLabel == "structure 404"


def test_empty_label_map_is_an_error(captured):
    sources = [FakeDataset(i) for i in range(4)]
    sop_uids = [f"{SERIES}.{i}" for i in range(4)]

    with pytest.raises(ValueError, match="empty"):
        build_multi_segmentation(
            sources, np.zeros((4, 3, 3), dtype=np.uint16), sop_uids, NAMES
        )
