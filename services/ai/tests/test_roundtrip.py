"""Put a mask in, get the same mask out.

An integration test: it needs Orthanc up with the phantom loaded, and skips
otherwise. Worth the dependency, because it covers in one assertion what four
unit tests cover separately — slice ordering, bit packing, empty-frame
omission and the store path — and it is the test that will catch a regression
when the mask stops being a simple threshold.
"""

from __future__ import annotations

import highdicom as hd
import httpx
import numpy as np
import pytest

from app.dicom_seg import build_segmentation, store_segmentation
from app.dicomweb import client_for, load_series

# Deleting is not a DICOMweb operation, so cleanup goes through Orthanc's own
# REST API. A test that leaves objects behind is a test you stop running: four
# runs left four SEG series in the PACS and made "which one renders?" a real
# question in the viewer.
ORTHANC = "http://localhost:8042"


def delete_series(series_uid: str) -> None:
    """Remove a series from Orthanc by its SeriesInstanceUID."""
    try:
        found = httpx.post(f"{ORTHANC}/tools/lookup", content=series_uid, timeout=30)
        for item in found.json():
            if item.get("Type") == "Series":
                httpx.delete(f"{ORTHANC}/series/{item['ID']}", timeout=30)
    except Exception:
        # Cleanup failing must not fail the test it is cleaning up after.
        pass


@pytest.fixture
def stored_seg():
    """Track SEG series created by a test and remove them afterwards."""
    created: list[str] = []
    yield created
    for series_uid in created:
        delete_series(series_uid)


@pytest.fixture(scope="module")
def phantom():
    """The phantom series, or a skip if the PACS is not up."""
    try:
        client = client_for()
        studies = client.search_for_studies()
    except Exception as exc:
        pytest.skip(f"no DICOMweb server: {exc}")

    if not studies:
        pytest.skip("PACS is empty; run scripts/push_dicom.py")

    study_uid = studies[0]["0020000D"]["Value"][0]

    for series in client.search_for_series(study_uid):
        if series.get("00080060", {}).get("Value", [None])[0] == "CT":
            series_uid = series["0020000E"]["Value"][0]
            break
    else:
        pytest.skip("no CT series in the PACS")

    sources = client.retrieve_series(
        study_instance_uid=study_uid, series_instance_uid=series_uid
    )
    volume = load_series(client, study_uid, series_uid)
    return client, study_uid, sources, volume


def test_mask_survives_the_round_trip(phantom, stored_seg):
    client, study_uid, sources, volume = phantom

    # A mask with deliberately empty slices at both ends, so frame omission is
    # exercised rather than assumed away. The phantom's spine spans every
    # slice, which would hide it.
    original = np.zeros(volume.shape, dtype=np.uint8)
    original[10:20, 100:140, 90:130] = 1

    segmentation = build_segmentation(sources, original, volume.sop_uids)
    seg_series_uid = store_segmentation(client, segmentation)
    stored_seg.append(seg_series_uid)

    stored = client.retrieve_series(
        study_instance_uid=study_uid, series_instance_uid=seg_series_uid
    )
    assert len(stored) == 1, "a multi-frame SEG is one instance, not one per frame"

    read_back = hd.seg.Segmentation.from_dataset(stored[0], copy=False)
    recovered = read_back.get_pixels_by_source_instance(
        source_sop_instance_uids=list(volume.sop_uids),
        segment_numbers=[1],
    )[..., 0]

    assert recovered.shape == original.shape
    # Every voxel, not just the count — a mask scattered onto the wrong slices
    # keeps its count exactly.
    assert np.array_equal(recovered.astype(np.uint8), original)


def test_empty_frames_are_omitted(phantom):
    """Confirms the behaviour that will surprise us at P2 step 5."""
    _, _, sources, volume = phantom

    mask = np.zeros(volume.shape, dtype=np.uint8)
    mask[10:20] = 1  # ten slices of sixty-four

    segmentation = build_segmentation(sources, mask, volume.sop_uids)

    assert int(segmentation.NumberOfFrames) == 10, (
        "omit_empty_frames defaults to True, so frame count follows the mask "
        "rather than the source series"
    )
