"""The AI service.

One endpoint for now. It exists to prove the plumbing — fetch a series over
DICOMweb, run something over it, report a number you can check by hand —
before any model is involved.
"""

from __future__ import annotations

import os

from dicomweb_client.api import DICOMwebClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .dicom_seg import build_segmentation, store_segmentation
from .dicomweb import client_for, load_series
from .segment import calculate_bounding_box, threshold_segment

app = FastAPI(title="Argus")

# One client for the process rather than one per request: it holds a session,
# and rebuilding it per call throws away connection reuse.
DICOMWEB_URL = os.environ.get("ARGUS_DICOMWEB", "http://localhost:8042/dicom-web")
_client: DICOMwebClient | None = None


def client() -> DICOMwebClient:
    global _client
    if _client is None:
        _client = client_for(DICOMWEB_URL)
    return _client


class SegmentRequest(BaseModel):
    study_uid: str
    series_uid: str

    # Bone is roughly >300 HU. The phantom's spine is 700.
    low: float = Field(default=300)
    high: float = Field(default=3000)

    store: bool = False


@app.post("/segment")
def segment(request: SegmentRequest):
    try:
        volume = load_series(client(), request.study_uid, request.series_uid)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=404, detail=f"Could not load series: {exc}"
        ) from exc

    mask = threshold_segment(volume, request.low, request.high)
    box = calculate_bounding_box(mask)

    result = {
        "study_uid": request.study_uid,
        "series_uid": request.series_uid,
        "shape": list(volume.shape),
        "spacing_mm": list(volume.spacing),
        "voxel_count": int(mask.sum()),
        # Inclusive indices, so a width is max - min + 1.
        "bounding_box": box,
        "seg_series_uid": None,
    }

    if not request.store:
        return result

    if box is None:
        # highdicom will not build a segmentation with no segmented voxels,
        # and storing an empty one would be a lie anyway.
        raise HTTPException(
            status_code=422,
            detail=f"No voxels in [{request.low}, {request.high}] HU to store",
        )

    # load_series fetches these and throws them away. Refetching is a wasted
    # round trip; it is not worth complicating that function's return for
    # until something measures it.
    sources = client().retrieve_series(
        study_instance_uid=request.study_uid,
        series_instance_uid=request.series_uid,
    )

    segmentation = build_segmentation(sources, mask, volume.sop_uids)
    result["seg_series_uid"] = store_segmentation(client(), segmentation)

    return result


@app.get("/health")
def health():
    return {"status": "ok", "dicomweb": DICOMWEB_URL}
