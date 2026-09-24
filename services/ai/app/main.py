"""The AI service.

A synchronous /segment for checking the service is sane, and a job queue for
work that takes minutes — which is everything once a real model is involved.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from dicomweb_client.api import DICOMwebClient
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import db, jobs, worker
from .cache import make_cache_key
from .config import (
    ARGUS_DEVICE,
    ARGUS_THREADS,
    DICOMWEB_URL,
    MODEL_NAME,
    MODEL_VERSION,
)
from .dicom_seg import build_segmentation, store_segmentation
from .dicomweb import client_for, load_series
from .segment import calculate_bounding_box, threshold_segment


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    # Anything left running or queued died with the last process.
    db.mark_interrupted_jobs_failed()
    worker.start()
    yield


app = FastAPI(title="Argus", lifespan=lifespan)

# Named origins, never a wildcard. OHIF's dev server proxies DICOMweb for us,
# but its proxy config is a single env-driven rule that replaces the list
# rather than appending, so the panel cannot reach this service the same way.
#
# This is not the mistake the Orthanc config made: that was "*" on an
# unauthenticated PACS, which let any page the user visited read studies. An
# allowlist of one localhost origin does not.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

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

    # "threshold" is a second and needs nothing; "totalsegmentator" is minutes
    # on CPU and downloads weights on first use.
    model: str = "threshold"
    fast: bool = True


@app.get("/health")
def health():
    return {
        "status": "ok",
        "dicomweb": DICOMWEB_URL,
        "device": ARGUS_DEVICE,
        "threads": ARGUS_THREADS,
        "model": f"{MODEL_NAME}@{MODEL_VERSION}",
    }


@app.post("/segment")
def segment(request: SegmentRequest):
    """Synchronous. Fine for a threshold, useless for anything slower."""
    try:
        volume = load_series(client(), request.study_uid, request.series_uid)
    except ValueError as exc:
        # A series that exists but is not a coherent volume. Not "not found".
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
        raise HTTPException(
            status_code=422,
            detail=f"No voxels in [{request.low}, {request.high}] HU to store",
        )

    sources = client().retrieve_series(
        study_instance_uid=request.study_uid,
        series_instance_uid=request.series_uid,
    )
    segmentation = build_segmentation(sources, mask, volume.sop_uids)
    result["seg_series_uid"] = store_segmentation(client(), segmentation)

    return result


@app.post("/jobs")
def create_job(request: SegmentRequest):
    # The parameters belong in the key. Threshold 300-3000 and 400-3000 are
    # different answers for the same series and the same model.
    parameters = {"store": request.store}
    if request.model == "totalsegmentator":
        parameters["fast"] = request.fast
    else:
        parameters.update({"low": request.low, "high": request.high})

    cache_key = make_cache_key(
        request.series_uid,
        request.model,
        MODEL_VERSION,
        parameters,
    )

    cached = db.find_cached_result(cache_key)
    if cached is not None:
        return {**cached, "cached": True}

    payload = request.model_dump()
    payload["cache_key"] = cache_key
    payload["dicomweb_url"] = DICOMWEB_URL

    return {**jobs.submit_job(payload), "cached": False}


@app.get("/jobs")
def list_jobs():
    return jobs.list_jobs()


@app.get("/jobs/{job_id}")
def read_job(job_id: str):
    job = jobs.get_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="No such job")

    return job


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = jobs.cancel_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="No such job")

    return job
