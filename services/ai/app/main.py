"""The AI service.

A synchronous /segment for checking the service is sane, and a job queue for
work that takes minutes — which is everything once a real model is involved.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from dicomweb_client.api import DICOMwebClient
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import db, jobs, worker
from .cache import code_version, make_cache_key
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
    worker.stop_all()


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

    low: float = 300
    high: float = 3000

    store: bool = False

    model: str = "threshold"
    fast: bool = True

    organ: str | None = None
    k: int = 5
    phase_centred: bool = False
    exclude_case: str | None = None

    # register / lesion_change: study_uid and series_uid are the current scan
    prior_study_uid: str | None = None
    prior_series_uid: str | None = None
    prior_study_date: str | None = None

    # lesion change
    point: list[float] | None = None

    # track: one point per lesion, and every earlier scan to follow them through
    points: list[list[float]] | None = None
    priors: list[dict] | None = None
    study_date: str | None = None

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


# a change to the model or to any service code starts the cache afresh
CACHE_VERSION = f"{MODEL_VERSION}+{code_version()}"


def registration_key(series_uid: str, prior_series_uid: str) -> str:
    """Cache key of a register job, also where its transform is saved."""
    return make_cache_key(
        series_uid,
        "register",
        CACHE_VERSION,
        {"prior_series_uid": prior_series_uid},
    )


@app.post("/jobs")
def create_job(request: SegmentRequest):
    # The parameters belong in the key. Threshold 300-3000 and 400-3000 are
    # different answers for the same series and the same model.
    parameters = {"store": request.store}
    if request.model == "similar":
        from .similar import load

        if request.organ is None:
            raise HTTPException(422, "similar needs an organ")
        parameters = {
            "organ": request.organ,
            "k": request.k,
            "phase_centred": request.phase_centred,
            "exclude_case": request.exclude_case,
            "index": load().version,
        }
    elif request.model in ("register", "lesion_change"):
        if not (request.prior_study_uid and request.prior_series_uid):
            raise HTTPException(422, f"{request.model} needs prior_study_uid and prior_series_uid")
        parameters = {"prior_series_uid": request.prior_series_uid}
        if request.model == "lesion_change":
            if request.point is None or len(request.point) != 3:
                raise HTTPException(422, "lesion_change needs a 3D point")
            parameters["point"] = [round(v, 2) for v in request.point]
    elif request.model == "track":
        if not request.points or not request.priors:
            raise HTTPException(422, "track needs points and priors")
        if any(len(p) != 3 for p in request.points):
            raise HTTPException(422, "each point is x, y, z")
        if any(not {"study_uid", "series_uid", "date"} <= set(p) for p in request.priors):
            raise HTTPException(422, "each prior needs study_uid, series_uid and date")
        parameters = {
            "priors": sorted(p["series_uid"] for p in request.priors),
            "points": [[round(v, 2) for v in p] for p in request.points],
        }
    elif request.model == "totalsegmentator":
        parameters["fast"] = request.fast
    else:
        parameters.update({"low": request.low, "high": request.high})

    cache_key = make_cache_key(
        request.series_uid,
        request.model,
        CACHE_VERSION,
        parameters,
    )

    cached = db.find_cached_result(cache_key)
    if cached is not None:
        return {**cached, "cached": True}

    payload = request.model_dump()
    payload["cache_key"] = cache_key
    payload["dicomweb_url"] = DICOMWEB_URL
    if request.model == "track":
        # each pair of scans is registered once and shared with the other jobs
        for prior in payload["priors"]:
            prior["registration_key"] = registration_key(request.series_uid, prior["series_uid"])
    if request.model == "lesion_change":
        payload["registration_key"] = registration_key(
            request.series_uid, request.prior_series_uid
        )

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
