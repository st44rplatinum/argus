from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .dicomweb import load_series
from .segment import (
    calculate_bounding_box,
    threshold_segment,
)

app = FastAPI(title="Argus")

class SegmentRequest(BaseModel):
    study_uid: str 
    series_uid: str 

    low: float = Field(default=300)
    high: float = Field(default=3000)

@app.post("/segment")
def segment(request: SegmentRequest):
    try:
        volume = load_series(
            request.study_uid,
            request.series_uid,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Could not load series: {exec}",
        ) from exec

    mask = threshold_segment(
        volume,
        request.low,
        request.high,
    )

    bbox = calculate_bounding_box(mask)

    return {
        "study_uid": request.study_uid,
        "series_uid": request.series_uid,
        "voxel_count": int(mask.sum()),
        "bounding_box": bbox,
    }

