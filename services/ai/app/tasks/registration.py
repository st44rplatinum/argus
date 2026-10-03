"""Register a prior onto the current scan and store it as a new series (P7)."""

from pathlib import Path

# Saved transforms, one per register job, named by its cache key.
REGISTRATIONS = Path(__file__).resolve().parents[4] / "data" / "derived" / "registrations"


def run(request, progress):
    import time

    import SimpleITK as sitk

    from ..dicomweb import client_for, load_series
    from ..nifti import volume_to_sitk
    from ..register import register_deformable, register_rigid
    from ..register_dicom import store_registered_series

    started = time.perf_counter()
    client = client_for(request["dicomweb_url"])

    progress("fetching")
    current = load_series(client, request["study_uid"], request["series_uid"])
    prior = load_series(client, request["prior_study_uid"], request["prior_series_uid"])
    fixed = volume_to_sitk(current)
    moving = volume_to_sitk(prior)

    progress("rigid")
    rigid = register_rigid(fixed, moving)

    progress("deformable")
    result = register_deformable(fixed, moving, rigid.transform)

    REGISTRATIONS.mkdir(parents=True, exist_ok=True)
    sitk.WriteTransform(result.transform, str(REGISTRATIONS / f"{request['cache_key']}.h5"))

    progress("storing")
    registered = sitk.Resample(moving, fixed, result.transform, sitk.sitkLinear, -1024.0, sitk.sitkFloat32)
    sources = client.retrieve_series(
        study_instance_uid=request["study_uid"],
        series_instance_uid=request["series_uid"],
    )
    date = request.get("prior_study_date") or "unknown date"
    series_uid = store_registered_series(
        client, sources, current.sop_uids, registered, f"Prior {date}, registered"
    )

    return {
        "registered_series_uid": series_uid,
        "rigid_parameters": list(rigid.transform.GetParameters()),
        "seconds": round(time.perf_counter() - started, 1),
    }
