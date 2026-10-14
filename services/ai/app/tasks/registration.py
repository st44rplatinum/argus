"""Register a prior onto the current scan and store it as a new series (P7)."""

from pathlib import Path

# Saved transforms, one per register job, named by its cache key.
REGISTRATIONS = Path(__file__).resolve().parents[4] / "data" / "derived" / "registrations"


def transform_for(key: str, fixed, moving, progress=lambda stage: None):
    """The transform from `fixed` points to `moving` points, saved under `key`.

    Registering takes about a minute and the answer doesn't change, so it is
    done once per pair of scans and read back after that.
    """
    import SimpleITK as sitk

    from ..register import register_deformable, register_rigid

    path = REGISTRATIONS / f"{key}.h5"
    if path.is_file():
        return sitk.ReadTransform(str(path))

    progress("rigid")
    rigid = register_rigid(fixed, moving)
    progress("deformable")
    transform = register_deformable(fixed, moving, rigid.transform).transform

    REGISTRATIONS.mkdir(parents=True, exist_ok=True)
    sitk.WriteTransform(transform, str(path))
    return transform


def run(request, progress):
    import time

    import SimpleITK as sitk

    from ..dicomweb import client_for, load_series
    from ..nifti import volume_to_sitk
    from ..register_dicom import store_registered_series

    started = time.perf_counter()
    client = client_for(request["dicomweb_url"])

    progress("fetching")
    current = load_series(client, request["study_uid"], request["series_uid"])
    prior = load_series(client, request["prior_study_uid"], request["prior_series_uid"])
    fixed = volume_to_sitk(current)
    moving = volume_to_sitk(prior)

    transform = transform_for(request["cache_key"], fixed, moving, progress)

    progress("storing")
    registered = sitk.Resample(moving, fixed, transform, sitk.sitkLinear, -1024.0, sitk.sitkFloat32)
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
        "seconds": round(time.perf_counter() - started, 1),
    }
