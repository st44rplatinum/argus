import multiprocessing
import os
import queue as queue_module
import threading
import traceback
from pathlib import Path

from . import jobs
from .config import ARGUS_THREADS

POLL_SECONDS = 0.2

_worker_thread = None


def _limit_threads():
    """Cap the maths libraries before anything imports them.

    Six cores and no hyperthreading: let a job take all six and everything
    else on the machine stutters. Set in the parent so spawned children
    inherit it — numpy and friends read these at import time, which in a
    spawned child is too early to set from inside the job.
    """
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(name, str(ARGUS_THREADS))


def run_segmentation(request, progress_queue, result_queue):
    """Runs in a subprocess. Everything it touches is imported here.

    Storing is the last action deliberately: terminating a cancelled job
    partway cannot then leave half a SEG in the PACS.
    """
    try:
        import numpy as np

        from .dicomweb import client_for, load_series

        client = client_for(request["dicomweb_url"])

        progress_queue.put("fetching")

        volume = load_series(
            client,
            request["study_uid"],
            request["series_uid"],
        )

        from .dicom_seg import (
            build_multi_segmentation,
            build_segmentation,
            store_segmentation,
        )
        from .segment import calculate_bounding_box, threshold_segment

        progress_queue.put("inferring")

        if request.get("model", "threshold") == "totalsegmentator":
            from .totalseg import run, structure_names

            labelmap = run(
                volume,
                fast=request.get("fast", True),
            )

            names = structure_names()
            present = [
                int(v)
                for v in np.unique(labelmap)
                if v != 0
            ]

            if not present:
                result_queue.put({
                    "status": "failed",
                    "error": "TotalSegmentator found no structures",
                })
                return

            result = {
                "shape": list(volume.shape),
                "spacing_mm": list(volume.spacing),
                "voxel_count": int((labelmap != 0).sum()),
                "structures": [
                    names.get(v, str(v))
                    for v in present
                ],
                "seg_series_uid": None,
            }

            build = lambda sources: build_multi_segmentation(
                sources,
                labelmap,
                volume.sop_uids,
                names,
            )

        else:
            mask = threshold_segment(
                volume,
                request["low"],
                request["high"],
            )

            box = calculate_bounding_box(mask)

            if box is None:
                result_queue.put({
                    "status": "failed",
                    "error": (
                        f"No voxels in "
                        f"[{request['low']}, {request['high']}] HU"
                    ),
                })
                return

            result = {
                "shape": list(volume.shape),
                "spacing_mm": list(volume.spacing),
                "voxel_count": int(mask.sum()),
                "bounding_box": box,
                "seg_series_uid": None,
            }

            build = lambda sources: build_segmentation(
                sources,
                mask,
                volume.sop_uids,
            )

        if request.get("store"):
            progress_queue.put("encoding")

            sources = client.retrieve_series(
                study_instance_uid=request["study_uid"],
                series_instance_uid=request["series_uid"],
            )

            segmentation = build(sources)

            progress_queue.put("storing")

            result["seg_series_uid"] = store_segmentation(
                client,
                segmentation,
            )

        result_queue.put({
            "status": "done",
            "result": result,
        })

    except Exception as exc:
        result_queue.put({
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })


def _contrast_phase(ct_nii, stats):
    """Phase from the classifier, reusing the statistics already computed."""
    from totalsegmentator.bin.totalseg_get_phase import get_ct_contrast_phase

    final = None
    for item in get_ct_contrast_phase(ct_nii, device="cpu", quiet=True, existing_stats=stats):
        final = item
    return final["result"]["phase"]


def run_similar(request, progress_queue, result_queue):
    """Runs in a subprocess, like run_segmentation."""
    try:
        import tempfile
        from pathlib import Path

        import nibabel as nib
        import numpy as np
        import SimpleITK as sitk

        from .describe import describe
        from .dicomweb import client_for, load_series
        from .nifti import sitk_like, volume_to_sitk
        from .similar import case_for_study, search
        from .totalseg import run, structure_names

        organ = request["organ"]
        client = client_for(request["dicomweb_url"])

        progress_queue.put("fetching")
        volume = load_series(client, request["study_uid"], request["series_uid"])

        # Same arguments the phase classifier uses, so its stats can be reused.
        progress_queue.put("segmenting")
        labelmap, stats = run(
            volume,
            fast=True,
            statistics=True,
            stats_aggregation="median",
            statistics_exclude_masks_at_border=False,
        )

        values = [v for v, name in structure_names().items() if name == organ]
        if not values:
            raise ValueError(f"unknown organ: {organ}")
        mask = (labelmap == values[0]).astype(np.uint8)
        if not mask.any():
            raise ValueError(f"TotalSegmentator found no {organ} in this series")

        # Through a .nii.gz and nib.load, the same loader the index used.
        with tempfile.TemporaryDirectory(prefix="argus-similar-") as workdir:
            ct_path = Path(workdir) / "ct.nii.gz"
            mask_path = Path(workdir) / "mask.nii.gz"
            ct_image = volume_to_sitk(volume)
            sitk.WriteImage(ct_image, str(ct_path))
            sitk.WriteImage(sitk_like(ct_image, mask), str(mask_path))
            ct_nii = nib.load(ct_path)
            mask_nii = nib.load(mask_path)

            progress_queue.put("classifying")
            phase = _contrast_phase(ct_nii, stats)

            progress_queue.put("describing")
            baseline, embedding = describe(ct_nii, mask_nii)

        if baseline is None or embedding is None:
            raise ValueError(f"{organ} too small to describe")

        progress_queue.put("searching")
        results = search(
            baseline,
            embedding,
            organ=organ,
            phase=phase,
            k=request.get("k", 5),
            phase_centred=request.get("phase_centred", False),
            exclude_case=request.get("exclude_case") or case_for_study(request["study_uid"]),
        )

        result_queue.put({
            "status": "done",
            "result": {"organ": organ, "phase": phase, "results": results},
        })

    except Exception as exc:
        result_queue.put({
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })


def _drain(progress_queue, job_id):
    """Move any pending stage updates onto the job."""
    while True:
        try:
            jobs.set_stage(job_id, progress_queue.get_nowait())
        except queue_module.Empty:
            return


def _run_one(job_id):
    job = jobs.get_job(job_id)

    # Cancelled while it sat in the queue.
    if job is None or job["status"] != "queued":
        return

    jobs.mark_running(job_id)

    progress_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()

    model = job["request"].get("model")
    if model == "similar":
        target = run_similar
    elif model == "register":
        target = run_register
    elif model == "lesion_change":
        target = run_lesion_change
    else:
        target = run_segmentation

    process = multiprocessing.Process(
        target=target,
        args=(job["request"], progress_queue, result_queue),
        # Not daemonic: nnU-Net inside TotalSegmentator starts its own pool,
        # and daemonic processes may not have children. stop_all() does the
        # cleanup daemon=True used to.
        daemon=False,
    )
    process.start()
    jobs.running_processes[job_id] = process

    while process.is_alive():
        _drain(progress_queue, job_id)
        process.join(timeout=POLL_SECONDS)

    _drain(progress_queue, job_id)

    # Cancelled: terminate() got here first and the status is already set.
    if jobs.get_job(job_id)["status"] == "cancelled":
        return

    try:
        outcome = result_queue.get_nowait()
    except queue_module.Empty:
        jobs.mark_failed(
            job_id,
            f"worker exited with code {process.exitcode} and no result",
        )
        return

    if outcome["status"] == "done":
        jobs.mark_done(job_id, outcome["result"])
    else:
        jobs.mark_failed(job_id, outcome.get("traceback") or outcome["error"])


def _loop():
    while True:
        job_id = jobs.job_queue.get()

        if job_id is None:
            return

        try:
            _run_one(job_id)
        except Exception as exc:
            jobs.mark_failed(job_id, f"worker loop: {exc}")


def start():
    """One worker thread, supervising one subprocess at a time."""
    global _worker_thread

    if _worker_thread is not None and _worker_thread.is_alive():
        return

    _limit_threads()
    _worker_thread = threading.Thread(target=_loop, daemon=True, name="argus-worker")
    _worker_thread.start()


def stop_all():
    """Terminate running jobs, so shutting the service down does not wait on them."""
    for process in list(jobs.running_processes.values()):
        if process.is_alive():
            process.terminate()


# Saved transforms, one per register job, named by its cache key.
REGISTRATIONS = Path(__file__).resolve().parents[3] / "data" / "derived" / "registrations"


def run_register(request, progress_queue, result_queue):
    """Register the prior onto the current scan and store it as a new series."""
    try:
        import time

        import SimpleITK as sitk

        from .dicomweb import client_for, load_series
        from .nifti import volume_to_sitk
        from .register import register_deformable, register_rigid
        from .register_dicom import store_registered_series

        started = time.perf_counter()
        client = client_for(request["dicomweb_url"])

        progress_queue.put("fetching")
        current = load_series(client, request["study_uid"], request["series_uid"])
        prior = load_series(client, request["prior_study_uid"], request["prior_series_uid"])

        fixed = volume_to_sitk(current)
        moving = volume_to_sitk(prior)

        progress_queue.put("rigid")
        rigid = register_rigid(fixed, moving)

        progress_queue.put("deformable")
        result = register_deformable(fixed, moving, rigid.transform)

        REGISTRATIONS.mkdir(parents=True, exist_ok=True)
        transform_path = REGISTRATIONS / f"{request['cache_key']}.h5"
        sitk.WriteTransform(result.transform, str(transform_path))

        progress_queue.put("storing")
        registered = sitk.Resample(moving, fixed, result.transform, sitk.sitkLinear, -1024.0, sitk.sitkFloat32)
        sources = client.retrieve_series(
            study_instance_uid=request["study_uid"],
            series_instance_uid=request["series_uid"],
        )
        date = request.get("prior_study_date") or "unknown date"
        series_uid = store_registered_series(
            client, sources, current.sop_uids, registered, f"Prior {date}, registered"
        )

        result_queue.put({
            "status": "done",
            "result": {
                "registered_series_uid": series_uid,
                "rigid_parameters": list(rigid.transform.GetParameters()),
                "seconds": round(time.perf_counter() - started, 1),
            },
        })

    except Exception as exc:
        result_queue.put({
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })


def _native_lesion(image, seed_point, radius_mm=20.0):
    """Grow a lesion from a point, within a box around it so it cannot leak far."""
    import numpy as np
    import SimpleITK as sitk

    index = np.asarray(image.TransformPhysicalPointToIndex(tuple(map(float, seed_point))))
    size = np.asarray(image.GetSize())
    if np.any(index < 0) or np.any(index >= size):
        raise ValueError("point is outside this scan")

    spacing = np.asarray(image.GetSpacing())
    radius = np.ceil(radius_mm / spacing).astype(int)
    lower = np.maximum(index - radius, 0)
    upper = np.minimum(index + radius + 1, size)

    roi = sitk.RegionOfInterest(image, [int(v) for v in upper - lower], [int(v) for v in lower])
    seed = tuple(int(v) for v in index - lower)
    values = sitk.GetArrayFromImage(roi)

    # Cut halfway between the lesion (around the seed) and what surrounds it
    # (the box's median). A blurred edge then counts half in, half out,
    # however blurred, so two scans of different sharpness still agree.
    z, y, x = seed[::-1]
    inside = float(np.median(values[max(z - 1, 0) : z + 2, max(y - 1, 0) : y + 2, max(x - 1, 0) : x + 2]))
    around = float(np.median(values))
    half = (inside + around) / 2.0
    span = abs(around - inside)
    lower_hu, upper_hu = (inside - span, half) if inside < around else (half, inside + span)

    grown = sitk.ConnectedThreshold(roi, seedList=[seed], lower=lower_hu, upper=upper_hu, replaceValue=1)

    # sitk arrays are [z, y, x]
    mask = sitk.GetArrayFromImage(grown) > 0
    zs, ys, xs = np.nonzero(mask)
    volume_mm3 = float(mask.sum() * np.prod(spacing))

    diameter = 0.0
    for z in np.unique(zs):
        points = np.column_stack([xs[zs == z] * spacing[0], ys[zs == z] * spacing[1]])
        if len(points) > 1:
            gaps = points[:, None, :] - points[None, :, :]
            diameter = max(diameter, float(np.sqrt((gaps**2).sum(-1)).max()))

    return {"volume_mm3": volume_mm3, "diameter_mm": diameter}


def _change(before, after):
    return (after - before) / before * 100.0 if before > 0 else None


def run_lesion_change(request, progress_queue, result_queue):
    """Measure a lesion in both native scans, matched through the registration."""
    try:
        import SimpleITK as sitk

        from .dicomweb import client_for, load_series
        from .nifti import volume_to_sitk

        transform_path = REGISTRATIONS / f"{request['registration_key']}.h5"
        if not transform_path.is_file():
            raise ValueError("no registration for this pair yet, run Register first")
        transform = sitk.ReadTransform(str(transform_path))

        client = client_for(request["dicomweb_url"])

        progress_queue.put("fetching")
        current = volume_to_sitk(load_series(client, request["study_uid"], request["series_uid"]))
        prior = volume_to_sitk(load_series(client, request["prior_study_uid"], request["prior_series_uid"]))

        # registration maps current points to prior points
        current_point = tuple(map(float, request["point"]))
        prior_point = transform.TransformPoint(current_point)

        progress_queue.put("measuring")
        now = _native_lesion(current, current_point)
        before = _native_lesion(prior, prior_point)

        # The wrong way, kept as a control: measured in the warped prior, the
        # deformable step has already absorbed some of the growth.
        warped = sitk.Resample(prior, current, transform, sitk.sitkLinear, -1024.0, sitk.sitkFloat32)
        warped_before = _native_lesion(warped, current_point)

        result_queue.put({
            "status": "done",
            "result": {
                "point_current": list(current_point),
                "point_prior": list(prior_point),
                "current": now,
                "prior": before,
                "change_percent": _change(before["volume_mm3"], now["volume_mm3"]),
                "warped_prior": warped_before,
                "warped_change_percent": _change(warped_before["volume_mm3"], now["volume_mm3"]),
            },
        })

    except Exception as exc:
        result_queue.put({
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
