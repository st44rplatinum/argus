import multiprocessing
import os
import queue as queue_module
import threading
import traceback

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

    target = run_similar if job["request"].get("model") == "similar" else run_segmentation
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
