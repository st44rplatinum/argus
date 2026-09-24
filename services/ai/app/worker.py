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

        from .dicom_seg import (
            build_multi_segmentation,
            build_segmentation,
            store_segmentation,
        )
        from .dicomweb import client_for, load_series
        from .segment import calculate_bounding_box, threshold_segment

        client = client_for(request["dicomweb_url"])
        model = request.get("model", "threshold")

        progress_queue.put("fetching")
        volume = load_series(client, request["study_uid"], request["series_uid"])

        progress_queue.put("inferring")

        if model == "totalsegmentator":
            from .totalseg import run, structure_names

            labelmap = run(volume, fast=request.get("fast", True))
            names = structure_names()
            present = [int(v) for v in np.unique(labelmap) if v != 0]

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
                "structures": [names.get(v, str(v)) for v in present],
                "seg_series_uid": None,
            }
            build = lambda sources: build_multi_segmentation(  # noqa: E731
                sources, labelmap, volume.sop_uids, names
            )
        else:
            mask = threshold_segment(volume, request["low"], request["high"])
            box = calculate_bounding_box(mask)

            if box is None:
                result_queue.put({
                    "status": "failed",
                    "error": f"No voxels in [{request['low']}, {request['high']}] HU",
                })
                return

            result = {
                "shape": list(volume.shape),
                "spacing_mm": list(volume.spacing),
                "voxel_count": int(mask.sum()),
                "bounding_box": box,
                "seg_series_uid": None,
            }
            build = lambda sources: build_segmentation(  # noqa: E731
                sources, mask, volume.sop_uids
            )

        if request.get("store"):
            progress_queue.put("encoding")
            sources = client.retrieve_series(
                study_instance_uid=request["study_uid"],
                series_instance_uid=request["series_uid"],
            )
            segmentation = build(sources)

            progress_queue.put("storing")
            result["seg_series_uid"] = store_segmentation(client, segmentation)

        result_queue.put({"status": "done", "result": result})

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

    process = multiprocessing.Process(
        target=run_segmentation,
        args=(job["request"], progress_queue, result_queue),
        daemon=True,
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
