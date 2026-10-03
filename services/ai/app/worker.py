"""The job queue: one worker thread, one subprocess per job.

The jobs themselves live in app/tasks/, one module each. This file only
picks the module for a job, runs it in a subprocess, relays its progress
and records how it ended.
"""

import importlib
import multiprocessing
import os
import queue as queue_module
import threading
import traceback

from . import jobs
from .config import ARGUS_THREADS

POLL_SECONDS = 0.2

# request "model" -> module in app/tasks
TASKS = {
    "threshold": "segmentation",
    "totalsegmentator": "segmentation",
    "similar": "similar_cases",
    "register": "registration",
    "lesion_change": "lesion_change",
}

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


def _child(task, request, progress_queue, result_queue):
    """Runs in the subprocess: one task, then its result or its error on the queue."""
    try:
        module = importlib.import_module(f".tasks.{task}", __package__)
        result = module.run(request, progress_queue.put)
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

    model = job["request"].get("model", "threshold")
    task = TASKS.get(model)
    if task is None:
        jobs.mark_failed(job_id, f"unknown model: {model}")
        return

    process = multiprocessing.Process(
        target=_child,
        args=(task, job["request"], progress_queue, result_queue),
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
