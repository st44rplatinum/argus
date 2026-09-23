import queue
import threading
import uuid
from datetime import datetime, timezone

from . import db

# Plain queue and lock rather than asyncio's. The worker supervises a
# subprocess, which is a blocking wait, and mixing that with an event loop on
# Windows buys nothing. FastAPI runs sync handlers in a threadpool anyway.
jobs = {}
job_queue = queue.Queue()
running_processes = {}
jobs_lock = threading.Lock()

TERMINAL = {"done", "failed", "cancelled"}


def now():
    return datetime.now(timezone.utc).isoformat()


def create_job(request):
    job = {
        "id": str(uuid.uuid4()),
        "status": "queued",
        "request": request,
        "result": None,
        "error": None,
        "stage": None,
        "created_at": now(),
        "started_at": None,
        "finished_at": None,
    }

    with jobs_lock:
        jobs[job["id"]] = job
        db.save_job(job)

    return job


def get_job(job_id):
    """Memory first, then the database — a job outlives the process."""
    with jobs_lock:
        job = jobs.get(job_id)

        if job is None:
            job = db.load_job(job_id)
            if job is not None:
                jobs[job_id] = job

        return dict(job) if job is not None else None


def list_jobs():
    return db.load_all_jobs()


def update_job(job_id, **changes):
    """The function four callers assumed existed."""
    with jobs_lock:
        job = jobs.get(job_id) or db.load_job(job_id)

        if job is None:
            return None

        job.update(changes)
        jobs[job_id] = job
        db.save_job(job)

        return dict(job)


def set_stage(job_id, stage):
    return update_job(job_id, stage=stage)


def mark_running(job_id):
    return update_job(job_id, status="running", started_at=now(), stage="starting")


def mark_done(job_id, result):
    running_processes.pop(job_id, None)

    return update_job(
        job_id,
        status="done",
        result=result,
        error=None,
        finished_at=now(),
        stage="done",
    )


def mark_failed(job_id, error):
    running_processes.pop(job_id, None)

    return update_job(
        job_id,
        status="failed",
        error=str(error),
        finished_at=now(),
    )


def mark_cancelled(job_id):
    running_processes.pop(job_id, None)

    return update_job(
        job_id,
        status="cancelled",
        finished_at=now(),
    )


def submit_job(request):
    job = create_job(request)
    job_queue.put(job["id"])
    return job


def cancel_job(job_id):
    job = get_job(job_id)

    if job is None:
        return None

    if job["status"] in TERMINAL:
        return job

    # Queued jobs stay in the queue; the worker skips anything already
    # cancelled when it pops. Removing from a Queue mid-flight is not worth
    # the machinery.
    if job["status"] == "running":
        process = running_processes.get(job_id)
        if process is not None and process.is_alive():
            process.terminate()

    return mark_cancelled(job_id)
