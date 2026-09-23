import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

from .config import DB_PATH

COLUMNS = (
    "id",
    "status",
    "request",
    "result",
    "error",
    "stage",
    "created_at",
    "started_at",
    "finished_at",
)


def connect():
    # check_same_thread=False: the worker thread writes, the request threads
    # read. Every call opens and closes its own connection, so they are never
    # shared, but sqlite refuses across threads without this.
    db = sqlite3.connect(DB_PATH, check_same_thread=False)
    db.row_factory = sqlite3.Row
    return db


def init_db():
    # closing() as well as the connection's own context manager: sqlite3's
    # `with` commits the transaction, it does not close the connection. Without
    # this every call leaks a file handle.
    with closing(connect()) as db, db:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                request TEXT NOT NULL,
                result TEXT,
                error TEXT,
                stage TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT
            )
            """
        )


def _row_to_job(row):
    return {
        "id": row["id"],
        "status": row["status"],
        "request": json.loads(row["request"]),
        "result": json.loads(row["result"]) if row["result"] else None,
        "error": row["error"],
        "stage": row["stage"],
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
    }


def save_job(job):
    with closing(connect()) as db, db:
        db.execute(
            f"""
            INSERT OR REPLACE INTO jobs ({", ".join(COLUMNS)})
            VALUES ({", ".join("?" * len(COLUMNS))})
            """,
            (
                job["id"],
                job["status"],
                json.dumps(job["request"]),
                json.dumps(job["result"]) if job["result"] is not None else None,
                job["error"],
                job["stage"],
                job["created_at"],
                job["started_at"],
                job["finished_at"],
            ),
        )


def load_job(job_id):
    with closing(connect()) as db:
        row = db.execute(
            "SELECT * FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()

        return _row_to_job(row) if row is not None else None


def load_all_jobs():
    with closing(connect()) as db:
        rows = db.execute("SELECT * FROM jobs ORDER BY created_at").fetchall()
        return [_row_to_job(row) for row in rows]


def find_cached_result(cache_key):
    """The newest finished job whose request carries this cache key."""
    with closing(connect()) as db:
        row = db.execute(
            """
            SELECT * FROM jobs
            WHERE status = 'done'
              AND json_extract(request, '$.cache_key') = ?
            ORDER BY finished_at DESC
            LIMIT 1
            """,
            (cache_key,),
        ).fetchone()

        return _row_to_job(row) if row is not None else None


def mark_interrupted_jobs_failed():
    """Anything still 'running' at startup died with the last process.

    Marked failed rather than requeued: a job that silently restarts itself
    after a crash is how you end up with duplicate SEGs and no idea why.
    """
    now = datetime.now(timezone.utc).isoformat()

    with closing(connect()) as db, db:
        db.execute(
            """
            UPDATE jobs
            SET status = 'failed',
                error = ?,
                finished_at = ?
            WHERE status IN ('running', 'queued')
            """,
            ("Service restarted while job was pending", now),
        )
