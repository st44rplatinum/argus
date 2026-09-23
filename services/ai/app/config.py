import os

ARGUS_DEVICE = os.getenv("ARGUS_DEVICE", "cpu")
ARGUS_THREADS = int(os.getenv("ARGUS_THREADS", "4"))
ARGUS_CONCURRENCY = int(os.getenv("ARGUS_CONCURRENCY", "1"))

DB_PATH = os.getenv(
    "ARGUS_DB_PATH",
    os.path.join(os.path.dirname(__file__), "jobs.sqlite3")
)

DICOMWEB_URL = os.getenv("ARGUS_DICOMWEB", "http://localhost:8042/dicom-web")

MODEL_NAME = os.getenv("ARGUS_MODEL_NAME", "threshold")

# str, not int. getenv returns a str when the variable is set, so an int
# default meant the cache key changed shape depending on the environment.
MODEL_VERSION = os.getenv("ARGUS_MODEL_VERSION", "1")

if ARGUS_DEVICE not in {"cpu", "gpu"}:
    raise ValueError("ARGUS_DEVICE must be cpu or gpu")

if ARGUS_CONCURRENCY != 1:
    raise ValueError("ARGUS_CONCURRENCY must be 1 for now")

if ARGUS_THREADS < 1:
    raise ValueError("ARGUS_THREADS must be at least 1")
