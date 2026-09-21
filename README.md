# radiology viewer

A CT/MR viewer with an AI layer, built for research. Local-first: no telemetry,
no network calls, no PHI leaving the machine.

**Research use only.** Nothing here is a medical device and nothing here is
validated for clinical decisions.

See [docs/ROADMAP.md](docs/ROADMAP.md) for the plan and the reasoning behind
the stack.

## Layout

    docker-compose.yml   Orthanc — the local PACS, bound to localhost only
    scripts/             Data in, sanity checks out
    services/ai/         FastAPI + PyTorch inference service (P3)
    viewer/              OHIF extension and mode (P1)
    research/            Experiments, notebooks, training. Not shipped.
    data/                Gitignored. Licences vary; read them.

## P0 — get DICOM flowing

Start the PACS. Docker Desktop must be running first.

```bash
docker compose up -d
```

Set up Python.

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r services/ai/requirements.txt
```

Make a test series and push it in. The phantom is a smoke-test fixture, not
data — it exists so the pipeline has something well-formed to fail against
before a real dataset is downloaded.

```bash
./.venv/Scripts/python.exe scripts/make_phantom.py
```

```bash
./.venv/Scripts/python.exe scripts/push_dicom.py data/raw/phantom
```

Confirm the PACS answers over DICOMweb — this is what OHIF and the AI service
will both talk to.

```bash
./.venv/Scripts/python.exe scripts/check_dicomweb.py
```

Orthanc's own UI is at <http://localhost:8042> (`orthanc` / `orthanc`).

### Torch

Installed separately, because the CUDA build must match the driver. This box is
a GTX 1050 Ti — 4 GB, Pascal, no tensor cores. It runs inference; it does not
run training. See the roadmap.

```bash
./.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Known gaps

- P0 only. No viewer, no models, no service yet.
- Orthanc runs with default credentials and permissive CORS. Both are fine for
  a localhost-bound research instance and must change before it is reachable
  from anywhere else.
