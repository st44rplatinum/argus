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

Orthanc's own UI is at <http://localhost:8042>.

It runs with authentication off and CORS off, bound to `127.0.0.1` only. That
bind is the whole security property — the browser never reaches Orthanc
directly, because the viewer proxies to it. If you ever unbind it from
localhost, put authentication back in the same change.

## P1 — the viewer

OHIF is a build dependency, pinned to a release tag and gitignored. Our code is
`viewer/config` and (later) our extension and mode — the same split as eov and
eov-similarity-plugin.

**The checkout lives on the SSD, not next to the repo.** `E:` is a 2009 SATA II
spinning disk; installing there took twenty minutes in pnpm's linking phase
alone, and the first bundle another ten. `viewer/Viewers` is a junction to
`C:\radviewer\Viewers` so the ~200k small files land on flash. The repo, and
`data/`, stay on `E:` where sequential reads of large volumes are fine.

```bash
New-Item -ItemType Junction -Path "viewer\Viewers" -Target "C:\radviewer\Viewers"
```

Junctions need no elevation, unlike symlinks.

```bash
git clone --depth 1 --branch v3.13.10 https://github.com/OHIF/Viewers.git viewer/Viewers
```

Needs Node >= 24 and pnpm >= 11. Corepack can run the pinned pnpm without a
global install — `corepack enable` wants write access to the Node install
directory and is not necessary.

```bash
cd viewer/Viewers && corepack pnpm@11.5.2 install
```

Copy our config into the checkout, then start it. `link_viewer.py` also writes
an `.npmrc` pinning pnpm's store next to the checkout — across volumes pnpm
cannot hardlink and copies every file of every package instead, which is what
made the first install so slow.

```bash
./.venv/Scripts/python.exe scripts/link_viewer.py
```

```bash
./.venv/Scripts/python.exe scripts/dev_viewer.py
```

Viewer at <http://localhost:3000>.

### Torch

Installed separately, and **not the same build on both machines**. The local
card is a GTX 1050 Ti (Pascal, sm_61); the burst box is an RTX 5090 (Blackwell,
sm_120, needs PyTorch >= 2.7 on CUDA 12.8). These sit at opposite ends of
PyTorch's build matrix, so one wheel may not cover both.

Ask the machine rather than guessing — if its compute capability is absent from
the arch list, CUDA calls fail with `no kernel image is available`:

```bash
./.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"
```

Local only needs torch for experiments; shipped inference goes through ONNX
Runtime, which is indifferent to all of the above. Training belongs on the
burst box. See [docs/ROADMAP.md](docs/ROADMAP.md).

## Known gaps

- P0 only. No viewer, no models, no service yet.
- Orthanc runs with default credentials and permissive CORS. Both are fine for
  a localhost-bound research instance and must change before it is reachable
  from anywhere else.
