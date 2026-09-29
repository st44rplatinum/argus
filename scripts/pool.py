"""Choose the 20-case Similar Cases pool and push it to Orthanc.

Cases with all six organs embedded, spread across phases so the pool is not
all portal venous. Each case becomes its own patient. pool.json maps case id
to the study you can open; nothing else knows that join, so commit it.

Resumable: cases already in pool.json are skipped.

  python scripts/pool.py --dry-run
  python scripts/pool.py
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pydicom

ROOT = Path(__file__).resolve().parents[1]

EMBEDDINGS = ROOT / "data/derived/organ_embeddings_dinov2s.npz"
PHASES = ROOT / "data/derived/contrast_phase.json"
POOL = ROOT / "data/derived/pool.json"
DATASET = ROOT / "data/raw/totalsegmentator"
WORK = ROOT / "data/derived/pool_dicom"

ORGANS = ("liver", "spleen", "kidney_left", "kidney_right", "aorta", "heart")

PHASE_COUNTS = {
    "native": 2,
    "arterial_early": 3,
    "arterial_late": 5,
    "portal_venous": 10,
}


def complete_cases() -> set[str]:
    data = np.load(EMBEDDINGS, allow_pickle=False)
    found: dict[str, set[str]] = {}
    for case, organ in zip(data["cases"], data["organs"]):
        found.setdefault(str(case), set()).add(str(organ))
    return {c for c, organs in found.items() if set(ORGANS) <= organs}


def choose() -> dict[str, str]:
    phases = {c: v["phase"] for c, v in json.loads(PHASES.read_text()).items()}
    complete = complete_cases()

    selected = {}
    for phase, count in PHASE_COUNTS.items():
        # sorted so the pool is the same every run
        candidates = sorted(c for c in complete if phases.get(c) == phase)
        if len(candidates) < count:
            raise SystemExit(f"need {count} {phase}, only {len(candidates)} complete")
        for case in candidates[:count]:
            selected[case] = phase
    return selected


def run(cmd: list[str]) -> str:
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out.stdout


def push_case(case: str, phase: str, url: str) -> dict:
    out_dir = WORK / case
    if out_dir.exists():
        shutil.rmtree(out_dir)

    run([
        sys.executable, str(ROOT / "scripts/nifti_to_dicom.py"),
        str(DATASET / case / "ct.nii.gz"),
        "--out", str(out_dir),
        "--patient-id", f"TS-{case}",
        "--description", f"Similar Cases {case}",
    ])

    first = next(out_dir.glob("*.dcm"))
    ds = pydicom.dcmread(first, stop_before_pixels=True)
    entry = {
        "study_uid": str(ds.StudyInstanceUID),
        "series_uid": str(ds.SeriesInstanceUID),
        "phase": phase,
    }

    # push_dicom exits 0 even when Orthanc rejects files, so check the count
    stdout = run([sys.executable, str(ROOT / "scripts/push_dicom.py"), str(out_dir), "--url", url])
    match = re.search(r"accepted (\d+)/(\d+)", stdout)
    if not match or match.group(1) != match.group(2):
        raise SystemExit(f"{case}: push incomplete\n{stdout}")

    # Orthanc has the copy now
    shutil.rmtree(out_dir)
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8042")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = choose()
    for case, phase in selected.items():
        print(f"  {case}  {phase}")
    if args.dry_run:
        return

    pool = json.loads(POOL.read_text()) if POOL.is_file() else {}

    for case, phase in selected.items():
        if case in pool:
            continue
        pool[case] = push_case(case, phase, args.url)
        # written after every case so a crash loses at most one
        POOL.write_text(json.dumps(pool, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"pushed {case} ({len(pool)}/{len(selected)})", flush=True)

    if WORK.exists() and not any(WORK.iterdir()):
        WORK.rmdir()
    print(f"wrote {POOL.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
