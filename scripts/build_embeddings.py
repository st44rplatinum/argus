"""DINOv2 organ embeddings for the cases on disk.

Resumable: re-running skips rows already in the file.

  python scripts/build_embeddings.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services" / "ai"))

from app.config import ARGUS_THREADS  # noqa: E402
from app.encoder import ENCODER_ID, PREPROCESS_VERSION, organ_embedding  # noqa: E402

SUBSET = Path("data/raw/totalsegmentator")
OUT = Path("data/derived/organ_embeddings_dinov2s.npz")


def load_existing(path: Path) -> dict:
    if not path.is_file():
        return {}
    data = np.load(path, allow_pickle=False)
    if str(data["encoder_id"]) != ENCODER_ID or int(data["preprocess_version"]) != PREPROCESS_VERSION:
        print("embeddings were built differently, starting again")
        return {}
    return {(str(c), str(o)): v for c, o, v in zip(data["cases"], data["organs"], data["embeddings"])}


def save(path: Path, rows: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(rows)
    np.savez_compressed(
        path,
        encoder_id=ENCODER_ID,
        preprocess_version=PREPROCESS_VERSION,
        cases=np.array([k[0] for k in keys]),
        organs=np.array([k[1] for k in keys]),
        embeddings=np.stack([rows[k] for k in keys]).astype(np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, default=SUBSET)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    manifest = json.loads((args.subset / "manifest.json").read_text())
    cases, organs = manifest["cases"], manifest["structures"]

    rows = load_existing(args.out)
    todo = [c for c in cases if not any((c, o) in rows for o in organs)]
    print(f"{len(cases)} cases, {len(todo)} to do, {len(rows)} rows already")

    started = time.time()
    for done, case in enumerate(todo, start=1):
        folder = args.subset / case
        ct_img = nib.as_closest_canonical(nib.load(folder / "ct.nii.gz"))
        ct = np.asarray(ct_img.dataobj, dtype=np.float32)
        spacing = tuple(float(z) for z in ct_img.header.get_zooms()[:3])

        for organ in organs:
            path = folder / "segmentations" / f"{organ}.nii.gz"
            if not path.is_file():
                continue
            mask = np.asarray(nib.as_closest_canonical(nib.load(path)).dataobj) > 0
            vector = organ_embedding(ct, mask, spacing, threads=ARGUS_THREADS)
            if vector is not None:
                rows[(case, organ)] = vector

        if done % 10 == 0 or done == len(todo):
            save(args.out, rows)
            rate = done / (time.time() - started) * 60
            print(f"  {done}/{len(todo)} cases, {len(rows)} rows, {rate:.1f}/min", flush=True)

    save(args.out, rows)
    print(f"{len(rows)} embeddings in {args.out}")


if __name__ == "__main__":
    main()
