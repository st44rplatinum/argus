"""Build the retrieval baseline over every case in the archive.

Reads straight out of the zip, so the 23.58 GB never becomes 23.58 GB of
extracted files. What survives is a feature table of a few hundred KB
describing all 1228 cases, which is the whole point: once this exists, the
archive can go.

Resumable. Re-running skips cases already in the index, so a crash at case
900 costs minutes.

  python scripts/build_feature_index.py --organs liver spleen kidney_left
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services" / "ai"))

from app.features import FEATURE_VERSION, LEN, NAMES, organ_features  # noqa: E402

ZIP = Path("data/raw/Totalsegmentator_dataset_v201.zip")
INDEX = Path("data/derived/organ_features.npz")

# Same discipline as the similarity plugin's EOVSIM header: a format version
# separate from the feature version, so a layout change and a feature change
# invalidate independently.
FORMAT_VERSION = 1


def read_nifti(archive: zipfile.ZipFile, member) -> nib.Nifti1Image:
    """Load a .nii.gz member without writing it to disk."""
    with archive.open(member) as handle:
        raw = gzip.decompress(handle.read())
    return nib.Nifti1Image.from_bytes(raw)


def index_cases(archive: zipfile.ZipFile) -> dict[str, dict]:
    cases: dict[str, dict] = defaultdict(lambda: {"ct": None, "structures": {}})

    for info in archive.infolist():
        if info.is_dir():
            continue
        parts = Path(info.filename).parts
        if len(parts) < 2:
            continue

        case_id = parts[0]
        if parts[-1] == "ct.nii.gz":
            cases[case_id]["ct"] = info
        elif "segmentations" in parts and parts[-1].endswith(".nii.gz"):
            cases[case_id]["structures"][parts[-1][: -len(".nii.gz")]] = info

    return {k: v for k, v in cases.items() if v["ct"] is not None}


def load_existing(path: Path) -> dict[tuple[str, str], np.ndarray]:
    """Rows already computed, if the index is still valid for this code."""
    if not path.is_file():
        return {}

    data = np.load(path, allow_pickle=False)

    if (
        int(data["format_version"]) != FORMAT_VERSION
        or int(data["feature_version"]) != FEATURE_VERSION
    ):
        print("index was built by different code, starting again")
        return {}

    return {
        (str(case), str(organ)): row
        for case, organ, row in zip(data["cases"], data["organs"], data["features"])
    }


def save(path: Path, rows: dict[tuple[str, str], np.ndarray], organs: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(rows)

    np.savez_compressed(
        path,
        format_version=FORMAT_VERSION,
        feature_version=FEATURE_VERSION,
        feature_names=np.array(NAMES),
        organs_requested=np.array(organs),
        cases=np.array([k[0] for k in keys]),
        organs=np.array([k[1] for k in keys]),
        features=np.stack([rows[k] for k in keys]) if keys else np.zeros((0, LEN)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=ZIP)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument(
        "--organs",
        nargs="*",
        default=["liver", "spleen", "kidney_left", "kidney_right", "aorta", "heart"],
    )
    parser.add_argument("--limit", type=int, default=0, help="0 means every case")
    parser.add_argument("--save-every", type=int, default=25)
    args = parser.parse_args()

    if not args.zip.is_file():
        raise SystemExit(f"no archive at {args.zip}")

    rows = load_existing(args.index)
    print(f"{len(rows)} rows already in the index")

    started = time.time()

    with zipfile.ZipFile(args.zip) as archive:
        cases = index_cases(archive)
        case_ids = sorted(cases)
        if args.limit:
            case_ids = case_ids[: args.limit]

        todo = [c for c in case_ids if not any((c, o) in rows for o in args.organs)]
        print(f"{len(case_ids)} cases, {len(todo)} to do")

        for done, case_id in enumerate(todo, start=1):
            case = cases[case_id]

            try:
                ct = read_nifti(archive, case["ct"])
                # nibabel gives [x, y, z]; spacing follows the same order.
                array = np.asarray(ct.dataobj, dtype=np.float32)
                spacing = tuple(float(z) for z in ct.header.get_zooms()[:3])

                for organ in args.organs:
                    member = case["structures"].get(organ)
                    if member is None:
                        continue

                    mask = np.asarray(read_nifti(archive, member).dataobj) > 0
                    features = organ_features(array, mask, spacing)

                    if features is not None:
                        rows[(case_id, organ)] = features

            except Exception as exc:
                print(f"  {case_id}: {type(exc).__name__}: {exc}")

            if done % args.save_every == 0 or done == len(todo):
                save(args.index, rows, args.organs)
                rate = done / (time.time() - started)
                left = (len(todo) - done) / rate / 60 if rate else 0
                print(
                    f"  {done}/{len(todo)} cases, {len(rows)} rows, "
                    f"{rate * 60:.1f}/min, ~{left:.0f} min left",
                    flush=True,
                )

    save(args.index, rows, args.organs)
    size = args.index.stat().st_size
    print()
    print(f"{len(rows)} rows describing {len({k[0] for k in rows})} cases")
    print(f"index {args.index} is {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
