"""P8 steps 0 and 1: split the portal venous livers, and plant test lesions.

Drops livers the scan cuts off, splits the rest into train / val / test by a
fixed seed, and writes the lesion recipes for val and test. Only recipes are
saved; app.lesions.planted() rebuilds any case from them.

  python scripts/anomaly_split.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services" / "ai"))

from app.lesions import DATASET, make_recipes, write_recipes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data/derived"
SEED = 7
COUNTS = {"train": 21, "val": 5}  # test gets the rest


def liver_check(case: str) -> tuple[float, str | None]:
    """Liver volume in ml, and why it can't be used, if it can't."""
    mask = nib.load(DATASET / case / "segmentations" / "liver.nii.gz")
    array = np.asarray(mask.dataobj) > 0
    ml = float(array.sum() * np.prod(mask.header.get_zooms()[:3]) / 1000)

    # cut off by the scan: liver on the first or last slice
    if array[:, :, 0].any() or array[:, :, -1].any():
        return ml, "liver touches the top or bottom slice"
    return ml, None


def main() -> None:
    phases = json.loads((DERIVED / "contrast_phase.json").read_text())
    pool = set(json.loads((DERIVED / "pool.json").read_text()))
    candidates = sorted(c for c, v in phases.items() if v["phase"] == "portal_venous")

    volumes, excluded = {}, []
    for case in candidates:
        ml, reason = liver_check(case)
        if reason:
            excluded.append({"case": case, "reason": reason})
        else:
            volumes[case] = ml

    # much smaller than the rest is partial, or barely there
    median = float(np.median(list(volumes.values())))
    for case, ml in list(volumes.items()):
        if ml < 0.5 * median:
            excluded.append({"case": case, "reason": f"liver {ml:.0f} ml, under half the median"})
            del volumes[case]

    # P5 pool cases go to train where possible, so test stays separate from them
    rng = np.random.default_rng(SEED)
    usable = sorted(volumes)
    others = [c for c in usable if c not in pool]
    rng.shuffle(others)
    in_pool = [c for c in usable if c in pool]

    test_count = len(usable) - COUNTS["train"] - COUNTS["val"]
    test = sorted(others[:test_count])
    val = sorted(others[test_count : test_count + COUNTS["val"]])
    train = sorted(set(usable) - set(test) - set(val))

    split = {
        "seed": SEED,
        "phase": "portal_venous",
        "excluded": excluded,
        "train": train,
        "val": val,
        "test": test,
        "counts": {"candidates": len(candidates), "excluded": len(excluded),
                   "train": len(train), "val": len(val), "test": len(test),
                   "pool_in_test": len(set(test) & pool), "pool_cases_usable": len(in_pool)},
    }
    (DERIVED / "anomaly_split.json").write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")

    recipes = []
    for i, case in enumerate(val + test):
        recipes += make_recipes(case, seed=SEED * 1000 + i)
    write_recipes(recipes, DERIVED / "anomaly_lesions.json")

    print(json.dumps(split["counts"]))
    print(f"excluded: {[e['case'] for e in excluded]}")
    print(f"{len(recipes)} lesions over {len(val) + len(test)} val/test cases, "
          f"{sum(1 for c in val + test if not any(r.case == c for r in recipes))} with none")


if __name__ == "__main__":
    main()
