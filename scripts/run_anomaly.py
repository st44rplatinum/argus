"""P8: score methods on the planted lesions and print the table.

Thresholds are chosen on val, then test is run once with them unchanged.
Everything is scored inside the liver minus a 3 mm rim (eval_anomaly.RIM_MM).

  python scripts/run_anomaly.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services" / "ai"))

from app.anomaly import (  # noqa: E402
    connected_blobs,
    population_statistics,
    score_own_liver,
    score_population,
)
from app.eval_anomaly import scoring_region, threshold_for_one_fp, voxel_auroc  # noqa: E402
from app.lesions import planted, read_recipes  # noqa: E402

DERIVED = Path(__file__).resolve().parents[1] / "data/derived"
SMALL_MM = 15.0   # lesion diameter
SUBTLE_HU = 30.0  # |contrast|
LARGE_ML = 0.5    # a flagged cluster this big in a "normal" liver is worth a look


def load(cases, recipes):
    """Each case as (image, scoring region, lesion mask, its recipes in crop voxels)."""
    out = []
    for case in cases:
        image, liver, lesions = planted(case, recipes)
        spacing = liver.GetSpacing()[::-1]
        offset = np.rint(np.asarray(liver.GetOrigin()[::-1]) / spacing).astype(int)
        region = scoring_region(sitk.GetArrayFromImage(liver), spacing)
        mine = [(np.asarray(r.centre_zyx) - offset, r) for r in recipes if r.case == case]
        out.append((image, region, lesions, mine))
    return out


def evaluate(maps, cases, threshold):
    """Per lesion: found or not, and its size and contrast. Plus false blobs per scan."""
    found, false_blobs = [], 0
    for score, (_, region, lesions, mine) in zip(maps, cases):
        blobs = connected_blobs(score * region, threshold)
        labels, _ = ndimage.label(lesions > 0, structure=np.ones((3, 3, 3)))
        hit = {int(labels[tuple(b["peak_zyx"])]) for b in blobs} - {0}
        false_blobs += sum(1 for b in blobs if labels[tuple(b["peak_zyx"])] == 0)
        for centre, recipe in mine:
            found.append((int(labels[tuple(centre)]) in hit, recipe))
    return found, false_blobs / len(cases)


def rate(found, keep=lambda r: True):
    chosen = [f for f, r in found if keep(r)]
    return f"{np.mean(chosen):4.0%} ({sum(chosen)}/{len(chosen)})" if chosen else "–"


def main() -> None:
    split = json.loads((DERIVED / "anomaly_split.json").read_text(encoding="utf-8"))
    recipes = read_recipes(DERIVED / "anomaly_lesions.json")

    print("loading...", flush=True)
    train = load(split["train"], [])
    val = load(split["val"], recipes)
    test = load(split["test"], recipes)

    images = [c[0] for c in train]
    regions = [sitk.GetImageFromArray(c[1]) for c in train]
    for region, image in zip(regions, images):
        region.CopyInformation(image)
    mean, sd = population_statistics(images, regions)
    print(f"population: training livers mean {mean:.1f} HU, sd {sd:.1f} (smoothed, inside the rim)")

    def as_mask(case):
        m = sitk.GetImageFromArray(case[1])
        m.CopyInformation(case[0])
        return m

    methods = {
        "threshold, own liver": lambda c: score_own_liver(c[0], as_mask(c)),
        "threshold, population": lambda c: score_population(c[0], as_mask(c), mean, sd),
    }

    rows = []
    for name, method in methods.items():
        val_maps = [method(c) for c in val]
        threshold = threshold_for_one_fp(val_maps, [c[2] for c in val], [c[1] for c in val])

        test_maps = [method(c) for c in test]
        aurocs = [voxel_auroc(m, c[2], c[1]) for m, c in zip(test_maps, test) if (c[2] & c[1]).any()]
        found, fp = evaluate(test_maps, test, threshold)
        rows.append((name, np.nanmean(aurocs), threshold, found, fp))

        # step 0: how normal the normal livers are, by this method at this threshold
        if name == "threshold, own liver":
            flagged = []
            for case, c in zip(split["train"], train):
                blobs = connected_blobs(method(c) * c[1], threshold)
                voxel_ml = float(np.prod(c[0].GetSpacing())) / 1000
                big = [b for b in blobs if b["voxels"] * voxel_ml >= LARGE_ML]
                if big:
                    flagged.append(f"{case} ({max(b['voxels'] for b in big) * voxel_ml:.1f} ml)")

    print(f"\ntest: {len(test)} scans, {sum(len(c[3]) for c in test)} lesions; "
          f"thresholds chosen on val ({len(val)} scans)\n")
    print(f"{'method':24}{'AUROC':>7}{'z thr':>7}{'FP/scan':>9}  {'found':>12}"
          f"{'<15 mm':>12}{'>=15 mm':>12}{'subtle':>12}{'obvious':>12}")
    for name, auroc, threshold, found, fp in rows:
        print(f"{name:24}{auroc:7.3f}{threshold:7.2f}{fp:9.1f}  {rate(found):>12}"
              f"{rate(found, lambda r: 2 * r.radius_mm < SMALL_MM):>12}"
              f"{rate(found, lambda r: 2 * r.radius_mm >= SMALL_MM):>12}"
              f"{rate(found, lambda r: abs(r.contrast_hu) < SUBTLE_HU):>12}"
              f"{rate(found, lambda r: abs(r.contrast_hu) >= SUBTLE_HU):>12}")

    print(f"\nnormal-set check: {len(flagged)} of {len(train)} training livers have a flagged "
          f"cluster of {LARGE_ML} ml or more: {flagged}")


if __name__ == "__main__":
    main()
