"""P8: score methods on the planted lesions and print the table.

Thresholds are chosen on val, then test is run once with them unchanged.
Everything is scored inside the liver minus a 3 mm rim (eval_anomaly.RIM_MM).
A method maps a case to (score map, scored region, one mask per lesion), so
the atlas can score in its own space and still be counted the same way.

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
    score_atlas,
    score_own_liver,
    score_population,
)
from app.eval_anomaly import scoring_region, threshold_for_one_fp, voxel_auroc  # noqa: E402
from app.lesions import planted, read_recipes  # noqa: E402

DERIVED = Path(__file__).resolve().parents[1] / "data/derived"
ATLAS = DERIVED / "atlas"
SMALL_MM = 15.0   # lesion diameter
SUBTLE_HU = 30.0  # |contrast|
LARGE_ML = 0.5    # a flagged cluster this big in a "normal" liver is worth a look
EDGE_MM = 5.0


def load(cases, recipes):
    out = []
    for case in cases:
        image, liver, _ = planted(case, recipes)
        mine = [r for r in recipes if r.case == case]
        out.append({
            "case": case,
            "image": image,
            "liver": liver,
            "region": scoring_region(sitk.GetArrayFromImage(liver), liver.GetSpacing()[::-1]),
            # one mask per lesion, so each can be followed into another space
            "lesions": [(planted(case, [r])[2], r) for r in mine],
        })
    return out


def union(lesions, shape):
    out = np.zeros(shape, np.uint8)
    for mask, _ in lesions:
        out |= mask
    return out


def as_image(array, like):
    out = sitk.GetImageFromArray(array)
    out.CopyInformation(like)
    return out


def threshold_method(score):
    def run(c):
        return score(c["image"], as_image(c["region"], c["image"])), c["region"], c["lesions"]
    return run


def atlas_method(c):
    result = score_atlas(c["image"], c["liver"], atlas_dir=ATLAS)
    template = sitk.ReadImage(str(ATLAS / "atlas_liver.nii.gz"))
    lesions = [
        (sitk.GetArrayFromImage(sitk.Resample(as_image(mask, c["image"]), template, result.transform,
                                              sitk.sitkNearestNeighbor, 0)), r)
        for mask, r in c["lesions"]
    ]
    return result.score, result.region, lesions


def evaluate(scored, threshold, edge=None):
    """Per lesion: found or not. Plus false blobs per scan, and how many sit near the edge."""
    found, false_blobs, near_edge = [], 0, 0
    for score, region, lesions in scored:
        blobs = connected_blobs(score * region, threshold)
        peaks = [tuple(b["peak_zyx"]) for b in blobs]
        for mask, recipe in lesions:
            found.append((any(mask[p] for p in peaks), recipe))
        every = union(lesions, score.shape)
        for p in peaks:
            if not every[p]:
                false_blobs += 1
                near_edge += bool(edge is not None and edge[p])
    return found, false_blobs / len(scored), near_edge / max(false_blobs, 1)


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

    mean, sd = population_statistics(
        [c["image"] for c in train], [as_image(c["region"], c["image"]) for c in train]
    )
    print(f"population: training livers mean {mean:.1f} HU, sd {sd:.1f} (smoothed, inside the rim)")

    methods = {
        "threshold, own liver": threshold_method(score_own_liver),
        "threshold, population": threshold_method(lambda i, m: score_population(i, m, mean, sd)),
        "atlas z-score": atlas_method,
    }

    # within EDGE_MM of the template liver's edge, for the atlas's false blobs
    template_liver = sitk.GetArrayFromImage(sitk.ReadImage(str(ATLAS / "atlas_liver.nii.gz"))).astype(bool)
    spacing = sitk.ReadImage(str(ATLAS / "atlas_liver.nii.gz")).GetSpacing()[::-1]
    template_edge = ndimage.distance_transform_edt(template_liver, sampling=spacing) <= EDGE_MM

    rows = []
    for name, method in methods.items():
        print(f"scoring: {name}", flush=True)
        val_scored = [method(c) for c in val]
        threshold = threshold_for_one_fp(
            [s for s, _, _ in val_scored],
            [union(l, s.shape) for s, _, l in val_scored],
            [r for _, r, _ in val_scored],
        )

        test_scored = [method(c) for c in test]
        aurocs = [voxel_auroc(s, union(l, s.shape), r) for s, r, l in test_scored if (union(l, s.shape) & r).any()]
        edge = template_edge if name == "atlas z-score" else None
        found, fp, edge_share = evaluate(test_scored, threshold, edge)
        rows.append((name, np.nanmean(aurocs), threshold, found, fp, edge_share if edge is not None else None))

        # step 0: how normal the normal livers are, by this method at this threshold
        if name == "threshold, own liver":
            flagged = []
            for c in train:
                score, region, _ = method(c)
                voxel_ml = float(np.prod(c["image"].GetSpacing())) / 1000
                big = [b for b in connected_blobs(score * region, threshold) if b["voxels"] * voxel_ml >= LARGE_ML]
                if big:
                    flagged.append(f"{c['case']} ({max(b['voxels'] for b in big) * voxel_ml:.1f} ml)")

    print(f"\ntest: {len(test)} scans, {sum(len(c['lesions']) for c in test)} lesions; "
          f"thresholds chosen on val ({len(val)} scans)\n")
    print(f"{'method':24}{'AUROC':>7}{'z thr':>7}{'FP/scan':>9}  {'found':>12}"
          f"{'<15 mm':>12}{'>=15 mm':>12}{'subtle':>12}{'obvious':>12}")
    for name, auroc, threshold, found, fp, _ in rows:
        print(f"{name:24}{auroc:7.3f}{threshold:7.2f}{fp:9.1f}  {rate(found):>12}"
              f"{rate(found, lambda r: 2 * r.radius_mm < SMALL_MM):>12}"
              f"{rate(found, lambda r: 2 * r.radius_mm >= SMALL_MM):>12}"
              f"{rate(found, lambda r: abs(r.contrast_hu) < SUBTLE_HU):>12}"
              f"{rate(found, lambda r: abs(r.contrast_hu) >= SUBTLE_HU):>12}")

    for name, *_, edge_share in rows:
        if edge_share is not None:
            print(f"\n{name}: {edge_share:.0%} of false blobs within {EDGE_MM:.0f} mm of the template liver's edge")
    print(f"normal-set check: {len(flagged)} of {len(train)} training livers have a flagged "
          f"cluster of {LARGE_ML} ml or more: {flagged}")


if __name__ == "__main__":
    main()
