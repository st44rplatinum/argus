"""Make one patient with four scans over time: longitudinal tracking, step 1.

Three liver lesions follow known courses (one grows, one holds, one shrinks).
Every scan after the first is moved and warped differently. The answer key,
data/derived/timeline_<case>.json, has each lesion's true place and size at
each date, and landmarks located in every scan.

  python scripts/make_timeline.py --dry-run      # build and check, push nothing
  python scripts/make_timeline.py
  python scripts/make_timeline.py --update-key   # refresh the key, keep the pushed scans
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "ai"))

from app.lesions import _choose_centre  # noqa: E402
from app.recist import axial_axes  # noqa: E402
from make_followup import DATASET, DERIVED, add_lesion, centroid, motion, push_series  # noqa: E402
from nifti_to_dicom import read_nifti  # noqa: E402

DATES = ["20240101", "20240701", "20250101", "20250701"]

# radius in mm at each date
COURSES = {
    "growing": [6.0, 8.0, 10.0, 12.0],
    "stable": [7.0, 7.0, 7.0, 7.0],
    "shrinking": [10.0, 8.0, 6.0, 5.0],
}

# how each later scan is moved: rotation in degrees, shift in mm
MOVES = [
    None,  # the first scan stays as it is
    ((1.5, -2.0, 2.0), (14.0, -11.0, 7.0)),
    ((-2.0, 1.0, -1.5), (-9.0, 12.0, -6.0)),
    ((1.0, 2.5, 1.0), (8.0, 6.0, 11.0)),
]
LANDMARKS = 30


def find_in_scan(transform: sitk.Transform, native_point, steps: int = 20) -> list[float]:
    """Where a native point sits in a moved scan.

    The transform goes scan -> native and a B-spline has no inverse, so step
    toward it: each guess is corrected by how far it lands from the target.
    """
    target = np.asarray(native_point, dtype=float)
    guess = target.copy()
    for _ in range(steps):
        guess += target - np.asarray(transform.TransformPoint(tuple(guess)))
    return [float(v) for v in guess]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", default="s0012")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--dry-run", action="store_true", help="build and check, push nothing")
    parser.add_argument("--update-key", action="store_true", help="rewrite the answer key without pushing again")
    args = parser.parse_args()

    path = DERIVED / f"timeline_{args.case}.json"
    pushed = {}
    if args.update_key:
        # the scans are rebuilt identically, so their uids still apply
        pushed = {t["date"]: t for t in json.loads(path.read_text(encoding="utf-8"))["timepoints"]}

    patient_id = f"TL-{args.case}"
    original = read_nifti(DATASET / args.case / "ct.nii.gz")
    liver = sitk.GetArrayFromImage(read_nifti(DATASET / args.case / "segmentations" / "liver.nii.gz")) > 0
    spacing_zyx = original.GetSpacing()[::-1]
    voxel_ml = float(np.prod(spacing_zyx)) / 1000

    # one spot per lesion, with room for its largest size, clear of the others
    rng = np.random.default_rng(args.seed)
    centres, taken = {}, []
    for name, radii in COURSES.items():
        centres[name] = _choose_centre(liver, spacing_zyx, max(radii), rng, taken)
        taken.append((centres[name], max(radii)))

    # landmarks: points inside the body of the unmoved scan
    body = np.argwhere(sitk.GetArrayFromImage(original) > -500)
    # more than needed: some will leave the field of view in a moved scan
    picks = body[rng.choice(len(body), LANDMARKS * 3, replace=False)]
    native_marks = [original.TransformIndexToPhysicalPoint((int(x), int(y), int(z))) for z, y, x in picks]

    timepoints, landmark_tracks = [], [[] for _ in native_marks]
    work = DERIVED / "timeline_dicom"

    for t, (date, move) in enumerate(zip(DATES, MOVES)):
        image, masks = original, {}
        for i, (name, radii) in enumerate(COURSES.items()):
            image, masks[name] = add_lesion(image, liver, centres[name], radii[t], args.seed + 10 * t + i)

        if move is None:
            transform = sitk.Transform(3, sitk.sitkIdentity)
        else:
            transform = motion(original, False, args.seed + 100 + t, degrees=move[0], shift_mm=move[1])
            image = sitk.Resample(image, original, transform, sitk.sitkLinear, -1024.0, sitk.sitkFloat32)

        lesions = []
        for name, radii in COURSES.items():
            mask = sitk.GetImageFromArray(masks[name].astype(np.uint8))
            mask.CopyInformation(original)
            moved = sitk.GetArrayFromImage(sitk.Resample(mask, original, transform, sitk.sitkNearestNeighbor, 0)) > 0
            axes = axial_axes(moved, spacing_zyx)
            lesions.append({
                "id": name,
                "radius_mm": radii[t],
                "volume_ml": round(float(moved.sum() * voxel_ml), 3),
                "long_axis_mm": round(axes["long_mm"], 2),
                "short_axis_mm": round(axes["short_mm"], 2),
                "centre": centroid(original, moved),
            })

        for track, mark in zip(landmark_tracks, native_marks):
            track.append(find_in_scan(transform, mark))

        entry = {"date": date, "lesions": lesions}
        if args.update_key:
            entry.update({k: pushed[date][k] for k in ("study_uid", "series_uid")})
        elif not args.dry_run:
            entry.update(push_series(image, work / date, f"Timeline {args.case}", patient_id, date))
        timepoints.append(entry)
        print(f"{date}  " + "  ".join(
            f"{l['id']} {l['long_axis_mm']:.1f} x {l['short_axis_mm']:.1f} mm" for l in lesions), flush=True)

    # keep landmarks that stay inside every scan; near the edge the warp is undefined
    size = np.array(original.GetSize())

    def inside(point):
        index = np.array(original.TransformPhysicalPointToContinuousIndex(tuple(point)))
        return bool(np.all(index >= 3) and np.all(index <= size - 4))

    kept = [i for i, track in enumerate(landmark_tracks) if all(inside(p) for p in track)][:LANDMARKS]
    landmark_tracks = [landmark_tracks[i] for i in kept]
    native_marks = [native_marks[i] for i in kept]

    # check the inversion: sending a found point back through must land on the mark
    worst = 0.0
    for t, move in enumerate(MOVES):
        if move is None:
            continue
        transform = motion(original, False, args.seed + 100 + t, degrees=move[0], shift_mm=move[1])
        for track, mark in zip(landmark_tracks, native_marks):
            worst = max(worst, float(np.linalg.norm(np.subtract(transform.TransformPoint(tuple(track[t])), mark))))
    print(f"{len(landmark_tracks)} landmarks located in every scan to within {worst:.4f} mm")

    if args.dry_run:
        print("dry run: nothing pushed, nothing written")
        return

    answer = {
        "case": args.case,
        "patient_id": patient_id,
        "convention": "points are mm, LPS, in each scan's own coordinates",
        "courses": COURSES,
        "timepoints": timepoints,
        # each landmark is one place in the body, given in every scan in date order
        "landmarks": landmark_tracks,
    }
    path.write_text(json.dumps(answer, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
