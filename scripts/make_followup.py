"""Make a synthetic follow-up from one case: P7 step 1.

The prior is the case plus a small liver lesion. The follow-up is the same
anatomy moved, gently warped, with the lesion grown. Both are pushed to
Orthanc as one patient a year apart, and the answer key goes to
data/derived/followup_<case>.json.

The motion T maps a follow-up (current) point to the prior point it came
from, the same direction registration recovers. So the answer key is just
landmark pairs: p in the follow-up, T(p) in the prior.

  python scripts/make_followup.py
  python scripts/make_followup.py --rigid-only
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
from scipy import ndimage

from nifti_to_dicom import read_nifti, write_series
from push_dicom import DEFAULT_AUTH, DEFAULT_URL, push

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/raw/totalsegmentator"
DERIVED = ROOT / "data/derived"
LANDMARKS = 30


def deepest_point(mask: np.ndarray, spacing_zyx) -> tuple[int, int, int]:
    distance = ndimage.distance_transform_edt(mask, sampling=spacing_zyx)
    return np.unravel_index(np.argmax(distance), distance.shape)


def add_lesion(image: sitk.Image, liver: np.ndarray, centre_zyx, radius_mm: float, seed: int):
    """Sphere 40 HU darker than liver, as metastases look in portal venous phase."""
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    spacing_zyx = np.array(image.GetSpacing()[::-1])

    zz, yy, xx = np.indices(array.shape, sparse=True)
    distance2 = (
        ((zz - centre_zyx[0]) * spacing_zyx[0]) ** 2
        + ((yy - centre_zyx[1]) * spacing_zyx[1]) ** 2
        + ((xx - centre_zyx[2]) * spacing_zyx[2]) ** 2
    )
    lesion = (distance2 <= radius_mm**2) & liver

    rng = np.random.default_rng(seed)
    array[lesion] = array[liver].mean() - 40.0 + rng.normal(0.0, 3.0, int(lesion.sum()))

    out = sitk.GetImageFromArray(array)
    out.CopyInformation(image)
    return out, lesion


def motion(image: sitk.Image, rigid_only: bool, seed: int) -> sitk.Transform:
    """Current point -> prior point: a gentle warp, then a rigid move."""
    rigid = sitk.Euler3DTransform()
    rigid.SetCenter(image.TransformContinuousIndexToPhysicalPoint([s / 2.0 for s in image.GetSize()]))
    rigid.SetRotation(math.radians(1.5), math.radians(-2.0), math.radians(2.0))
    rigid.SetTranslation((14.0, -11.0, 7.0))
    if rigid_only:
        return rigid

    warp = sitk.BSplineTransformInitializer(image, [8, 8, 8], order=3)
    rng = np.random.default_rng(seed)
    warp.SetParameters(tuple(rng.normal(0.0, 2.5, len(warp.GetParameters()))))

    # last added is applied first: warp, then rigid
    composite = sitk.CompositeTransform(3)
    composite.AddTransform(rigid)
    composite.AddTransform(warp)
    return composite


def centroid(image: sitk.Image, mask: np.ndarray) -> list[float]:
    z, y, x = np.argwhere(mask).mean(axis=0)
    return list(image.TransformContinuousIndexToPhysicalPoint((float(x), float(y), float(z))))


def push_series(image: sitk.Image, out_dir: Path, description: str, patient_id: str, date: str) -> dict:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    write_series(image, out_dir, description, patient_id, study_date=date)
    ds = pydicom.dcmread(next(out_dir.glob("*.dcm")), stop_before_pixels=True)
    push(out_dir, DEFAULT_URL, DEFAULT_AUTH)
    shutil.rmtree(out_dir)
    return {"study_uid": str(ds.StudyInstanceUID), "series_uid": str(ds.SeriesInstanceUID), "date": date}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", default="s0012")
    parser.add_argument("--rigid-only", action="store_true")
    parser.add_argument("--seed", type=int, default=12)
    args = parser.parse_args()

    suffix = "-rigid" if args.rigid_only else ""
    patient_id = f"FU-{args.case}{suffix}"

    original = read_nifti(DATASET / args.case / "ct.nii.gz")
    liver_image = read_nifti(DATASET / args.case / "segmentations" / "liver.nii.gz")
    liver = sitk.GetArrayFromImage(liver_image) > 0
    if liver.shape != sitk.GetArrayFromImage(original).shape:
        raise SystemExit("CT and liver mask have different sizes")

    centre = deepest_point(liver, original.GetSpacing()[::-1])
    prior, prior_lesion = add_lesion(original, liver, centre, 8.0, args.seed)
    grown, grown_lesion = add_lesion(original, liver, centre, 11.0, args.seed + 1)

    # Follow-up pixel at p takes the native value at T(p).
    transform = motion(original, args.rigid_only, args.seed + 2)
    followup = sitk.Resample(grown, original, transform, sitk.sitkLinear, -1024.0, sitk.sitkFloat32)

    lesion_image = sitk.GetImageFromArray(grown_lesion.astype(np.uint8))
    lesion_image.CopyInformation(original)
    followup_lesion = sitk.GetArrayFromImage(
        sitk.Resample(lesion_image, original, transform, sitk.sitkNearestNeighbor, 0)
    ) > 0

    # Landmarks: random points inside the follow-up's body, and where they came from.
    rng = np.random.default_rng(args.seed + 3)
    body = np.argwhere(sitk.GetArrayFromImage(followup) > -500)
    picks = body[rng.choice(len(body), LANDMARKS, replace=False)]
    landmarks = []
    for z, y, x in picks:
        current = original.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))
        landmarks.append({"current": list(current), "prior": list(transform.TransformPoint(current))})

    voxel_mm3 = float(np.prod(original.GetSpacing()))
    prior_ml = prior_lesion.sum() * voxel_mm3 / 1000
    followup_ml = followup_lesion.sum() * voxel_mm3 / 1000

    work = DERIVED / "followup_dicom"
    prior_uids = push_series(prior, work / "prior", f"Prior {args.case}", patient_id, "20250101")
    followup_uids = push_series(followup, work / "followup", f"Follow-up {args.case}", patient_id, "20260101")
    shutil.rmtree(work, ignore_errors=True)

    answer = {
        "case": args.case,
        "patient_id": patient_id,
        "rigid_only": args.rigid_only,
        "convention": "landmark 'current' is a follow-up point, 'prior' is where it was; mm, LPS",
        "prior": {**prior_uids, "lesion_centre": centroid(original, prior_lesion), "lesion_ml": round(prior_ml, 3)},
        "followup": {**followup_uids, "lesion_centre": centroid(original, followup_lesion), "lesion_ml": round(followup_ml, 3)},
        "expected_change_percent": round((followup_ml - prior_ml) / prior_ml * 100, 1),
        "landmarks": landmarks,
    }

    path = DERIVED / f"followup_{args.case}{suffix}.json"
    path.write_text(json.dumps(answer, indent=2) + "\n", encoding="utf-8")
    print(f"prior lesion {prior_ml:.2f} ml, follow-up {followup_ml:.2f} ml, "
          f"change {answer['expected_change_percent']}%")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
