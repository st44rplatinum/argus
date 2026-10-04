"""P8 step 4: a normal-liver atlas from the training livers.

The template is the training liver nearest the median volume. Every other
one is registered onto it by shape (anomaly.align_liver), less its own
median HU, and summed per voxel. Writes data/derived/atlas/.

  python scripts/build_atlas.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services" / "ai"))

from app.anomaly import align_liver, save_atlas, to_template  # noqa: E402
from app.lesions import DATASET, planted  # noqa: E402

DERIVED = Path(__file__).resolve().parents[1] / "data/derived"
MIN_DICE = 0.85


def main() -> None:
    split = json.loads((DERIVED / "anomaly_split.json").read_text(encoding="utf-8"))
    train = split["train"]

    # a mirrored scan can't be fixed by registration
    for case in train:
        codes = "".join(nib.aff2axcodes(nib.load(DATASET / case / "ct.nii.gz").affine))
        assert codes == "RAS", f"{case} is stored {codes}, not RAS"

    # volumes from the masks alone; each CT is loaded once, below, and let go
    volumes = {}
    for case in train:
        mask = nib.load(DATASET / case / "segmentations" / "liver.nii.gz")
        volumes[case] = float((np.asarray(mask.dataobj) > 0).sum() * np.prod(mask.header.get_zooms()[:3]) / 1000)
    median = float(np.median(list(volumes.values())))
    name = min(volumes, key=lambda c: abs(volumes[c] - median))
    template, template_liver, _ = planted(name, [])
    template = sitk.Cast(template, sitk.sitkFloat32)
    print(f"template {name}, {volumes[name]:.0f} ml (median {median:.0f})", flush=True)

    shape = sitk.GetArrayFromImage(template).shape
    total = np.zeros(shape, np.float64)
    squares = np.zeros(shape, np.float64)
    count = np.zeros(shape, np.uint16)
    used, dropped = [], []

    for case in train:
        image, liver, _ = planted(case, [])
        started = time.time()
        if case == name:
            transform, overlap = sitk.Transform(3, sitk.sitkIdentity), 1.0
        else:
            transform, overlap = align_liver(template_liver, liver)

        if overlap < MIN_DICE:
            dropped.append(case)
            print(f"  {case}  Dice {overlap:.3f}  dropped", flush=True)
            continue

        values, inside = to_template(image, liver, transform, template)
        total[inside] += values[inside]
        squares[inside] += values[inside] ** 2
        count[inside] += 1
        used.append({"case": case, "dice": round(overlap, 3)})
        print(f"  {case}  Dice {overlap:.3f}  {time.time() - started:.0f}s", flush=True)

    covered = count > 0
    mean = np.zeros(shape, np.float32)
    sd = np.zeros(shape, np.float32)
    mean[covered] = total[covered] / count[covered]
    sd[covered] = np.sqrt(np.maximum(squares[covered] / count[covered] - mean[covered] ** 2, 0))

    def like_template(array):
        out = sitk.GetImageFromArray(array)
        out.CopyInformation(template)
        return out

    out = DERIVED / "atlas"
    save_atlas(template, template_liver, like_template(mean), like_template(sd), like_template(count), out)
    (out / "atlas.json").write_text(json.dumps({
        "template": name, "min_dice": MIN_DICE, "used": used, "dropped": dropped,
    }, indent=2) + "\n", encoding="utf-8")

    dices = [u["dice"] for u in used if u["case"] != name]
    print(f"\n{len(used)} livers in the atlas, {len(dropped)} dropped; "
          f"Dice median {np.median(dices):.3f}, worst {min(dices):.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
