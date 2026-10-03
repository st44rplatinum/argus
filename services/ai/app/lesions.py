"""Planted liver lesions for P8: an answer key that never touches disk as CT.

A recipe says where a lesion goes and what it looks like. planted() rebuilds a
case from its recipes in a second, cropped to the liver, the same every time.

Arrays here are [z, y, x] and spacing is (z, y, x) mm. Everything stays in
voxel space, so no patient orientation is involved.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

DATASET = Path(__file__).resolve().parents[3] / "data" / "raw" / "totalsegmentator"
MARGIN_MM = 10.0


@dataclass(frozen=True)
class LesionRecipe:
    case: str
    centre_zyx: tuple[int, int, int]  # in the full scan, not the crop
    radius_mm: float
    contrast_hu: float
    edge: str  # "hard", "soft" (blurred) or "core" (darker centre)
    seed: int


def load_case(case: str):
    """Full CT and liver mask as [z, y, x], with spacing (z, y, x)."""
    ct = nib.load(DATASET / case / "ct.nii.gz")
    liver = nib.load(DATASET / case / "segmentations" / "liver.nii.gz")
    image = np.asarray(ct.dataobj, dtype=np.float32).transpose(2, 1, 0)
    mask = (np.asarray(liver.dataobj) > 0).transpose(2, 1, 0)
    spacing_zyx = tuple(float(v) for v in ct.header.get_zooms()[:3][::-1])
    return image, mask, spacing_zyx


def _choose_centre(liver_mask, spacing_zyx, radius_mm, rng, taken=()):
    """A point at least a radius inside the liver, clear of lesions already placed."""
    distance = ndimage.distance_transform_edt(liver_mask, sampling=spacing_zyx)
    candidates = np.argwhere(distance >= radius_mm + 2.0)
    if not len(candidates):
        raise ValueError(f"liver has no room for a {radius_mm:.1f} mm lesion")

    for _ in range(50):
        centre = candidates[rng.integers(len(candidates))]
        clear = all(
            np.linalg.norm((centre - np.array(c)) * spacing_zyx) > radius_mm + r + 5.0
            for c, r in taken
        )
        if clear:
            return tuple(int(v) for v in centre)
    raise ValueError("no room left for another lesion")


def make_recipes(case: str, seed: int) -> list[LesionRecipe]:
    """0-3 lesions of varied size, contrast and edge. About a quarter get none."""
    rng = np.random.default_rng(seed)
    _, liver, spacing = load_case(case)

    recipes, taken = [], []
    for i in range(int(rng.choice([0, 1, 2, 3], p=[0.25, 0.35, 0.25, 0.15]))):
        radius = float(rng.uniform(4.0, 15.0))
        # mostly darker than liver, as in portal venous; some brighter
        contrast = float(rng.uniform(-60, -15) if rng.random() < 0.8 else rng.uniform(20, 40))
        centre = _choose_centre(liver, spacing, radius, rng, taken)
        taken.append((centre, radius))
        recipes.append(LesionRecipe(
            case=case,
            centre_zyx=centre,
            radius_mm=round(radius, 2),
            contrast_hu=round(contrast, 1),
            edge=str(rng.choice(["hard", "soft", "core"])),
            seed=seed * 10 + i,
        ))
    return recipes


def add_lesion(image, liver_mask, recipe, spacing_zyx, offset=(0, 0, 0)):
    """Rebuild one lesion. `offset` is where the crop starts in the full scan.

    Returns the modified image and the lesion mask.
    """
    rng = np.random.default_rng(recipe.seed)
    spacing = np.asarray(spacing_zyx)
    centre = np.asarray(recipe.centre_zyx) - np.asarray(offset)

    # only a box around the lesion: a full-scan grid is gigabytes
    reach = np.ceil((recipe.radius_mm + 4.0) / spacing).astype(int)
    lo = np.maximum(centre - reach, 0)
    hi = np.minimum(centre + reach + 1, image.shape)
    box = tuple(slice(a, b) for a, b in zip(lo, hi))

    z, y, x = np.ogrid[box]
    distance = np.sqrt(((z - centre[0]) * spacing[0]) ** 2
                       + ((y - centre[1]) * spacing[1]) ** 2
                       + ((x - centre[2]) * spacing[2]) ** 2)
    sphere = distance <= recipe.radius_mm

    # how much of the contrast each voxel gets
    alpha = sphere.astype(np.float32)
    if recipe.edge == "soft":
        alpha = ndimage.gaussian_filter(alpha, sigma=1.5 / spacing)
    elif recipe.edge == "core":
        alpha[distance <= recipe.radius_mm / 2] = 1.6
    alpha *= liver_mask[box]

    # mild noise, so the lesion is not perfectly flat
    noise = rng.normal(0.0, max(abs(recipe.contrast_hu) * 0.08, 1.0), size=alpha.shape)

    result = image.astype(np.float32, copy=True)
    result[box] += alpha * (recipe.contrast_hu + noise).astype(np.float32)

    lesion = np.zeros(image.shape, dtype=np.uint8)
    lesion[box] = sphere & liver_mask[box]
    return result, lesion


def planted(case: str, recipes: list[LesionRecipe]):
    """A case with its lesions, cropped to the liver plus a margin.

    Returns (image, liver mask) as sitk images with the right spacing, and the
    lesion mask as a [z, y, x] array on the same crop.
    """
    image, liver, spacing = load_case(case)

    margin = np.ceil(MARGIN_MM / np.asarray(spacing)).astype(int)
    where = np.argwhere(liver)
    lo = np.maximum(where.min(axis=0) - margin, 0)
    hi = np.minimum(where.max(axis=0) + margin + 1, liver.shape)
    crop = tuple(slice(a, b) for a, b in zip(lo, hi))
    image, liver = image[crop], liver[crop]

    lesions = np.zeros(image.shape, dtype=np.uint8)
    for recipe in recipes:
        if recipe.case == case:
            image, lesion = add_lesion(image, liver, recipe, spacing, offset=lo)
            lesions |= lesion

    def to_sitk(array):
        out = sitk.GetImageFromArray(array)
        out.SetSpacing(spacing[::-1])
        return out

    return to_sitk(image), to_sitk(liver.astype(np.uint8)), lesions


def write_recipes(recipes: list[LesionRecipe], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in recipes], indent=2) + "\n", encoding="utf-8")


def read_recipes(path: Path) -> list[LesionRecipe]:
    return [
        LesionRecipe(
            case=item["case"],
            centre_zyx=tuple(item["centre_zyx"]),
            radius_mm=float(item["radius_mm"]),
            contrast_hu=float(item["contrast_hu"]),
            edge=item["edge"],
            seed=int(item["seed"]),
        )
        for item in json.loads(path.read_text(encoding="utf-8"))
    ]
