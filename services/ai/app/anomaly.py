"""Liver anomaly scores for P8, higher meaning more abnormal.

Three levels of scoring:

Patient specific - compares each voxel against the patient's own liver.
Population - compares against a mean and sd learned from training livers.
Template - compares against a registered 3D atlas, a mean and sd per voxel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

def _smooth(image: sitk.Image, sigma_mm: float = 1.0) -> sitk.Image:
    """Gaussian smoothing in physical millimetres."""
    # variance is in mm² already: sitk uses the image spacing by default
    return sitk.DiscreteGaussian(image, variance=sigma_mm**2)


def _robust_sigma(values: np.ndarray) -> float:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    sigma = 1.4826 * mad

    # Prevent a very homogeneous synthetic liver from producing
    # absurdly large scores from floating point noise.
    return max(sigma, 1.0)


def score_threshold(
    image,
    liver_mask,
    *,
    mode="own",
    population_mean=None,
    population_sd=None,
) -> np.ndarray:
    """
    Population z-score.

    If population statistics are omitted, this becomes the patient's
    own-liver robust z-score. Scores are zero outside the liver.
    """
    smooth = _smooth(image)
    values = sitk.GetArrayViewFromImage(smooth).astype(np.float32)
    mask = sitk.GetArrayViewFromImage(liver_mask).astype(bool)

    if not mask.any():
        raise ValueError("liver mask is empty")

    liver_values = values[mask]

    if population_mean is None or population_sd is None:
        center = float(np.median(liver_values))
        scale = _robust_sigma(liver_values)
    else:
        center = float(population_mean)
        scale = max(float(population_sd), 1.0)

    score = np.zeros_like(values, dtype=np.float32)
    score[mask] = np.abs(values[mask] - center) / scale

    return score


def score_population(
    image: sitk.Image,
    liver_mask: sitk.Image,
    mean: float,
    sd: float,
) -> np.ndarray:
    return score_threshold(
        image,
        liver_mask,
        population_mean=mean,
        population_sd=sd,
    )


def score_own_liver(
    image: sitk.Image,
    liver_mask: sitk.Image,
) -> np.ndarray:
    return score_threshold(image, liver_mask)


def connected_blobs(
    score: np.ndarray,
    threshold: float,
    *,
    min_voxels: int = 3,
) -> list[dict]:
    """26-connected regions at or above the threshold, with their peaks.

    Peaks are array coordinates: z, y, x.
    """
    labels, count = ndimage.label(score >= threshold, structure=np.ones((3, 3, 3)))
    if count == 0:
        return []

    # one pass over the volume for every blob at once
    index = np.arange(1, count + 1)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)[1:]
    peaks = ndimage.maximum_position(score, labels, index)
    values = ndimage.maximum(score, labels, index)

    return [
        {
            "label": int(label),
            "voxels": int(size),
            "peak_score": float(value),
            "peak_zyx": [int(v) for v in peak],
        }
        for label, size, peak, value in zip(index, sizes, peaks, values)
        if size >= min_voxels
    ]


def population_statistics(
    images: list[sitk.Image],
    liver_masks: list[sitk.Image],
) -> tuple[float, float]:
    """Pooled training-liver HU statistics."""
    chunks = []

    for image, mask in zip(images, liver_masks):
        # smoothed, as score_threshold smooths what it scores
        values = sitk.GetArrayFromImage(_smooth(image))
        liver = sitk.GetArrayViewFromImage(mask).astype(bool)

        if not liver.any():
            continue

        chunks.append(values[liver].astype(np.float64))

    if not chunks:
        raise ValueError("no training liver voxels")

    values = np.concatenate(chunks)

    return float(values.mean()), float(values.std(ddof=1))


# atlas voxels covered by fewer training livers than this are not scored
MIN_COUNT = 10


def save_atlas(template: sitk.Image, liver: sitk.Image, mean: sitk.Image, sd: sitk.Image,
               count: sitk.Image, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, image in (("template", template), ("liver", liver), ("mean", mean),
                        ("sd", sd), ("count", count)):
        sitk.WriteImage(image, str(directory / f"atlas_{name}.nii.gz"))


def load_atlas(directory: Path) -> dict[str, sitk.Image]:
    """template CT, its liver, and per voxel: mean, sd, and how many livers cover it."""
    return {
        name: sitk.ReadImage(str(directory / f"atlas_{name}.nii.gz"))
        for name in ("template", "liver", "mean", "sd", "count")
    }


@dataclass
class AtlasScore:
    score: np.ndarray       # template space, zero where not scored
    region: np.ndarray      # where the score means something
    transform: sitk.Transform | None
    dice: float


def score_atlas(
    image: sitk.Image,
    liver_mask: sitk.Image,
    *,
    atlas_dir: Path,
    register_to_template: bool = True,
    rim_mm: float = 3.0,
) -> AtlasScore:
    """z against the atlas, voxel by voxel, in template space.

    Scored only where the template's liver (less the rim), the case's warped
    liver, and enough training livers all agree. Pass register_to_template=
    False for a case already on the template grid.
    """
    atlas = load_atlas(atlas_dir)

    if register_to_template:
        transform, overlap = align_liver(atlas["liver"], liver_mask)
    else:
        transform, overlap = sitk.Transform(3, sitk.sitkIdentity), 1.0
    values, _ = to_template(image, liver_mask, transform, atlas["template"])

    def inner(mask: sitk.Image) -> np.ndarray:
        """The liver less the rim, the same rule every method is scored by."""
        array = sitk.GetArrayFromImage(mask).astype(bool)
        return ndimage.distance_transform_edt(array, sampling=mask.GetSpacing()[::-1]) > rim_mm

    # the rim on both livers: the template's, and the case's own, warped in
    case_inner = sitk.GetImageFromArray(inner(liver_mask).astype(np.uint8))
    case_inner.CopyInformation(liver_mask)
    warped_inner = sitk.GetArrayFromImage(
        sitk.Resample(case_inner, atlas["liver"], transform, sitk.sitkNearestNeighbor, 0)
    ).astype(bool)
    region = inner(atlas["liver"]) & warped_inner & (sitk.GetArrayFromImage(atlas["count"]) >= MIN_COUNT)

    mean = sitk.GetArrayFromImage(atlas["mean"])
    # a floor, so a voxel where every liver happened to agree can't explode
    sd = np.maximum(sitk.GetArrayFromImage(atlas["sd"]), 5.0)

    score = np.zeros(values.shape, dtype=np.float32)
    score[region] = np.abs(values[region] - mean[region]) / sd[region]
    return AtlasScore(score, region.astype(np.uint8), transform, overlap)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    total = a.sum() + b.sum()
    return float(2 * (a & b).sum() / total) if total else 0.0


def _shape(mask: sitk.Image, mm: float = 3.0) -> sitk.Image:
    """A liver mask as a signed distance map on a 3 mm grid: the shape, nothing else."""
    size = [int(round(n * s / mm)) for n, s in zip(mask.GetSize(), mask.GetSpacing())]
    grid = sitk.Image(size, sitk.sitkUInt8)
    grid.SetSpacing((mm,) * 3)
    grid.SetOrigin(mask.GetOrigin())
    coarse = sitk.Resample(mask, grid, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
    distance = sitk.SignedMaurerDistanceMap(coarse, insideIsPositive=False, squaredDistance=False, useImageSpacing=True)
    # far from the liver nothing matters, so cap it
    return sitk.Clamp(distance, sitk.sitkFloat32, -30.0, 30.0)


def align_liver(template_liver: sitk.Image, liver: sitk.Image, mesh_mm: float = 40.0):
    """Register a case's liver shape onto the template's: affine, then B-spline.

    Only the outlines are matched, never the CT values. Between two people,
    matching CT values pulled livers apart (Dice 0.57 against 0.73-0.81 for
    just lining up centres), and a shape-only match cannot warp away a lesion
    inside the liver.

    Returns the transform (template point -> case point) and the Dice of the
    warped liver against the template's.
    """
    fixed, moving = _shape(template_liver), _shape(liver)

    def method():
        r = sitk.ImageRegistrationMethod()
        r.SetMetricAsMeanSquares()
        r.SetMetricSamplingStrategy(r.RANDOM)
        r.SetMetricSamplingPercentage(0.05)
        r.SetInterpolator(sitk.sitkLinear)
        return r

    # affine, so a bigger liver can be scaled down to the template
    start = sitk.CenteredTransformInitializer(
        template_liver, liver, sitk.AffineTransform(3), sitk.CenteredTransformInitializerFilter.MOMENTS
    )
    r = method()
    r.SetOptimizerAsRegularStepGradientDescent(1.0, 0.001, 200, relaxationFactor=0.7)
    r.SetOptimizerScalesFromPhysicalShift()
    r.SetShrinkFactorsPerLevel([2, 1])
    r.SetSmoothingSigmasPerLevel([3, 0])
    r.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    r.SetInitialTransform(start, inPlace=False)
    affine = r.Execute(fixed, moving)

    # then a B-spline for the shape, control points about every 40 mm
    extent = np.array(fixed.GetSize()) * np.array(fixed.GetSpacing())
    bspline = sitk.BSplineTransformInitializer(fixed, [max(1, round(v / mesh_mm)) for v in extent], order=3)
    r = method()
    r.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=100,
                           maximumNumberOfCorrections=5, maximumNumberOfFunctionEvaluations=1000,
                           costFunctionConvergenceFactor=1e7)
    r.SetMovingInitialTransform(affine)
    r.SetInitialTransform(bspline, inPlace=True)
    r.Execute(fixed, moving)

    transform = sitk.CompositeTransform(3)
    transform.AddTransform(affine)
    transform.AddTransform(bspline)
    transform.FlattenTransform()

    warped = sitk.Resample(liver, template_liver, transform, sitk.sitkNearestNeighbor, 0)
    return transform, dice(sitk.GetArrayFromImage(warped), sitk.GetArrayFromImage(template_liver))


def to_template(image: sitk.Image, liver: sitk.Image, transform, template: sitk.Image):
    """A case in template space, less its own liver median, and its warped liver."""
    values = sitk.GetArrayFromImage(image)
    median = float(np.median(values[sitk.GetArrayFromImage(liver) > 0]))
    shifted = sitk.Cast(image, sitk.sitkFloat32) - median

    warped = sitk.Resample(shifted, template, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    warped_liver = sitk.Resample(liver, template, transform, sitk.sitkNearestNeighbor, 0)
    return sitk.GetArrayFromImage(warped), sitk.GetArrayFromImage(warped_liver).astype(bool)
