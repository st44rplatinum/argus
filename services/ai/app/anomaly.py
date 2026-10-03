"""Liver anomaly scores for P8, higher meaning more abnormal.

Three levels of scoring:

Patient specific - compares each voxel against the patient's own liver.
Population - compares against a mean and sd learned from training livers.
Template - compares against a registered 3D atlas, a mean and sd per voxel.
"""

from __future__ import annotations

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
    image: sitk.Image,
    liver_mask: sitk.Image,
    *,
    population_mean: float | None = None,
    population_sd: float | None = None,
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


def save_atlas(
    mean: sitk.Image,
    sd: sitk.Image,
    liver_mask: sitk.Image,
    directory: Path,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)

    sitk.WriteImage(mean, str(directory / "atlas_mean.nii.gz"))
    sitk.WriteImage(sd, str(directory / "atlas_sd.nii.gz"))
    sitk.WriteImage(liver_mask, str(directory / "atlas_liver.nii.gz"))


def load_atlas(directory: Path):
    mean = sitk.ReadImage(str(directory / "atlas_mean.nii.gz"))
    sd = sitk.ReadImage(str(directory / "atlas_sd.nii.gz"))
    liver = sitk.ReadImage(str(directory / "atlas_liver.nii.gz"))

    return mean, sd, liver


def score_atlas(
    image_in_template: sitk.Image,
    liver_in_template: sitk.Image,
    atlas_mean: sitk.Image,
    atlas_sd: sitk.Image,
) -> np.ndarray:
    values = sitk.GetArrayViewFromImage(image_in_template).astype(np.float32)
    mean = sitk.GetArrayViewFromImage(atlas_mean).astype(np.float32)
    sd = sitk.GetArrayViewFromImage(atlas_sd).astype(np.float32)
    liver = sitk.GetArrayViewFromImage(liver_in_template).astype(bool)

    score = np.zeros_like(values, dtype=np.float32)

    safe_sd = np.maximum(sd, 5.0)
    score[liver] = np.abs(values[liver] - mean[liver]) / safe_sd[liver]

    return score
