"""What one organ looks like, as a short vector.

Every number is something a radiologist could name: how dense, how varied,
how big, how round, how busy. Nothing here is learned, so a case's description
never depends on a downloaded model — and when a retrieval result is strange,
you can look at which number caused it.

This is the baseline the learned embedding has to beat. Write its number down.

pyradiomics would have covered this ground, but it has no Python 3.13 wheels
and its build is broken, and a dependency that stale is not worth the
correctness it would have bought here.
"""

from __future__ import annotations

import numpy as np

# Bumped when the numbers below change, so an index built by an older version
# is recomputed rather than compared against.
FEATURE_VERSION = 1

NAMES = (
    "mean_hu",
    "std_hu",
    "p10_hu",
    "p90_hu",
    "skew_hu",
    "volume_ml",
    "surface_to_volume",
    "elongation",
    "flatness",
    "gradient_mean",
    "heterogeneity",
)

LEN = len(NAMES)


def _shape_axes(mask: np.ndarray, spacing: tuple[float, float, float]) -> tuple[float, float]:
    """Elongation and flatness, from the spread of the voxel cloud.

    The eigenvalues of the coordinate covariance are the squared lengths of
    the principal axes. Their ratios say whether an organ is cigar-shaped or
    pancake-shaped without needing a mesh.
    """
    coords = np.argwhere(mask)
    if len(coords) < 4:
        return 0.0, 0.0

    # Into millimetres, so the ratios are not an artefact of anisotropy.
    physical = coords * np.array(spacing, dtype=np.float64)
    centred = physical - physical.mean(axis=0)

    eigenvalues = np.linalg.eigvalsh(np.cov(centred, rowvar=False))
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    major, middle, minor = sorted(eigenvalues, reverse=True)

    if major <= 0:
        return 0.0, 0.0

    elongation = float(np.sqrt(middle / major))
    flatness = float(np.sqrt(minor / major))
    return elongation, flatness


def _surface_voxels(mask: np.ndarray) -> int:
    """Voxels with at least one face not inside the mask."""
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    interior = (
        padded[:-2, 1:-1, 1:-1]
        & padded[2:, 1:-1, 1:-1]
        & padded[1:-1, :-2, 1:-1]
        & padded[1:-1, 2:, 1:-1]
        & padded[1:-1, 1:-1, :-2]
        & padded[1:-1, 1:-1, 2:]
    )
    return int((mask & ~interior).sum())


def organ_features(
    volume: np.ndarray,
    mask: np.ndarray,
    spacing: tuple[float, float, float],
) -> np.ndarray | None:
    """Describe one organ. None when it is absent or too small to describe.

    `spacing` is (z, y, x) in millimetres, matching Volume.spacing.
    """
    mask = mask.astype(bool)
    count = int(mask.sum())

    # Under a hundred voxels is a sliver at the edge of the field of view,
    # not an organ, and its shape statistics are noise.
    if count < 100:
        return None

    values = volume[mask].astype(np.float64)
    mean = float(values.mean())
    std = float(values.std())

    voxel_ml = float(np.prod(spacing)) / 1000.0
    volume_ml = count * voxel_ml

    surface = _surface_voxels(mask)
    elongation, flatness = _shape_axes(mask, spacing)

    # Gradient magnitude inside the organ: high in a heterogeneous liver,
    # low in a uniform one. Computed on the whole array then sampled, because
    # a gradient at the organ's edge would mostly measure the edge.
    gz, gy, gx = np.gradient(volume.astype(np.float32))
    gradient = np.sqrt(gz**2 + gy**2 + gx**2)
    interior = mask & ~_edge_mask(mask)
    gradient_mean = float(gradient[interior].mean()) if interior.any() else 0.0

    # How much of the organ sits far from its own middle, in its own units —
    # scale-free, so a contrast phase shift does not dominate it.
    heterogeneity = (
        float((np.abs(values - mean) > std).mean()) if std > 0 else 0.0
    )

    skew = float(((values - mean) ** 3).mean() / (std**3)) if std > 0 else 0.0

    return np.array(
        [
            mean,
            std,
            float(np.percentile(values, 10)),
            float(np.percentile(values, 90)),
            skew,
            volume_ml,
            surface / count if count else 0.0,
            elongation,
            flatness,
            gradient_mean,
            heterogeneity,
        ],
        dtype=np.float64,
    )


def _edge_mask(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    interior = (
        padded[:-2, 1:-1, 1:-1]
        & padded[2:, 1:-1, 1:-1]
        & padded[1:-1, :-2, 1:-1]
        & padded[1:-1, 2:, 1:-1]
        & padded[1:-1, 1:-1, :-2]
        & padded[1:-1, 1:-1, 2:]
    )
    return mask & ~interior
