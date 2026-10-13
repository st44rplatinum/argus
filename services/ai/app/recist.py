"""Lesion diameters the way RECIST reports them.

Both are measured on one axial slice: the long axis is the longest line
across the lesion, the short axis the widest extent at right angles to it.
The slice is the one where the long axis is greatest.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def _edge(mask_2d: np.ndarray) -> np.ndarray:
    """Outline pixels as (row, col): the longest line always ends on the outline."""
    return np.argwhere(mask_2d & ~ndimage.binary_erosion(mask_2d))


def axial_axes(mask: np.ndarray, spacing_zyx) -> dict:
    """Long and short axis in mm for a [z, y, x] mask, and where they are.

    Distances run between pixel centres, so one pixel is added to reach the
    lesion's outer edges; without it a 20 mm sphere reads about 19.
    """
    sy, sx = float(spacing_zyx[1]), float(spacing_zyx[2])
    pixel = (sy + sx) / 2.0
    best = {"long_mm": 0.0, "short_mm": 0.0, "slice": None, "long_ends_zyx": None}

    for z in np.flatnonzero(mask.any(axis=(1, 2))):
        points = _edge(mask[z]) * (sy, sx)
        if len(points) < 2:
            continue

        gaps = points[:, None, :] - points[None, :, :]
        distance = np.sqrt((gaps**2).sum(-1))
        i, j = np.unravel_index(np.argmax(distance), distance.shape)
        if distance[i, j] + pixel <= best["long_mm"]:
            continue

        # short axis: how far the outline spreads across the long axis
        along = (points[j] - points[i]) / distance[i, j]
        across = points @ np.array([-along[1], along[0]])
        best = {
            "long_mm": float(distance[i, j] + pixel),
            "short_mm": float(across.max() - across.min() + pixel),
            "slice": int(z),
            "long_ends_zyx": [
                [int(z), *(points[i] / (sy, sx)).round().astype(int).tolist()],
                [int(z), *(points[j] / (sy, sx)).round().astype(int).tolist()],
            ],
        }

    return best
