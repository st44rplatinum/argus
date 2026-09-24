from __future__ import annotations

from collections import defaultdict

import numpy as np


INTENSITY_COLUMNS = (
    "mean_hu",
    "std_hu",
    "p10_hu",
    "p90_hu",
    "skew_hu",
    "gradient_mean",
    "heterogeneity",
)


def center_by_group(
    vectors,
    organs,
    phases,
    columns=None,
    min_group=5,
    leave_one_out=True,
):
    """
    Remove the contrast-phase shift from each vector.

    Each row loses its phase offset: the mean of its (organ, phase) group
    minus the mean of its organ. The organ mean itself stays, so a liver
    still looks like a liver. Rows with no phase, or in a group smaller than
    min_group, are left alone.

    If columns is given, only the intensity columns change. If columns is
    None, the whole vector changes and rows are re-normalised to unit length.

    leave_one_out: a row is not part of the means it is corrected by.
    """
    x = np.asarray(vectors, dtype=np.float64).copy()
    organs = np.asarray(organs, dtype=object)
    phases = np.asarray(phases, dtype=object)

    if x.ndim != 2:
        raise ValueError(f"vectors must be 2-D, got {x.shape}")

    if len(x) != len(organs) or len(x) != len(phases):
        raise ValueError("vectors, organs and phases must have the same length")

    if min_group < (2 if leave_one_out else 1):
        raise ValueError("min_group too small")

    if columns is None:
        indices = np.arange(x.shape[1])
        renormalise = True
    else:
        lookup = {str(name): i for i, name in enumerate(columns)}
        missing = [name for name in INTENSITY_COLUMNS if name not in lookup]
        if missing:
            raise ValueError(f"Missing intensity columns: {missing}")
        indices = np.array([lookup[name] for name in INTENSITY_COLUMNS], dtype=int)
        renormalise = False

    # Columns being corrected. Was x[indices][rows], which picks rows, not columns.
    sub = x[:, indices]
    offsets = np.zeros_like(sub)

    organ_rows = defaultdict(list)
    group_rows = defaultdict(list)
    for i, (organ, phase) in enumerate(zip(organs, phases)):
        organ_rows[organ].append(i)
        if phase is not None and phase != "":
            group_rows[(organ, phase)].append(i)

    for (organ, _), rows in group_rows.items():
        if len(rows) < min_group:
            continue

        rows = np.asarray(rows)
        organ_all = np.asarray(organ_rows[organ])
        group_sum = sub[rows].sum(axis=0)
        organ_sum = sub[organ_all].sum(axis=0)

        for i in rows:
            if leave_one_out:
                group_mean = (group_sum - sub[i]) / (len(rows) - 1)
                organ_mean = (organ_sum - sub[i]) / (len(organ_all) - 1)
            else:
                group_mean = group_sum / len(rows)
                organ_mean = organ_sum / len(organ_all)
            offsets[i] = group_mean - organ_mean

    x[:, indices] = sub - offsets

    if renormalise:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)

    return x
