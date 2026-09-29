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


def _column_indices(
    vectors: np.ndarray,
    columns,
) -> np.ndarray:
    if columns is None:
        return np.arange(vectors.shape[1])

    lookup = {
        str(name): i
        for i, name in enumerate(columns)
    }

    missing = [
        name
        for name in INTENSITY_COLUMNS
        if name not in lookup
    ]

    if missing:
        raise ValueError(
            f"Missing intensity columns: {missing}"
        )

    return np.array(
        [
            lookup[name]
            for name in INTENSITY_COLUMNS
        ],
        dtype=int,
    )


def fit_phase_offsets(
    vectors,
    organs,
    phases,
    columns=None,
    min_group=5,
):
    """
    Fit one phase offset per (organ, phase) group.

    The offset is:

        mean(group) - mean(organ)

    Groups smaller than min_group are not fitted.

    No leave-one-out is used here. These are fitted parameters that will
    subsequently be applied to new/query rows.
    """
    x = np.asarray(vectors, dtype=np.float64)

    organs = np.asarray(organs, dtype=object)
    phases = np.asarray(phases, dtype=object)

    if x.ndim != 2:
        raise ValueError(
            f"vectors must be 2-D, got {x.shape}"
        )

    if len(x) != len(organs) or len(x) != len(phases):
        raise ValueError(
            "vectors, organs and phases must have the same length"
        )

    if min_group < 1:
        raise ValueError("min_group must be >= 1")

    indices = _column_indices(x, columns)
    sub = x[:, indices]

    organ_rows = defaultdict(list)
    group_rows = defaultdict(list)

    for i, (organ, phase) in enumerate(
        zip(organs, phases)
    ):
        organ = str(organ)
        organ_rows[organ].append(i)

        if phase is not None and phase != "":
            group_rows[
                (organ, str(phase))
            ].append(i)

    offsets = {}

    for (organ, phase), rows in group_rows.items():
        if len(rows) < min_group:
            continue

        rows = np.asarray(rows, dtype=int)
        organ_all = np.asarray(
            organ_rows[organ],
            dtype=int,
        )

        group_mean = sub[rows].mean(axis=0)
        organ_mean = sub[organ_all].mean(axis=0)

        offsets[(organ, phase)] = (
            group_mean - organ_mean
        ).astype(np.float32)

    return offsets


def apply_phase_offsets(
    vectors,
    organs,
    phases,
    offsets,
    columns=None,
):
    """
    Apply previously fitted phase offsets.

    When columns=None the entire vector is corrected and each row is
    re-normalised to unit length. When columns is supplied, only the
    intensity features are corrected and no re-normalisation is performed.
    """
    x = np.asarray(vectors, dtype=np.float64).copy()

    organs = list(organs)
    phases = list(phases)

    if len(x) != len(organs) or len(x) != len(phases):
        raise ValueError(
            "vectors, organs and phases must have the same length"
        )

    indices = _column_indices(x, columns)

    sub = x[:, indices]

    for i, (organ, phase) in enumerate(
        zip(organs, phases)
    ):
        if phase is None or phase == "":
            continue

        offset = offsets.get(
            (str(organ), str(phase))
        )

        if offset is not None:
            sub[i] -= offset

    x[:, indices] = sub

    if columns is None:
        norms = np.linalg.norm(
            x,
            axis=1,
            keepdims=True,
        )

        x = np.divide(
            x,
            norms,
            out=np.zeros_like(x),
            where=norms > 0,
        )

    return x.astype(np.float32)


def center_by_group(
    vectors,
    organs,
    phases,
    columns=None,
    min_group=5,
    leave_one_out=True,
):
    """
    Fit phase offsets and apply them to the same rows.

    leave_one_out: a row is not part of the means it is corrected by, so
    in-sample results are not flattered. Used by compare_encoders.py.
    """
    if not leave_one_out:
        offsets = fit_phase_offsets(
            vectors,
            organs,
            phases,
            columns=columns,
            min_group=min_group,
        )

        return apply_phase_offsets(
            vectors,
            organs,
            phases,
            offsets,
            columns=columns,
        )

    if min_group < 2:
        raise ValueError("min_group too small for leave_one_out")

    x = np.asarray(vectors, dtype=np.float64).copy()

    if x.ndim != 2:
        raise ValueError(f"vectors must be 2-D, got {x.shape}")

    if len(x) != len(organs) or len(x) != len(phases):
        raise ValueError(
            "vectors, organs and phases must have the same length"
        )

    indices = _column_indices(x, columns)
    sub = x[:, indices]

    organ_rows = defaultdict(list)
    group_rows = defaultdict(list)

    for i, (organ, phase) in enumerate(zip(organs, phases)):
        organ_rows[str(organ)].append(i)

        if phase is not None and phase != "":
            group_rows[(str(organ), str(phase))].append(i)

    offsets = np.zeros_like(sub)

    for (organ, _), rows in group_rows.items():
        if len(rows) < min_group:
            continue

        organ_all = organ_rows[organ]
        group_sum = sub[rows].sum(axis=0)
        organ_sum = sub[organ_all].sum(axis=0)

        # each row's own offset, from means without it
        for i in rows:
            group_mean = (group_sum - sub[i]) / (len(rows) - 1)
            organ_mean = (organ_sum - sub[i]) / (len(organ_all) - 1)
            offsets[i] = group_mean - organ_mean

    x[:, indices] = sub - offsets

    if columns is None:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)

    return x
