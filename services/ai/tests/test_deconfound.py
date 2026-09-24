"""Phase centring removes the phase shift and keeps the organ."""

from __future__ import annotations

import numpy as np
import pytest

from app.deconfound import INTENSITY_COLUMNS, center_by_group


def two_organs_two_phases(n=10, shift=100.0):
    """1-D values: liver ~60, aorta ~200, arterial phase adds `shift`."""
    rng = np.random.default_rng(0)
    organs, phases, values = [], [], []
    for organ, base in (("liver", 60.0), ("aorta", 200.0)):
        for phase, add in (("native", 0.0), ("arterial", shift)):
            for _ in range(n):
                organs.append(organ)
                phases.append(phase)
                values.append(base + add + rng.normal(0, 1))
    return np.array(values)[:, None], np.array(organs), np.array(phases, dtype=object)


def test_organ_mean_survives_and_phase_gap_closes():
    x, organs, phases = two_organs_two_phases()
    names = list(INTENSITY_COLUMNS)
    wide = np.repeat(x, len(names), axis=1)

    out = center_by_group(wide, organs, phases, columns=names, min_group=5, leave_one_out=False)[:, 0]

    liver = organs == "liver"
    aorta = organs == "aorta"
    native = phases == "native"

    # organ identity kept: livers and aortas still far apart
    assert out[aorta].mean() - out[liver].mean() == pytest.approx(140.0, abs=2.0)
    # phase gone: arterial and native livers now sit together
    assert abs(out[liver & native].mean() - out[liver & ~native].mean()) < 2.0


def test_shape_columns_are_untouched():
    names = list(INTENSITY_COLUMNS) + ["volume_ml"]
    x, organs, phases = two_organs_two_phases()
    wide = np.hstack([np.repeat(x, len(INTENSITY_COLUMNS), axis=1), np.full((len(x), 1), 1500.0)])

    out = center_by_group(wide, organs, phases, columns=names, min_group=5)

    assert np.array_equal(out[:, -1], wide[:, -1])


def test_small_groups_and_unknown_phase_are_left_alone():
    x = np.array([[1.0], [2.0], [3.0]])
    organs = np.array(["liver"] * 3)
    phases = np.array(["arterial", None, "native"], dtype=object)

    out = center_by_group(np.repeat(x, 7, axis=1), organs, phases,
                          columns=list(INTENSITY_COLUMNS), min_group=5)

    assert np.array_equal(out, np.repeat(x, 7, axis=1))


def test_leave_one_out_differs_in_a_small_group():
    x, organs, phases = two_organs_two_phases(n=5)
    names = list(INTENSITY_COLUMNS)
    wide = np.repeat(x, len(names), axis=1)

    loo = center_by_group(wide, organs, phases, columns=names, min_group=5, leave_one_out=True)
    full = center_by_group(wide, organs, phases, columns=names, min_group=5, leave_one_out=False)

    assert not np.allclose(loo, full)


def test_whole_vector_mode_returns_unit_rows():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(20, 8))
    organs = np.array(["liver"] * 20)
    phases = np.array(["a"] * 10 + ["b"] * 10, dtype=object)

    out = center_by_group(x, organs, phases, columns=None, min_group=5)

    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
