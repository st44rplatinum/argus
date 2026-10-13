"""Long and short axis on shapes whose answers are known."""

from __future__ import annotations

import numpy as np
import pytest

from app.recist import axial_axes


def ellipsoid(radii_zyx, spacing=(1.0, 1.0, 1.0), turn_degrees=0.0, shape=(40, 80, 80)):
    """An ellipsoid in mm, optionally turned within the axial plane."""
    z, y, x = (np.indices(shape) - np.array(shape)[:, None, None, None] / 2) * np.array(spacing)[:, None, None, None]
    a = np.radians(turn_degrees)
    u, v = x * np.cos(a) + y * np.sin(a), -x * np.sin(a) + y * np.cos(a)
    return (z / radii_zyx[0]) ** 2 + (v / radii_zyx[1]) ** 2 + (u / radii_zyx[2]) ** 2 <= 1


def test_a_sphere_is_as_wide_as_it_is_long():
    axes = axial_axes(ellipsoid((10, 10, 10)), (1, 1, 1))
    assert axes["long_mm"] == pytest.approx(20, abs=1.0)
    assert axes["short_mm"] == pytest.approx(20, abs=1.0)


@pytest.mark.parametrize("turn", [0.0, 30.0])
def test_an_ellipse_keeps_its_axes_when_turned(turn):
    # 30 x 16 mm in the axial plane, whichever way it lies
    axes = axial_axes(ellipsoid((8, 8, 15), turn_degrees=turn), (1, 1, 1))
    assert axes["long_mm"] == pytest.approx(30, abs=1.5)
    assert axes["short_mm"] == pytest.approx(16, abs=1.5)


def test_spacing_is_respected():
    # the same 30 x 16 mm shape on 0.7 mm pixels, 2 mm slices
    mask = ellipsoid((8, 8, 15), spacing=(2.0, 0.7, 0.7), shape=(20, 100, 100))
    axes = axial_axes(mask, (2.0, 0.7, 0.7))
    assert axes["long_mm"] == pytest.approx(30, abs=1.0)
    assert axes["short_mm"] == pytest.approx(16, abs=1.0)


def test_the_widest_slice_is_chosen():
    mask = ellipsoid((10, 10, 10))
    assert axial_axes(mask, (1, 1, 1))["slice"] == pytest.approx(20, abs=1)


def test_nothing_to_measure():
    assert axial_axes(np.zeros((5, 5, 5), bool), (1, 1, 1))["long_mm"] == 0.0
