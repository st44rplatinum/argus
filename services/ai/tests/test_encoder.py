"""Encoder preprocessing helpers. No model weights needed."""

from __future__ import annotations

import numpy as np

from app.encoder import GRID, crop_fov, patch_weights, select_slices


def test_select_slices_spans_the_organ():
    mask = np.zeros((4, 4, 50), dtype=bool)
    mask[:, :, 10:40] = True

    picks = select_slices(mask, count=5)

    assert len(picks) == 5
    assert picks[0] == 10 and picks[-1] == 39


def test_select_slices_takes_all_when_few():
    mask = np.zeros((4, 4, 20), dtype=bool)
    mask[:, :, [3, 7]] = True

    assert list(select_slices(mask, count=8)) == [3, 7]


def test_crop_pads_outside_the_image():
    image = np.ones((10, 10), dtype=np.float32)
    out = crop_fov(image, centre=(0, 0), half=(4, 4), fill=-1.0)

    assert out.shape == (8, 8)
    assert out[0, 0] == -1.0     # outside
    assert out[-1, -1] == 1.0    # inside


def test_crop_keeps_the_centre_in_the_middle():
    image = np.zeros((20, 20), dtype=np.float32)
    image[12, 7] = 5.0
    out = crop_fov(image, centre=(12, 7), half=(3, 3), fill=0.0)

    assert out[3, 3] == 5.0


def test_patch_weights_cover_the_grid():
    mask = np.zeros((64, 64), dtype=bool)
    mask[:32, :] = True          # top half

    w = patch_weights(mask)

    assert w.shape == (GRID * GRID,)
    assert np.isclose(w.mean(), 0.5, atol=0.02)
    assert w.max() == 1.0 and w.min() == 0.0
