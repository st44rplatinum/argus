"""P8: planted lesions, the rim, the scorer, and the scores."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from app.anomaly import align_liver, connected_blobs, score_own_liver
from app.eval_anomaly import scoring_region, sensitivity_at_threshold, voxel_auroc
from app.lesions import LesionRecipe, add_lesion

SPACING = (1.5, 1.5, 1.5)


def liver_box(shape=(40, 50, 50)):
    """A noisy block of liver at 60 HU."""
    rng = np.random.default_rng(1)
    image = rng.normal(60.0, 8.0, shape).astype(np.float32)
    liver = np.zeros(shape, bool)
    liver[5:-5, 5:-5, 5:-5] = True
    return image, liver


def recipe(**kw):
    base = dict(case="x", centre_zyx=(20, 25, 25), radius_mm=6.0, contrast_hu=-40.0, edge="hard", seed=3)
    return LesionRecipe(**{**base, **kw})


def test_planted_lesion_is_repeatable_and_inside_the_liver():
    image, liver = liver_box()
    a, mask_a = add_lesion(image, liver, recipe(), SPACING)
    b, mask_b = add_lesion(image, liver, recipe(), SPACING)
    assert np.array_equal(a, b) and np.array_equal(mask_a, mask_b)
    assert mask_a.any() and not (mask_a.astype(bool) & ~liver).any()
    assert a[mask_a > 0].mean() < image[mask_a > 0].mean() - 30
    assert np.array_equal(a[~liver], image[~liver])


def test_rim_takes_the_edge_and_keeps_the_inside():
    _, liver = liver_box()
    region = scoring_region(liver, SPACING).astype(bool)
    assert not region[5].any()  # the liver's first slice is all rim
    assert region[20, 25, 25]


def test_scorer_cheat_is_perfect_and_noise_is_chance():
    image, liver = liver_box()
    _, lesion = add_lesion(image, liver, recipe(), SPACING)
    assert voxel_auroc(lesion.astype(np.float32), lesion, liver) == 1.0
    noise = np.random.default_rng(0).random(lesion.shape)
    assert 0.4 < voxel_auroc(noise, lesion, liver) < 0.6
    found, fp = sensitivity_at_threshold([lesion.astype(np.float32)], [lesion], [liver.astype(np.uint8)], 0.5)
    assert (found, fp) == (1.0, 0.0)


def test_blob_peaks_are_zyx():
    score = np.zeros((10, 10, 10), np.float32)
    score[2:4, 5:7, 7:9] = 3.0
    score[3, 6, 8] = 9.0
    (blob,) = connected_blobs(score, 2.0)
    assert blob["peak_zyx"] == [3, 6, 8] and blob["voxels"] == 8


def test_own_liver_score_flags_a_planted_lesion():
    image, liver = liver_box()
    planted, lesion = add_lesion(image, liver, recipe(), SPACING)
    as_image = sitk.GetImageFromArray(planted)
    as_image.SetSpacing(SPACING)
    mask = sitk.GetImageFromArray(liver.astype(np.uint8))
    mask.SetSpacing(SPACING)
    score = score_own_liver(as_image, mask)
    assert np.median(score[lesion > 0]) > 3 * np.median(score[liver & (lesion == 0)])


def ellipsoid(radii_mm, size=60, spacing=2.0):
    z, y, x = (np.indices((size,) * 3) - size / 2) * spacing
    inside = (z / radii_mm[0]) ** 2 + (y / radii_mm[1]) ** 2 + (x / radii_mm[2]) ** 2 <= 1
    image = sitk.GetImageFromArray(inside.astype(np.uint8))
    image.SetSpacing((spacing,) * 3)
    return image


def test_shape_alignment_matches_livers_of_different_size():
    _, dice = align_liver(ellipsoid((30, 40, 45)), ellipsoid((36, 44, 44)))
    assert dice > 0.9
