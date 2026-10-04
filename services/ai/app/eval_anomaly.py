"""Scoring for P8: voxel AUROC, and sensitivity at a false-alarm rate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_auc_score

from .anomaly import connected_blobs

# Every method is scored inside the liver minus this rim. The mask's outer
# voxels are part fat or air, so without it every method mostly flags the
# edge. No planted lesion is lost: all sit their radius + 2 mm inside.
RIM_MM = 3.0


def scoring_region(liver_mask: np.ndarray, spacing_zyx) -> np.ndarray:
    """The liver shrunk by RIM_MM. Pass this wherever a liver mask is asked for."""
    inside = ndimage.distance_transform_edt(liver_mask.astype(bool), sampling=spacing_zyx)
    return (inside > RIM_MM).astype(np.uint8)


@dataclass
class Lesion:
    mask: np.ndarray
    radius_mm: float
    contrast_hu: float

def voxel_auroc(
        score: np.ndarray,
        lesion_mask: np.ndarray,
        liver_mask: np.ndarray,
) -> float:
    valid = liver_mask.astype(bool)

    x = score[valid].astype(np.float64)
    y = lesion_mask[valid].astype(np.float64)


    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")

    return float(roc_auc_score(y,x))

def sensitivity_at_threshold(
    score_maps: list[np.ndarray],
    lesion_masks: list[np.ndarray],
    liver_masks: list[np.ndarray],
    threshold: float,
) -> tuple [float, float]: 
    """
    Returns:
        sensitivity, false_blobs_per_scan
    """
    found = 0 
    total_lesions = 0
    false_blobs = 0

    for score, lesions, liver in zip(
        score_maps, 
        lesion_masks,
        liver_masks,
    ):
        blobs = connected_blobs(
            score * liver,
            threshold,
        )

        labels, n = ndimage.label(lesions > 0, structure=np.ones((3, 3, 3)))

        # which lesion, if any, each blob's peak lands in (0 = none)
        hits = {int(labels[tuple(blob["peak_zyx"])]) for blob in blobs}
        total_lesions += n
        found += len(hits - {0})
        false_blobs += sum(1 for blob in blobs if labels[tuple(blob["peak_zyx"])] == 0)
    scans = max(len(score_maps), 1)

    sensitivity = (
        found / total_lesions
        if total_lesions > 0
        else float("nan")
    )

    return sensitivity, false_blobs / scans

def threshold_for_one_fp(
    score_maps: list[np.ndarray],
    lesion_masks: list[np.ndarray],
    liver_masks: list[np.ndarray],
) -> float:
    """
    Choose the highest threshold whose validation false-blob rate is
    approximately <= 1/scan, while favouring sensitivity.
    """
    # 1 false blob per scan sits far up the tail: the top 0.1% of a liver
    # is still thousands of voxels, so look from the 99th percentile up
    pooled = np.concatenate([s[l.astype(bool)] for s, l in zip(score_maps, liver_masks)])
    tail = 1.0 - np.logspace(-2, -6, 40)
    candidates = sorted(set(np.quantile(pooled, tail).tolist()))

    best = candidates[-1]
    best_sensitivity = -1.0

    for threshold in candidates:
        sensitivity, fp = sensitivity_at_threshold(
            score_maps,
            lesion_masks,
            liver_masks,
            threshold,
        )

        if fp <= 1.0 and sensitivity > best_sensitivity:
            best = threshold
            best_sensitivity = sensitivity

    return float(best)

def sanity_check(
    cheating_score: np.ndarray,
    random_score: np.ndarray,
    lesion_mask: np.ndarray,
    liver_mask: np.ndarray,
) -> None:
    cheat = voxel_auroc(
        cheating_score,
        lesion_mask,
        liver_mask,
    )

    random = voxel_auroc(
        random_score,
        lesion_mask,
        liver_mask,
    )

    assert cheat == 1.0, cheat
    assert 0.40 <= random <= 0.60, random