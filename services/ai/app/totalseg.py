"""TotalSegmentator behind the same interface as the threshold.

The threshold stays. It runs in a second with no GPU and no weights, which is
how you tell a plumbing problem from a model problem — and on a machine where
this model takes minutes on CPU, that distinction is worth a second of work.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .config import ARGUS_DEVICE, ARGUS_THREADS
from .nifti import volume_to_sitk

TASK = "total"


def structure_names() -> dict[int, str]:
    """Label number to structure name, from the model's own map."""
    from totalsegmentator.map_to_binary import class_map

    return dict(class_map[TASK])


def run(
    volume,
    fast: bool = True,
    force_split: bool = False,
    statistics: bool = False,
    stats_aggregation: str = "median",
    statistics_exclude_masks_at_border: bool = False,
):
    """Segment a volume.

    Returns the label map by default.

    When ``statistics=True``, returns:

        (labelmap, stats)

    where ``stats`` is the parsed TotalSegmentator statistics.json.
    """
    from totalsegmentator.python_api import totalsegmentator

    image = volume_to_sitk(volume)

    with tempfile.TemporaryDirectory(prefix="argus-ts-") as workdir:
        work = Path(workdir)
        source = work / "input.nii.gz"
        target = work / "labels.nii.gz"

        sitk.WriteImage(image, str(source))

        totalsegmentator(
            str(source),
            str(target),
            task=TASK,
            # One file of labels rather than 117 binary masks on disk.
            ml=True,
            fast=fast,
            device=ARGUS_DEVICE,
            force_split=force_split,
            nr_thr_resamp=ARGUS_THREADS,
            nr_thr_saving=ARGUS_THREADS,
            quiet=True,
            statistics=statistics,
            stats_aggregation=stats_aggregation,
            statistics_exclude_masks_at_border=(
                statistics_exclude_masks_at_border
            ),
        )

        labels = sitk.ReadImage(str(target))
        labelmap = sitk.GetArrayFromImage(labels)

        stats = None

        if statistics:
            stats_files = list(work.rglob("statistics.json"))

            if not stats_files:
                raise RuntimeError(
                    "TotalSegmentator was asked for statistics but no "
                    f"statistics.json was produced under {work}"
                )

            if len(stats_files) > 1:
                raise RuntimeError(
                    "Multiple statistics.json files found: "
                    + ", ".join(map(str, stats_files))
                )

            with stats_files[0].open() as f:
                stats = json.load(f)

    # The model resamples internally and puts the result back on the input
    # grid, so this should already hold. Assert it rather than assume it.
    if labelmap.shape != volume.array.shape:
        raise ValueError(
            f"label map {labelmap.shape} does not match volume "
            f"{volume.array.shape}"
        )

    labelmap = labelmap.astype(np.uint16)

    if statistics:
        return labelmap, stats

    return labelmap
