"""TotalSegmentator behind the same interface as the threshold.

The threshold stays. It runs in a second with no GPU and no weights, which is
how you tell a plumbing problem from a model problem — and on a machine where
this model takes minutes on CPU, that distinction is worth a second of work.
"""

from __future__ import annotations

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


def run(volume, fast: bool = True, force_split: bool = False) -> np.ndarray:
    """Segment a volume, returning a label map the same shape as its array.

    `fast` is the 3 mm model. The full-resolution one wants 7 GB of VRAM and
    this machine's card has 4, so fast is the default rather than the option.
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
        )

        labels = sitk.ReadImage(str(target))
        labelmap = sitk.GetArrayFromImage(labels)

    # The model resamples internally and puts the result back on the input
    # grid, so this should already hold. Assert it rather than assume it: a
    # label map of the wrong shape would be caught later by the SEG builder,
    # but with a message about source instances rather than about resampling.
    if labelmap.shape != volume.array.shape:
        raise ValueError(
            f"label map {labelmap.shape} does not match volume "
            f"{volume.array.shape}"
        )

    return labelmap.astype(np.uint16)
