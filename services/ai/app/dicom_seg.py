"""Turn a mask into a DICOM SEG that lands in the right place.

A segmentation is only meaningful relative to the images it came from. The
source datasets carry that relationship — frame of reference, positions,
orientation — so highdicom derives the geometry from them rather than being
told. That is why this takes datasets and not just an array, and why their
order has to match the mask's slice order exactly.
"""

from __future__ import annotations

import io

import highdicom as hd
import numpy as np
from pydicom.sr.codedict import codes
from pydicom.uid import generate_uid

# 900 keeps derived series out of the way of acquired ones in a series list.
SERIES_NUMBER = 900


def build_segmentation(
    source_datasets,
    mask,
    sop_uids,
    *,
    segment_label: str = "Bone",
    algorithm_name: str = "Argus threshold",
    software_version: str = "0.1.0",
):
    """Build a SEG whose frames line up with the mask.

    `sop_uids` is the order the mask's slices are in — i.e. `Volume.sop_uids`.
    This argument is not a convenience, it is the whole correctness of the
    function. `retrieve_series` returns instances in whatever order the server
    felt like: measured against this Orthanc, all 64 of the phantom's slices
    came back in a different position than geometric order.

    highdicom pairs frame *i* of the mask with source image *i*. Hand it the
    server's order and every frame lands on the wrong slice — with the correct
    shape, the correct voxel count, and no error anywhere.
    """
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D mask, got {mask.shape}")

    mask = mask.astype(np.uint8)

    if len(sop_uids) != mask.shape[0]:
        raise ValueError(
            "Number of source instances does not match "
            f"mask depth: {len(sop_uids)} != {mask.shape[0]}"
        )

    by_uid = {str(dataset.SOPInstanceUID): dataset for dataset in source_datasets}
    missing = [uid for uid in sop_uids if uid not in by_uid]
    if missing:
        raise ValueError(
            f"{len(missing)} slice(s) named in sop_uids are absent from the "
            f"source datasets, first is {missing[0]}"
        )

    source_datasets = [by_uid[uid] for uid in sop_uids]

    segment_description = hd.seg.SegmentDescription(
        segment_number=1,
        segment_label=segment_label,
        # 91723000. Correct.
        segmented_property_category=codes.SCT.AnatomicalStructure,
        # 272673000 "Bone". The earlier 71341001 is bone structure of femur,
        # which would have been a confident, specific lie.
        segmented_property_type=codes.SCT.Bone,
        algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
        # When algorithm_type is AUTOMATIC the algorithm has to identify
        # itself, and it is a sequence rather than a name. CID 7162 has no
        # "thresholding"; histogram analysis is the nearest true statement,
        # and calling a threshold "Artificial Intelligence" would not be.
        algorithm_identification=hd.AlgorithmIdentificationSequence(
            name=algorithm_name,
            family=codes.cid7162.HistogramAnalysis,
            version=software_version,
        ),
    )

    return hd.seg.Segmentation(
        source_images=source_datasets,
        pixel_array=mask,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=[segment_description],
        series_instance_uid=generate_uid(),
        sop_instance_uid=generate_uid(),
        series_number=SERIES_NUMBER,
        instance_number=1,
        manufacturer="Argus",
        manufacturer_model_name="Argus threshold",
        software_versions=software_version,
        device_serial_number="ARGUS",
        series_description=f"Argus {segment_label}",
    )


def segmentation_to_bytes(segmentation) -> bytes:
    """Serialise to Part-10 bytes, ready to POST at Orthanc.

    `enforce_file_format=True` is what writes the 128-byte preamble, the DICM
    magic and the file meta group. pydicom 3 made the old `write_like_original`
    positional-only, so passing it by keyword lands in **kwargs and is silently
    ignored — the file still writes, without a preamble, and Orthanc rejects it
    with nothing to explain why.
    """
    buffer = io.BytesIO()
    segmentation.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()
