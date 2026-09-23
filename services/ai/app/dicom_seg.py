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
    """
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D mask, got {mask.shape}")

    mask = mask.astype(np.uint8)

    if len(sop_uids) != mask.shape[0]:
        raise ValueError(
            "Number of source instances does not match "
            f"mask depth: {len(sop_uids)} != {mask.shape[0]}"
        )

    by_uid = {
        str(dataset.SOPInstanceUID): dataset
        for dataset in source_datasets
    }

    missing = [
        uid
        for uid in sop_uids
        if uid not in by_uid
    ]

    if missing:
        raise ValueError(
            f"{len(missing)} slice(s) named in sop_uids are absent from "
            f"the source datasets, first is {missing[0]}"
        )

    # Critical: make source image order identical to mask slice order.
    source_datasets = [
        by_uid[uid]
        for uid in sop_uids
    ]

    segment_description = hd.seg.SegmentDescription(
        segment_number=1,
        segment_label=segment_label,
        segmented_property_category=(
            codes.SCT.AnatomicalStructure
        ),
        segmented_property_type=(
            codes.SCT.Bone
        ),
        algorithm_type=(
            hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC
        ),
        algorithm_identification=(
            hd.AlgorithmIdentificationSequence(
                name=algorithm_name,
                family=codes.cid7162.HistogramAnalysis,
                version=software_version,
            )
        ),
    )

    return hd.seg.Segmentation(
        source_images=source_datasets,
        pixel_array=mask,
        segmentation_type=(
            hd.seg.SegmentationTypeValues.BINARY
        ),
        segment_descriptions=[
            segment_description
        ],
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


def store_segmentation(client, segmentation) -> str:
    """Store a DICOM SEG and return its SeriesInstanceUID.

    STOW-RS can return HTTP success while reporting individual instance
    failures in FailedSOPSequence, so the response body must be checked.
    """
    response = client.store_instances([segmentation])

    failed = getattr(
        response,
        "FailedSOPSequence",
        None,
    )

    if failed:
        failures = []

        for item in failed:
            sop_uid = getattr(
                item,
                "ReferencedSOPInstanceUID",
                "<unknown>",
            )

            reason = getattr(
                item,
                "FailureReason",
                "<unknown>",
            )

            failures.append(
                f"SOPInstanceUID={sop_uid}, "
                f"FailureReason={reason}"
            )

        raise RuntimeError(
            "Orthanc rejected the DICOM SEG: "
            + "; ".join(failures)
        )

    referenced = getattr(
        response,
        "ReferencedSOPSequence",
        None,
    )

    if not referenced:
        raise RuntimeError(
            "STOW-RS returned no ReferencedSOPSequence; "
            "the DICOM SEG cannot be considered stored."
        )

    return str(segmentation.SeriesInstanceUID)


def segmentation_to_bytes(segmentation) -> bytes:
    """Serialise to Part-10 bytes for debugging or disk output.

    `enforce_file_format=True` writes the Part-10 preamble, DICM magic,
    and file-meta information required when saving the SEG as a file.
    """
    buffer = io.BytesIO()

    segmentation.save_as(
        buffer,
        enforce_file_format=True,
    )

    return buffer.getvalue()
