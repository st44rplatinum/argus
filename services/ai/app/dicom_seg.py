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

from . import anatomy
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


def build_multi_segmentation(
    source_datasets,
    labelmap,
    sop_uids,
    names,
    *,
    algorithm_name: str = "TotalSegmentator",
    software_version: str = "0.1.0",
):
    """A SEG holding every structure a label map contains.

    `labelmap` is a 3D array whose values are the model's own label numbers,
    and `names` maps those numbers to structure names.

    Two things happen here that the single-segment path does not need:

    Labels are renumbered to 1..N over the structures actually *present*.
    highdicom requires consecutive segment numbers, and a model that can emit
    117 structures will emit a handful on any given scan — describing the
    other 110 would fill the viewer's segment list with empty rows.

    The type is LABELMAP rather than BINARY. A stacked-segments array would be
    one binary mask per structure, so a 191-slice scan with 30 structures
    becomes 30 times the frames; a label map stays one value per voxel. It is
    only valid because the segments are mutually exclusive, which they are —
    a voxel is liver or spleen, not both.
    """
    if labelmap.ndim != 3:
        raise ValueError(f"Expected a 3D label map, got {labelmap.shape}")

    if len(sop_uids) != labelmap.shape[0]:
        raise ValueError(
            "Number of source instances does not match label map depth: "
            f"{len(sop_uids)} != {labelmap.shape[0]}"
        )

    present = [int(v) for v in np.unique(labelmap) if v != 0]
    if not present:
        raise ValueError("label map is empty")

    renumbered = np.zeros(labelmap.shape, dtype=np.uint8)
    descriptions = []

    for segment_number, original in enumerate(present, start=1):
        renumbered[labelmap == original] = segment_number
        name = names.get(original, f"structure {original}")

        descriptions.append(
            hd.seg.SegmentDescription(
                segment_number=segment_number,
                segment_label=anatomy.label_for(name),
                segmented_property_category=codes.SCT.AnatomicalStructure,
                segmented_property_type=anatomy.concept_for(name),
                algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
                algorithm_identification=hd.AlgorithmIdentificationSequence(
                    name=algorithm_name,
                    family=codes.cid7162.ArtificialIntelligence,
                    version=software_version,
                ),
            )
        )

    by_uid = {str(dataset.SOPInstanceUID): dataset for dataset in source_datasets}
    missing = [uid for uid in sop_uids if uid not in by_uid]
    if missing:
        raise ValueError(
            f"{len(missing)} slice(s) named in sop_uids are absent from the "
            f"source datasets, first is {missing[0]}"
        )

    ordered = [by_uid[uid] for uid in sop_uids]

    return hd.seg.Segmentation(
        source_images=ordered,
        pixel_array=renumbered,
        segmentation_type=hd.seg.SegmentationTypeValues.LABELMAP,
        segment_descriptions=descriptions,
        series_instance_uid=generate_uid(),
        sop_instance_uid=generate_uid(),
        series_number=SERIES_NUMBER,
        instance_number=1,
        manufacturer="Argus",
        manufacturer_model_name=algorithm_name,
        software_versions=software_version,
        device_serial_number="ARGUS",
        series_description=f"Argus {algorithm_name}",
    )


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
