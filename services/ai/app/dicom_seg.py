import io

import highdicom as hd
import numpy as np
from pydicom.dataset import Dataset
from pydicom.uid import generate_uid


def build_segmentation(
    source_datasets,
    mask,
):
    if mask.ndim != 3:
        raise ValueError(
            f"Expected 3D mask, got {mask.shape}"
        )

    mask = mask.astype(np.uint8)

    if len(source_datasets) != mask.shape[0]:
        raise ValueError(
            "Number of source instances does not match "
            f"mask depth: {len(source_datasets)} != {mask.shape[0]}"
        )

    segment_description = hd.seg.SegmentDescription(
        segment_number=1,
        segment_label="Bone",
        segmented_property_category=hd.sr.CodedConcept(
            value="91723000",
            scheme_designator="SCT",
            meaning="Anatomical structure",
        ),
        segmented_property_type=hd.sr.CodedConcept(
            value="71341001",
            scheme_designator="SCT",
            meaning="Bone structure",
        ),
        algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
        algorithm_name="RadAI Threshold",
    )

    return hd.seg.Segmentation(
        source_images=source_datasets,
        pixel_array=mask,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=[
            segment_description
        ],
        series_instance_uid=generate_uid(),
        sop_instance_uid=generate_uid(),
        series_number=900,
        instance_number=1,
        manufacturer="RadViewer",
        manufacturer_model_name="Rad AI",
        software_versions="0.1.0",
        device_serial_number="RADAI",
    )


def segmentation_to_bytes(segmentation):
    buffer = io.BytesIO()

    segmentation.save_as(
        buffer,
        write_like_original=False,
    )

    return buffer.getvalue()
