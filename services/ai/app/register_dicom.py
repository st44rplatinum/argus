from __future__ import annotations

import copy

import numpy as np
import SimpleITK as sitk
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

# One STOW request per batch rather than ~300 slices in one body.
BATCH = 50


def store_registered_series(
    client,
    sources,
    sop_uids,
    image: sitk.Image,
    description: str,
):
    """Clone the current series and replace only its pixel data and UIDs.

    Geometry stays exactly that of the current series, including
    FrameOfReferenceUID. `sop_uids` is the current Volume's slice order,
    which is the order of the image's slices; the server's order is not.
    """
    if len(sop_uids) != image.GetSize()[2]:
        raise ValueError(
            f"registered image has {image.GetSize()[2]} slices but "
            f"current series has {len(sop_uids)}"
        )

    by_uid = {str(ds.SOPInstanceUID): ds for ds in sources}
    missing = [uid for uid in sop_uids if uid not in by_uid]
    if missing:
        raise ValueError(f"{len(missing)} current slices missing from the server")

    array = sitk.GetArrayFromImage(image)
    stored = np.rint(array).astype(np.int16)

    series_uid = generate_uid()
    instances = []

    for index, uid in enumerate(sop_uids):
        ds = copy.deepcopy(by_uid[uid])

        sop_uid = generate_uid()
        ds.SOPInstanceUID = sop_uid
        ds.SeriesInstanceUID = series_uid
        ds.SeriesNumber = int(getattr(ds, "SeriesNumber", 1)) + 100
        ds.SeriesDescription = description

        # transfer syntax lives in file meta, not the dataset
        if hasattr(ds, "file_meta"):
            ds.file_meta.MediaStorageSOPInstanceUID = sop_uid
            ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds.Rows, ds.Columns = stored.shape[1], stored.shape[2]
        ds.PixelRepresentation = 1
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.RescaleSlope = 1
        ds.RescaleIntercept = 0
        ds.RescaleType = "HU"
        ds.PixelData = stored[index].tobytes()

        # Deliberately unchanged: StudyInstanceUID, FrameOfReferenceUID,
        # ImagePositionPatient, ImageOrientationPatient, PixelSpacing.
        instances.append(ds)

    for start in range(0, len(instances), BATCH):
        client.store_instances(instances[start : start + BATCH])

    return series_uid
