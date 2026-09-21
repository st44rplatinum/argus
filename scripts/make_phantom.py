"""Generate a valid CT series, so P0 can be verified without waiting on TCIA.

This is a smoke-test fixture, not data. It exists because every downstream
piece — Orthanc ingest, DICOMweb query, OHIF rendering, SEG round-tripping —
needs *something* well-formed to fail against, and a phantom you can regenerate
in two seconds is a better first input than a 400 MB download behind a licence.

Real work uses real scans. See docs/ROADMAP.md for the datasets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

# Stored pixel values are HU + 1024, which keeps them unsigned. Anything
# reading this back must apply RescaleIntercept to get HU again.
INTERCEPT = -1024

ROWS = COLS = 256
SLICES = 64
PIXEL_SPACING = 1.5  # mm, in-plane
SLICE_THICKNESS = 2.0  # mm


def phantom_volume() -> np.ndarray:
    """A body-shaped ellipse with a bone ring and two focal lesions, in HU."""
    z, y, x = np.mgrid[0:SLICES, 0:ROWS, 0:COLS].astype(np.float32)
    cy, cx = ROWS / 2, COLS / 2

    body = ((y - cy) / 95.0) ** 2 + ((x - cx) / 115.0) ** 2
    volume = np.full((SLICES, ROWS, COLS), -1000.0, dtype=np.float32)
    volume[body <= 1.0] = 40.0  # soft tissue

    # A vertebral body, so windowing and bone presets have something to show.
    spine = ((y - (cy + 70)) / 16.0) ** 2 + ((x - cx) / 16.0) ** 2
    volume[spine <= 1.0] = 700.0

    # Two lesions at different depths, denser than their surroundings. These
    # are the targets for interactive segmentation (P4) and retrieval (P5).
    for (sz, sy, sx, radius, hu) in [
        (20, cy - 30, cx - 40, 9.0, 95.0),
        (44, cy + 10, cx + 50, 6.0, 110.0),
    ]:
        blob = ((z - sz) / (radius / SLICE_THICKNESS * PIXEL_SPACING)) ** 2 \
            + ((y - sy) / radius) ** 2 + ((x - sx) / radius) ** 2
        volume[blob <= 1.0] = hu

    # Mild noise, so a denoiser or anomaly model is not handed a trivially
    # piecewise-constant image.
    rng = np.random.default_rng(0)
    volume += rng.normal(0.0, 12.0, volume.shape).astype(np.float32)
    return volume


def write_series(out_dir: Path) -> tuple[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)

    study_uid = generate_uid()
    series_uid = generate_uid()
    frame_uid = generate_uid()

    volume = phantom_volume()
    stored = np.clip(volume - INTERCEPT, 0, 4095).astype(np.uint16)

    for index in range(SLICES):
        sop_uid = generate_uid()

        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = sop_uid
        meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)

        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = sop_uid
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame_uid

        ds.PatientName = "PHANTOM^SYNTHETIC"
        ds.PatientID = "PHANTOM001"
        ds.PatientBirthDate = ""
        ds.PatientSex = "O"

        ds.StudyDate = "20260101"
        ds.StudyTime = "120000"
        ds.StudyID = "1"
        ds.AccessionNumber = "PHANTOM"
        ds.StudyDescription = "Synthetic abdomen phantom"
        ds.SeriesDescription = "AXIAL 2.0mm"
        ds.Modality = "CT"
        ds.SeriesNumber = 1
        ds.InstanceNumber = index + 1

        # Geometry. Getting this wrong is the usual reason a volume renders
        # as a scrambled stack, so it is set explicitly rather than defaulted.
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [
            -COLS / 2 * PIXEL_SPACING,
            -ROWS / 2 * PIXEL_SPACING,
            index * SLICE_THICKNESS,
        ]
        ds.SliceThickness = SLICE_THICKNESS
        ds.SpacingBetweenSlices = SLICE_THICKNESS
        ds.SliceLocation = index * SLICE_THICKNESS
        ds.PixelSpacing = [PIXEL_SPACING, PIXEL_SPACING]

        ds.Rows = ROWS
        ds.Columns = COLS
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0  # unsigned, hence the intercept below
        ds.RescaleIntercept = INTERCEPT
        ds.RescaleSlope = 1
        ds.RescaleType = "HU"
        ds.WindowCenter = 40
        ds.WindowWidth = 400

        ds.PixelData = stored[index].tobytes()

        ds.save_as(out_dir / f"slice_{index:04d}.dcm", enforce_file_format=True)

    return series_uid, SLICES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw/phantom"),
        help="directory to write the series into",
    )
    args = parser.parse_args()

    series_uid, count = write_series(args.out)
    print(f"wrote {count} instances to {args.out}")
    print(f"SeriesInstanceUID {series_uid}")


if __name__ == "__main__":
    main()
