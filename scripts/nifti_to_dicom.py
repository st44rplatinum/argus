"""Write a NIfTI volume out as a DICOM CT series.

So a downloaded scan goes through the same path as everything else — PACS,
DICOMweb, the service, the viewer — rather than being special-cased.

SimpleITK does the coordinate work: it reads NIfTI's RAS and hands back LPS,
which is what DICOM wants. Doing that conversion by hand is where the
left-right flips come from, and a flipped abdomen looks entirely normal.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

# Signed storage with a zero intercept: stored values are Hounsfield units
# directly. Unsigned with a -1024 intercept cannot represent anything below
# -1024, and this scan reaches -1094 in air outside the patient.
INTERCEPT = 0


def read_nifti(path: Path) -> sitk.Image:
    """Read a NIfTI, squaring up direction cosines ITK rejects.

    Some TotalSegmentator cases have directions a few 1e-4 off orthogonal
    (float32 rounding) and no qform, and ITK refuses them outright. The
    nearest orthonormal matrix moves nothing by more than a fraction of a mm.
    """
    try:
        return sitk.ReadImage(str(path))
    except RuntimeError:
        pass

    import nibabel as nib

    image = nib.load(path)
    affine = image.affine.copy()
    zooms = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    u, _, vt = np.linalg.svd(affine[:3, :3] / zooms)
    affine[:3, :3] = (u @ vt) * zooms

    fixed = nib.Nifti1Image(np.asanyarray(image.dataobj), affine, image.header)
    fixed.set_sform(affine, code=2)
    fixed.set_qform(affine, code=1)

    with tempfile.TemporaryDirectory(prefix="argus-n2d-") as workdir:
        target = Path(workdir) / "fixed.nii.gz"
        nib.save(fixed, target)
        return sitk.ReadImage(str(target))


def write_series(
    image: sitk.Image,
    out_dir: Path,
    description: str,
    patient_id: str,
    study_date: str = "20260101",
) -> tuple[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # first row at the front, first column at the patient's right, as viewers
    # expect; a RAS NIfTI stored as-is shows upside down in a stack view
    image = sitk.DICOMOrient(image, "LPS")

    array = sitk.GetArrayFromImage(image).astype(np.float32)
    spacing = image.GetSpacing()
    origin = np.array(image.GetOrigin(), dtype=np.float64)
    direction = np.array(image.GetDirection(), dtype=np.float64).reshape(3, 3)

    axis_x = direction[:, 0]
    axis_y = direction[:, 1]
    axis_z = direction[:, 2]

    slices, rows, columns = array.shape

    study_uid = generate_uid()
    series_uid = generate_uid()
    frame_uid = generate_uid()

    stored = np.rint(array - INTERCEPT).astype(np.int16)

    for k in range(slices):
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

        ds.PatientName = patient_id
        ds.PatientID = patient_id
        ds.PatientBirthDate = ""
        ds.PatientSex = "O"

        ds.StudyDate = study_date
        ds.StudyTime = "120000"
        ds.StudyID = "1"
        ds.AccessionNumber = "EXAMPLE"
        ds.StudyDescription = description
        ds.SeriesDescription = f"AXIAL {spacing[2]:.1f}mm"
        ds.Modality = "CT"
        ds.SeriesNumber = 1
        ds.InstanceNumber = k + 1

        ds.ImageOrientationPatient = [*axis_x, *axis_y]
        ds.ImagePositionPatient = list(origin + k * spacing[2] * axis_z)
        ds.SliceThickness = float(spacing[2])
        ds.SpacingBetweenSlices = float(spacing[2])
        ds.PixelSpacing = [float(spacing[1]), float(spacing[0])]

        ds.Rows = rows
        ds.Columns = columns
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.RescaleIntercept = INTERCEPT
        ds.RescaleSlope = 1
        ds.RescaleType = "HU"
        ds.WindowCenter = 40
        ds.WindowWidth = 400

        ds.PixelData = stored[k].tobytes()
        ds.save_as(
            out_dir / f"slice_{k:04d}.dcm",
            enforce_file_format=True,
        )

    return series_uid, slices



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nifti", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--description", default="Example CT")
    parser.add_argument(
        "--patient-id",
        default="EXAMPLE001",
        help="DICOM PatientID and PatientName.",
    )
    parser.add_argument(
    "--study-date",
    default="20260101",
    help="DICOM StudyDate in YYYYMMDD format.",
    )
    args = parser.parse_args()

    image = read_nifti(args.nifti)
    series_uid, count = write_series(
        image,
        args.out,
        args.description,
        args.patient_id,
        study_date=args.study_date,
    )

    print(f"wrote {count} instances to {args.out}")
    print(f"SeriesInstanceUID {series_uid}")


if __name__ == "__main__":
    main()
