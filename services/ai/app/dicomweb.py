"""Pull a DICOM series out of the PACS and assemble it into a volume.

Every later project starts here: segmentation, retrieval, registration and
anomaly detection all need a correctly oriented array of Hounsfield units plus
enough geometry to write a mask back into the same frame of reference.

Two things are easy to get wrong and are handled explicitly below, because
both produce output that looks plausible and is silently unusable:

1.  **Slice order.** Instances are sorted by projecting ImagePositionPatient
    onto the slice normal, not by InstanceNumber or SliceLocation. Numbering
    is a display convention that can be reversed, duplicated across
    reconstructions, or simply absent; the normal is geometry.

2.  **Slice spacing.** Taken from the spacing between successive positions,
    not from SliceThickness. Those differ whenever slices overlap or a gap
    exists, and a mask built on the wrong spacing lands in the wrong place.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from dicomweb_client.api import DICOMwebClient
from pydicom.dataset import Dataset

# Positions closer together than this are treated as the same slice location.
POSITION_EPSILON_MM = 1e-3

# Spacing is expected to be regular. Beyond this, the series is either a
# gantry-tilted acquisition or several reconstructions mixed together, and
# resampling it as a uniform grid would be wrong.
SPACING_TOLERANCE_MM = 1e-2


@dataclass(frozen=True)
class Volume:
    """A series as a dense array, with what is needed to map back to patient space."""

    #: (slices, rows, columns), Hounsfield units for CT.
    array: np.ndarray
    #: (z, y, x) millimetres.
    spacing: tuple[float, float, float]
    #: Patient-space coordinate of the first voxel.
    origin: tuple[float, float, float]
    #: Row and column direction cosines, as given by ImageOrientationPatient.
    orientation: tuple[float, ...]

    study_uid: str
    series_uid: str
    #: SOP Instance UIDs in the same order as `array`, so a derived object can
    #: reference the exact slices it was computed from.
    sop_uids: tuple[str, ...]
    modality: str

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.array.shape


def slice_normal(orientation: list[float]) -> np.ndarray:
    """The through-plane direction, from the in-plane direction cosines."""
    row = np.array(orientation[0:3], dtype=np.float64)
    column = np.array(orientation[3:6], dtype=np.float64)
    return np.cross(row, column)


def client_for(base_url: str = "http://localhost:8042/dicom-web") -> DICOMwebClient:
    return DICOMwebClient(url=base_url)


def _rescaled(dataset: Dataset) -> np.ndarray:
    """Stored pixel values converted to real units, usually HU."""
    pixels = dataset.pixel_array.astype(np.float32)
    slope = float(getattr(dataset, "RescaleSlope", 1) or 1)
    intercept = float(getattr(dataset, "RescaleIntercept", 0) or 0)
    return pixels * slope + intercept


def load_series(
    client: DICOMwebClient,
    study_uid: str,
    series_uid: str,
) -> Volume:
    """Fetch one series and return it as a geometrically coherent volume."""
    instances = client.retrieve_series(
        study_instance_uid=study_uid,
        series_instance_uid=series_uid,
    )
    return assemble(instances, study_uid=study_uid, series_uid=series_uid)


def assemble(
    instances: list[Dataset],
    study_uid: str,
    series_uid: str,
) -> Volume:
    """Order and stack already-fetched instances.

    Split out from the fetch so the geometry can be tested against
    hand-built datasets without a PACS — this is where the bugs live.
    """
    if not instances:
        raise ValueError(f"series {series_uid} returned no instances")

    orientation = [float(v) for v in instances[0].ImageOrientationPatient]
    normal = slice_normal(orientation)

    # Sort by where each slice sits along the normal. See the module docstring.
    def depth(dataset: Dataset) -> float:
        position = np.array(
            [float(v) for v in dataset.ImagePositionPatient], dtype=np.float64
        )
        return float(np.dot(position, normal))

    ordered = sorted(instances, key=depth)

    # A series containing two reconstructions of the same anatomy will have
    # pairs of slices at identical depths. Stacking them yields a volume with
    # the right shape and interleaved content, which is worse than an error.
    depths = np.array([depth(d) for d in ordered])
    duplicates = np.sum(np.diff(depths) < POSITION_EPSILON_MM)
    if duplicates:
        raise ValueError(
            f"series {series_uid} has {duplicates} slices at duplicate positions — "
            "probably more than one reconstruction under a single series UID"
        )

    if len(ordered) > 1:
        gaps = np.diff(depths)
        if np.ptp(gaps) > SPACING_TOLERANCE_MM:
            raise ValueError(
                f"series {series_uid} has irregular slice spacing "
                f"({gaps.min():.4f}-{gaps.max():.4f} mm); "
                "resampling is required before it can be treated as a volume"
            )
        z_spacing = float(gaps.mean())
    else:
        z_spacing = float(getattr(ordered[0], "SliceThickness", 1.0) or 1.0)

    row_spacing, column_spacing = (float(v) for v in ordered[0].PixelSpacing)

    array = np.stack([_rescaled(d) for d in ordered], axis=0)

    return Volume(
        array=array,
        spacing=(z_spacing, row_spacing, column_spacing),
        origin=tuple(float(v) for v in ordered[0].ImagePositionPatient),
        orientation=tuple(orientation),
        study_uid=study_uid,
        series_uid=series_uid,
        sop_uids=tuple(str(d.SOPInstanceUID) for d in ordered),
        modality=str(getattr(ordered[0], "Modality", "")),
    )
