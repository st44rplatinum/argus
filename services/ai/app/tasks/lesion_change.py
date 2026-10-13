"""Measure a lesion in a prior and the current scan, matched through a registration (P7)."""

from .registration import REGISTRATIONS


SNAP_MM = 4.0


def _snap(roi, seed):
    """Move the seed up to SNAP_MM to where the lesion is clearest.

    A point near a lesion's edge reads half liver, and the half-way cut then
    has nothing to cut between. So step to the voxel that differs most from
    its surroundings, the same way the seed does (darker or brighter).
    """
    import numpy as np
    import SimpleITK as sitk

    values = sitk.GetArrayFromImage(sitk.DiscreteGaussian(roi, variance=1.0))
    spacing = np.asarray(roi.GetSpacing())[::-1]
    x, y, z = seed
    difference = values - np.median(values)
    # a seed on plain liver hasn't picked a side yet; then either will do
    sign = np.sign(difference[z, y, x]) if abs(difference[z, y, x]) >= 5.0 else 0.0
    score = sign * difference if sign else np.abs(difference)

    zz, yy, xx = np.indices(values.shape)
    near = (((zz - z) * spacing[0]) ** 2 + ((yy - y) * spacing[1]) ** 2
            + ((xx - x) * spacing[2]) ** 2) <= SNAP_MM**2
    best = np.unravel_index(np.argmax(np.where(near, score, -np.inf)), values.shape)
    return tuple(int(v) for v in best[::-1])


def measure_lesion(image, seed_point, radius_mm=20.0):
    """Grow a lesion from a point, within a box around it so it cannot leak far."""
    import numpy as np
    import SimpleITK as sitk

    from ..recist import axial_axes

    index = np.asarray(image.TransformPhysicalPointToIndex(tuple(map(float, seed_point))))
    size = np.asarray(image.GetSize())
    if np.any(index < 0) or np.any(index >= size):
        raise ValueError("point is outside this scan")

    spacing = np.asarray(image.GetSpacing())
    radius = np.ceil(radius_mm / spacing).astype(int)
    lower = np.maximum(index - radius, 0)
    upper = np.minimum(index + radius + 1, size)

    roi = sitk.RegionOfInterest(image, [int(v) for v in upper - lower], [int(v) for v in lower])
    seed = _snap(roi, tuple(int(v) for v in index - lower))
    values = sitk.GetArrayFromImage(roi)

    # Cut halfway between the lesion (around the seed) and what surrounds it
    # (the box's median). A blurred edge then counts half in, half out,
    # however blurred, so two scans of different sharpness still agree.
    z, y, x = seed[::-1]
    inside = float(np.median(values[max(z - 1, 0) : z + 2, max(y - 1, 0) : y + 2, max(x - 1, 0) : x + 2]))

    # air or gas, not tissue: nothing to measure
    if inside < -500:
        raise ValueError(f"that point is air or gas ({inside:.0f} HU), place the Probe on the lesion")

    around = float(np.median(values))
    half = (inside + around) / 2.0
    span = abs(around - inside)
    lower_hu, upper_hu = (inside - span, half) if inside < around else (half, inside + span)

    grown = sitk.ConnectedThreshold(roi, seedList=[seed], lower=lower_hu, upper=upper_hu, replaceValue=1)

    # sitk arrays are [z, y, x]
    mask = sitk.GetArrayFromImage(grown) > 0
    axes = axial_axes(mask, spacing[::-1])

    return {
        "volume_mm3": float(mask.sum() * np.prod(spacing)),
        "long_axis_mm": axes["long_mm"],
        "short_axis_mm": axes["short_mm"],
        # kept for older results and the panel's first column
        "diameter_mm": axes["long_mm"],
    }


def _change(before, after):
    return (after - before) / before * 100.0 if before > 0 else None


def run(request, progress):
    import SimpleITK as sitk

    from ..dicomweb import client_for, load_series
    from ..nifti import volume_to_sitk

    transform_path = REGISTRATIONS / f"{request['registration_key']}.h5"
    if not transform_path.is_file():
        raise ValueError("no registration for this pair yet, run Register first")
    transform = sitk.ReadTransform(str(transform_path))

    client = client_for(request["dicomweb_url"])

    progress("fetching")
    current = volume_to_sitk(load_series(client, request["study_uid"], request["series_uid"]))
    prior = volume_to_sitk(load_series(client, request["prior_study_uid"], request["prior_series_uid"]))

    # registration maps current points to prior points
    current_point = tuple(map(float, request["point"]))
    prior_point = transform.TransformPoint(current_point)

    progress("measuring")
    now = measure_lesion(current, current_point)
    before = measure_lesion(prior, prior_point)

    # The wrong way, kept as a control: measured in the warped prior, the
    # deformable step has already absorbed some of the growth.
    warped = sitk.Resample(prior, current, transform, sitk.sitkLinear, -1024.0, sitk.sitkFloat32)
    warped_before = measure_lesion(warped, current_point)

    return {
        "point_current": list(current_point),
        "point_prior": list(prior_point),
        "current": now,
        "prior": before,
        "change_percent": _change(before["volume_mm3"], now["volume_mm3"]),
        "warped_prior": warped_before,
        "warped_change_percent": _change(warped_before["volume_mm3"], now["volume_mm3"]),
    }
