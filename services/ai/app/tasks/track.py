"""Follow several lesions through a patient's earlier scans.

A lesion is a point in the current scan. Registration carries that point into
each earlier scan, so lesion 2 here is lesion 2 there, and it is measured in
each scan's own pixels.
"""

from .lesion_change import measure_lesion
from .registration import transform_for


def _row(date, measured):
    return {
        "date": date,
        "seen": True,
        "volume_mm3": measured["volume_mm3"],
        "long_axis_mm": measured["long_axis_mm"],
        "short_axis_mm": measured["short_axis_mm"],
    }


def run(request, progress):
    from ..dicomweb import client_for, load_series
    from ..nifti import volume_to_sitk

    client = client_for(request["dicomweb_url"])
    priors = sorted(request["priors"], key=lambda p: p["date"])

    progress("fetching")
    current = volume_to_sitk(load_series(client, request["study_uid"], request["series_uid"]))

    # the current scan first: it says what kind of lesion each one is
    lesions = []
    for number, point in enumerate(request["points"], start=1):
        point = tuple(map(float, point))
        now = measure_lesion(current, point)
        lesions.append({"id": number, "point": list(point), "contrast": now["contrast_hu"], "now": now, "rows": []})

    for index, prior in enumerate(priors, start=1):
        progress(f"scan {index} of {len(priors)}")
        image = volume_to_sitk(load_series(client, prior["study_uid"], prior["series_uid"]))
        # current point -> where that tissue was in the earlier scan
        transform = transform_for(prior["registration_key"], current, image)

        for lesion in lesions:
            then = transform.TransformPoint(tuple(lesion["point"]))
            try:
                lesion["rows"].append({**_row(prior["date"], measure_lesion(image, then, like=lesion["contrast"])),
                                       "point": list(then)})
            except ValueError as exc:
                # NotSeen: new, or too faint to tell from liver. Otherwise the
                # point fell outside that scan, or on air.
                lesion["rows"].append({"date": prior["date"], "seen": False, "reason": str(exc), "point": list(then)})

    date = request.get("study_date") or "now"
    return {
        "dates": [p["date"] for p in priors] + [date],
        "lesions": [
            {
                "id": lesion["id"],
                "measurements": lesion["rows"] + [{**_row(date, lesion["now"]), "point": lesion["point"]}],
            }
            for lesion in lesions
        ],
    }
