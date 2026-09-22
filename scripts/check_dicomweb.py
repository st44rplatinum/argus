"""Walk the local PACS over DICOMweb and print what it holds.

This is P0's ships-when check. It deliberately speaks QIDO-RS rather than
Orthanc's own REST API: DICOMweb is the contract OHIF and the AI service will
both use, so if this works, they will too — and if it does not, the problem is
found here rather than three layers up inside a React data source.

Tags are printed by keyword and number together. The numbers are the actual
protocol; the keywords are a convenience that hides how thin the model is.
"""

from __future__ import annotations

import argparse

import httpx

DEFAULT_URL = "http://localhost:8042/dicom-web"
DEFAULT_AUTH = ("orthanc", "orthanc")

# (tag, keyword) pairs we care about at each level.
STUDY_TAGS = [
    ("0010,0010", "PatientName"),
    ("0010,0020", "PatientID"),
    ("0008,0020", "StudyDate"),
    ("0008,1030", "StudyDescription"),
    ("0020,000D", "StudyInstanceUID"),
]
SERIES_TAGS = [
    ("0020,000E", "SeriesInstanceUID"),
    ("0008,0060", "Modality"),
    ("0008,103E", "SeriesDescription"),
    ("0020,0011", "SeriesNumber"),
]


def value(item: dict, tag: str) -> str:
    """Pull a DICOM JSON value out of QIDO's {vr, Value:[...]} wrapper."""
    entry = item.get(tag.replace(",", ""))
    if not entry:
        return "-"
    values = entry.get("Value")
    if not values:
        return "-"
    first = values[0]
    if isinstance(first, dict):  # PN is {Alphabetic: "..."}
        return str(first.get("Alphabetic", "-"))
    return str(first)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--user", default=DEFAULT_AUTH[0])
    parser.add_argument("--password", default=DEFAULT_AUTH[1])
    args = parser.parse_args()

    with httpx.Client(
        base_url=args.url, auth=(args.user, args.password), timeout=30.0
    ) as client:
        try:
            # QIDO returns a fixed default set per level; anything outside it
            # comes back missing rather than empty, which reads as a data bug
            # when it is a query bug. Ask for what we print.
            studies = client.get(
                "/studies",
                params={"limit": 50, "includefield": "StudyDescription"},
            )
        except httpx.ConnectError:
            raise SystemExit(
                f"cannot reach {args.url} — is `docker compose up -d` running?"
            )

        studies.raise_for_status()
        payload = studies.json()

        if not payload:
            print("PACS reachable, but empty.")
            print("  python scripts/make_phantom.py")
            print("  python scripts/push_dicom.py data/raw/phantom")
            return

        for study in payload:
            study_uid = value(study, "0020,000D")
            print()
            for tag, keyword in STUDY_TAGS:
                print(f"  {tag}  {keyword:<20} {value(study, tag)}")

            series = client.get(f"/studies/{study_uid}/series")
            series.raise_for_status()

            for item in series.json():
                series_uid = value(item, "0020,000E")
                print()
                for tag, keyword in SERIES_TAGS:
                    print(f"      {tag}  {keyword:<20} {value(item, tag)}")

                instances = client.get(
                    f"/studies/{study_uid}/series/{series_uid}/instances"
                )
                instances.raise_for_status()
                print(f"      {'':<6}{'instances':<26} {len(instances.json())}")


if __name__ == "__main__":
    main()
