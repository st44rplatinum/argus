"""Upload a directory of DICOM files into the local Orthanc.

Uses Orthanc's REST /instances endpoint rather than STOW-RS: it reports per
file why something was rejected, which matters a great deal the first time a
series does not appear and the reason is one bad transfer syntax.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx

DEFAULT_URL = "http://localhost:8042"
DEFAULT_AUTH = ("orthanc", "orthanc")


def push(directory: Path, base_url: str, auth: tuple[str, str]) -> None:
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    if not files:
        raise SystemExit(f"no files under {directory}")

    sent = 0
    failed: list[tuple[Path, str]] = []

    with httpx.Client(base_url=base_url, auth=auth, timeout=60.0) as client:
        for path in files:
            response = client.post(
                "/instances",
                content=path.read_bytes(),
                headers={"Content-Type": "application/dicom"},
            )
            if response.status_code in (200, 201):
                sent += 1
            else:
                # Orthanc's body explains the rejection; keep it, do not
                # collapse every failure into one count.
                failed.append((path, f"{response.status_code} {response.text[:200]}"))

    print(f"accepted {sent}/{len(files)}")
    for path, reason in failed[:10]:
        print(f"  rejected {path.name}: {reason}")
    if len(failed) > 10:
        print(f"  ... and {len(failed) - 10} more")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="folder of DICOM files")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--user", default=DEFAULT_AUTH[0])
    parser.add_argument("--password", default=DEFAULT_AUTH[1])
    args = parser.parse_args()

    push(args.directory, args.url, (args.user, args.password))


if __name__ == "__main__":
    main()
