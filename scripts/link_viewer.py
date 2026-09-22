"""Copy our viewer sources into the OHIF checkout.

The checkout under viewer/Viewers is a build dependency, pinned to a release
tag and gitignored. Our code lives in viewer/ and is committed. This script is
the seam between them, and it only ever writes into the checkout — so a wiped
checkout costs a re-clone plus one run of this, and never costs source.

Run it after cloning OHIF, and again whenever the config changes.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CHECKOUT = REPO / "viewer" / "Viewers"

# (our file, path inside the checkout)
COPIES = [
    (REPO / "viewer" / "config" / "radviewer.js",
     CHECKOUT / "platform" / "app" / "public" / "config" / "radviewer.js"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkout",
        type=Path,
        default=CHECKOUT,
        help="path to the OHIF checkout",
    )
    args = parser.parse_args()

    if not (args.checkout / "platform" / "app").is_dir():
        sys.exit(
            f"no OHIF checkout at {args.checkout}\n"
            "  git clone --depth 1 https://github.com/OHIF/Viewers.git viewer/Viewers"
        )

    for source, destination in COPIES:
        if not source.exists():
            sys.exit(f"missing source file {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"{source.relative_to(REPO)} -> {destination.relative_to(args.checkout)}")


if __name__ == "__main__":
    main()
