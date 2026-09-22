"""Copy our viewer sources into the OHIF checkout.

The checkout under viewer/Viewers is a build dependency, pinned to a release
tag and gitignored. Our code lives in viewer/ and is committed. This script is
the seam between them, and it only ever writes into the checkout — so a wiped
checkout costs a re-clone plus one run of this, and never costs source.

Run it after cloning OHIF, and again whenever the config changes.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CHECKOUT = REPO / "viewer" / "Viewers"

# (our file, path inside the checkout)
COPIES = [
    (REPO / "viewer" / "config" / "radviewer.js",
     CHECKOUT / "platform" / "app" / "public" / "config" / "radviewer.js"),
]


def check_store_volume(checkout: Path) -> str:
    """Warn if pnpm's store is on a different volume from the checkout.

    pnpm hardlinks packages out of its store. Across volumes it cannot, and
    copies every file of every package instead — on this machine's spinning
    disk that was twenty minutes of pure I/O for one install.

    pnpm already picks a store on the project's own drive, so this asserts the
    property rather than trying to set it. An earlier version of this script
    wrote store-dir into .npmrc; pnpm ignored it, and the config quietly did
    nothing while appearing to work.
    """
    try:
        result = subprocess.run(
            ["corepack", "pnpm@11.5.2", "store", "path"],
            cwd=checkout,
            capture_output=True,
            text=True,
            shell=os.name == "nt",
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "store: could not ask pnpm (corepack missing or slow)"

    store = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    if not store:
        return "store: pnpm did not report a path"

    same = Path(store).drive.upper() == checkout.resolve().drive.upper()
    verdict = "same volume, hardlinks work" if same else "DIFFERENT VOLUME — installs will copy, not link"
    return f"store: {store} ({verdict})"


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

    print(check_store_volume(args.checkout))


if __name__ == "__main__":
    main()
