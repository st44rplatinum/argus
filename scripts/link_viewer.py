"""Copy our viewer sources into the OHIF checkout.

The checkout under viewer/Viewers is a build dependency, pinned to a release
tag and gitignored. Our code lives in viewer/ and is committed. This script is
the seam between them, and it only ever writes into the checkout — so a wiped
checkout costs a re-clone plus one run of this, and never costs source.

Run it after cloning OHIF, and again whenever the config changes.
"""

from __future__ import annotations

import argparse
import json
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

# Our packages have to sit inside the checkout to resolve `workspace:*` peer
# deps — OHIF's pnpm workspace globs are `extensions/*` and `modes/*`. Source
# still lives here; these are copies into disposable territory.
PACKAGE_TREES = [
    (REPO / "viewer" / "extensions", "extensions"),
    (REPO / "viewer" / "modes", "modes"),
]

# Build output and dependencies belong to the checkout, not to us.
IGNORE = shutil.ignore_patterns("node_modules", "dist", "*.tsbuildinfo")


def sync_packages(checkout: Path) -> list[str]:
    """Copy our extension and mode packages into the checkout."""
    done = []

    for source_root, destination_name in PACKAGE_TREES:
        if not source_root.is_dir():
            continue

        for package in sorted(p for p in source_root.iterdir() if p.is_dir()):
            destination = checkout / destination_name / package.name
            shutil.copytree(package, destination, dirs_exist_ok=True, ignore=IGNORE)
            done.append(f"{destination_name}/{package.name}")

    return done


def package_names() -> tuple[list[str], list[str]]:
    """Read the package names out of our manifests, rather than hardcoding."""
    extensions, modes = [], []

    for source_root, kind in PACKAGE_TREES:
        if not source_root.is_dir():
            continue

        for package in sorted(p for p in source_root.iterdir() if p.is_dir()):
            manifest = package / "package.json"
            if not manifest.is_file():
                continue
            name = json.loads(manifest.read_text()).get("name")
            if name:
                (extensions if kind == "extensions" else modes).append(name)

    return extensions, modes


def register_plugins(checkout: Path) -> str:
    """Add our packages to OHIF's pluginConfig.json, without disturbing theirs.

    Idempotent, and a merge rather than a replacement: OHIF's own entries
    change between releases, so shipping a whole replacement file would mean
    silently pinning their list to whatever it looked like the day we copied
    it.
    """
    config_path = checkout / "platform" / "app" / "pluginConfig.json"
    if not config_path.is_file():
        return f"no pluginConfig.json at {config_path}"

    config = json.loads(config_path.read_text())
    extensions, modes = package_names()

    added = []
    for key, names in (("extensions", extensions), ("modes", modes)):
        entries = config.setdefault(key, [])
        present = {entry.get("packageName") for entry in entries}
        for name in names:
            if name not in present:
                entries.append({"packageName": name})
                added.append(name)

    if added:
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        return "registered " + ", ".join(added)

    return "already registered"


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

    for name in sync_packages(args.checkout):
        print(f"viewer/{name} -> {name}")

    print(register_plugins(args.checkout))
    print(check_store_volume(args.checkout))


if __name__ == "__main__":
    main()
