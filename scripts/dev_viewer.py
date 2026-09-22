"""Start the OHIF dev server against the local Orthanc.

The environment below is what makes viewer/config/radviewer.js work: OHIF's
dev server proxies /pacs/dicom-web to Orthanc, so the browser stays
same-origin and Orthanc needs no CORS at all.

This lives here rather than as a script in the OHIF checkout because the
checkout is gitignored and disposable — a re-clone must not cost us the
knowledge of how to start the thing.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CHECKOUT = REPO / "viewer" / "Viewers"

ORTHANC = "http://localhost:8042"
PROXY_PATH = "/pacs/dicom-web"
PORT = "3000"

# Corepack runs the pinned pnpm without needing a global install or the
# elevated permissions that `corepack enable` wanted.
PNPM_VERSION = "pnpm@11.5.2"
PNPM = ["corepack", PNPM_VERSION]

# OHIF's own scripts shell out to a bare `pnpm` (the root `dev` script is
# `pnpm --filter @ohif/app run dev:viewer`). Without corepack's shims that is
# not on PATH, so we generate one and prepend it. Cheaper than requiring an
# elevated `corepack enable`, and it survives a wiped checkout.
SHIM_DIR = REPO / ".tooling" / "bin"


def ensure_pnpm_shim() -> Path:
    SHIM_DIR.mkdir(parents=True, exist_ok=True)

    if os.name == "nt":
        shim = SHIM_DIR / "pnpm.cmd"
        shim.write_text(f"@echo off\r\ncorepack {PNPM_VERSION} %*\r\n")
    else:
        shim = SHIM_DIR / "pnpm"
        shim.write_text(f'#!/bin/sh\nexec corepack {PNPM_VERSION} "$@"\n')
        shim.chmod(0o755)

    return SHIM_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=PORT)
    parser.add_argument("--orthanc", default=ORTHANC)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="rsbuild instead of rspack — faster reloads, occasionally divergent",
    )
    args = parser.parse_args()

    if not (CHECKOUT / "node_modules").is_dir():
        sys.exit(
            f"no dependencies in {CHECKOUT}\n"
            "  cd viewer/Viewers && corepack pnpm@11.5.2 install"
        )

    shim_dir = ensure_pnpm_shim()

    env = os.environ | {
        "PATH": f"{shim_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "NODE_ENV": "development",
        "APP_CONFIG": "config/radviewer.js",
        "OHIF_PORT": args.port,
        # Don't hijack a browser window on start; the URL is printed below.
        "OHIF_OPEN": "false",
        # PROXY_TARGET is only a presence check in OHIF's webpack config; the
        # rewrite pair below is what actually routes.
        "PROXY_TARGET": f"http://localhost:{args.port}{PROXY_PATH}",
        "PROXY_DOMAIN": args.orthanc,
        "PROXY_PATH_REWRITE_FROM": PROXY_PATH,
        "PROXY_PATH_REWRITE_TO": "/dicom-web",
    }

    script = "dev:fast" if args.fast else "dev"
    command = [*PNPM, "run", script]

    print(f"OHIF on http://localhost:{args.port}")
    print(f"  {PROXY_PATH} -> {args.orthanc}/dicom-web")

    try:
        raise SystemExit(
            subprocess.call(command, cwd=CHECKOUT, env=env, shell=os.name == "nt")
        )
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
