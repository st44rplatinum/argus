from __future__ import annotations

import hashlib
import json
from pathlib import Path


def code_version(root: Path | None = None) -> str:
    """A fingerprint of the service's code, so a fix can't be hidden by an old cached answer."""
    digest = hashlib.sha256()
    for path in sorted((root or Path(__file__).parent).rglob("*.py")):
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def make_cache_key(
    series_instance_uid,
    model_name,
    model_version,
    parameters,
):
    value = {
        "series_instance_uid": series_instance_uid,
        "model_name": model_name,
        "model_version": str(model_version),
        "parameters": parameters,
    }

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()
