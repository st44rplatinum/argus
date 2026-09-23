import hashlib
import json


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
