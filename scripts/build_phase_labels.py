import json
import os 
import sys 
from pathlib import Path

from totalsegmentator.bin.totalseg_get_phase import get_ct_contrast_phase

ROOT = Path(__file__).resolve().parents[1]
SUBSET = ROOT / "data/raw/totalsegmentator"
MANIFEST_PATH = SUBSET / "manifest.json"
OUTPUT_PATH = ROOT / "data/derived/contrast_phase.json"

SAVE_EVERY = 5

def load_manifest(path: Path):
    with path.open() as f:
        manifest = json.load(f)

    if isinstance(manifest, list):
        return manifest

    if isinstance(manifest, dict):
        if "cases" in manifest and isinstance(manifest["cases"], list):
            return manifest["cases"]

        return [
            {"case": case, **value}
            if isinstance(value, dict)
            else {"case": case, "ct_path": value}
            for case, value in manifest.items()
        ]

    raise ValueError(f"Unsupported manifest format: {type(manifest)!r}")

def get_case_id(record):
    # our manifest lists plain case ids like "s0000"
    if isinstance(record, str):
        return record
    for key in ("case", "case_id", "id", "name"):
        if key in record:
            return str(record[key])
    raise KeyError(f"Could not find case id in manifest record: {record}")

def get_ct_path(record):
    if isinstance(record, str):
        return SUBSET / record / "ct.nii.gz"
    for key in ("ct_path", "image_path", "path", "ct"):
        if key in record:
            path =  Path(record[key])
            if not path.is_absolute():
                path = ROOT / path
            return path

    raise KeyError(
        f"Could not find CT path in manifest record: {record}"
    )

def predict_phase(ct_path: Path):
    """
    TotalSegmentator returns several progress dictionaries.

    The final yield has:
        {
            "id": 5,
            "progress": 100,
            "status": "Done",
            "result": {
                "pi_time": ...,
                "phase": ...,
                "probability": ...,
                ...
            }
        }
    """
    final = None

    for item in get_ct_contrast_phase(
        ct_path,
        device="cpu",
        quiet=True,
    ):
        # Useful if this is ever called interactively/debugged.
        print(item, flush=True)
        final = item

    if final is None:
        raise RuntimeError(f"No result yielded for {ct_path}")

    result = final.get("result")
    if result is None:
        raise RuntimeError(
            f"Generator finished without a result for {ct_path}: {final}"
        )

    return {
        "phase": result["phase"],
        "pi_time": result["pi_time"],
        "probability": result["probability"],
    }

def save_results(results):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Atomic-ish write: 
    # don't leave a half-written JSON if interrupted
    tmp_path = OUTPUT_PATH.with_suffix(".json.tmp")
    with tmp_path.open("w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write("\n")

    tmp_path.replace(OUTPUT_PATH)

def main():
    manifest = load_manifest(MANIFEST_PATH)

    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open() as f:
            results = json.load(f)
    else:
        results = {}

    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Cases:    {len(manifest)}")
    print(f"Existing: {len(results)}")
    print()

    failures = []

    for i, record in enumerate(manifest, start=1):
        case = get_case_id(record)

        if case in results:
            print(f"[{i}/{len(manifest)}] {case}: already done")
            continue

        ct_path = get_ct_path(record)

        if not ct_path.exists():
            print(f"[{i}/{len(manifest)}] {case}: MISSING {ct_path}")
            failures.append(case)
            continue

        print(
            f"[{i}/{len(manifest)}] {case}: "
            f"classifying {ct_path}",
            flush=True,
        )

        try:
            result = predict_phase(ct_path)
        except Exception as exc:
            print(
                f"    ERROR: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            failures.append(case)
            continue

        results[case] = result

        print(
            f"    phase={result['phase']} "
            f"pi_time={result['pi_time']} "
            f"probability={result['probability']}",
            flush=True,
        )

        if len(results) % SAVE_EVERY == 0:
            save_results(results)
            print(f"    saved {len(results)} results")

    save_results(results)

    print()
    print(f"Finished: {len(results)}/{len(manifest)}")
    if failures:
        print(f"Failed/missing: {len(failures)}")
        for case in failures:
            print(f"  {case}")


if __name__ == "__main__":
    main()