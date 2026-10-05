"""Golden files: saved inputs and the outputs today's code gives for them.

A port (Rust, Dart) or a refactor has to reproduce these numbers. Inputs are
plain .npy files, noise included, so nothing has to be regenerated from a
random seed.

  python scripts/make_golden.py            # writes, refuses to overwrite
  python scripts/make_golden.py --update   # regenerate on purpose
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "ai"))

from app import encoder  # noqa: E402
from app.features import FEATURE_VERSION, NAMES, organ_features  # noqa: E402

GOLDEN = ROOT / "services/ai/tests/golden"
PHANTOM = ROOT / "data/raw/phantom"
ONNX = ROOT / "weights/dinov2s_224.onnx"


def ball(shape, centre, radius):
    z, y, x = np.indices(shape)
    return ((z - centre[0]) ** 2 + (y - centre[1]) ** 2 + (x - centre[2]) ** 2) <= radius**2


def phantom_spine():
    """The phantom's spine, cropped: CT as int16 and its mask, [z, y, x]."""
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(PHANTOM)))
    image = reader.Execute()
    ct = sitk.GetArrayFromImage(image).astype(np.int16)
    spacing_zyx = image.GetSpacing()[::-1]
    return ct, ct > 300, spacing_zyx


def feature_cases():
    """name -> (ct float32, mask bool, spacing), arrays [z, y, x]."""
    rng = np.random.default_rng(0)
    shape = (32, 32, 32)
    sphere = ball(shape, (16, 16, 16), 10)
    rod = np.zeros(shape, bool)
    rod[3:29, 14:18, 14:18] = True
    disc = ball(shape, (16, 16, 16), 12) & (np.abs(np.indices(shape)[0] - 16) <= 2)
    tiny = ball(shape, (16, 16, 16), 2)

    def flat(mask, inside=50.0):
        return np.where(mask, inside, -100.0).astype(np.float32)

    noisy = flat(sphere) + rng.normal(0, 15, shape).astype(np.float32)

    ct, spine, spacing = phantom_spine()
    _, ys, xs = np.nonzero(spine)
    crop = (slice(None), slice(ys.min() - 8, ys.max() + 9), slice(xs.min() - 8, xs.max() + 9))

    return {
        "sphere": (flat(sphere), sphere, (1.0, 1.0, 1.0)),
        "rod": (flat(rod), rod, (1.0, 1.0, 1.0)),
        "disc": (flat(disc), disc, (1.0, 1.0, 1.0)),
        "noisy_sphere": (noisy, sphere, (1.0, 1.0, 1.0)),
        # the only case where axis order and spacing can go wrong in a port
        "anisotropic": (flat(sphere), sphere, (2.0, 0.7, 0.7)),
        "tiny": (flat(tiny), tiny, (1.0, 1.0, 1.0)),
        "phantom_spine": (ct[crop].astype(np.float32), spine[crop], tuple(float(s) for s in spacing)),
    }


def write_features(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cases = {}
    for name, (ct, mask, spacing) in feature_cases().items():
        # CT values here are whole numbers, so int16 stores them exactly
        np.save(out / f"{name}_ct.npy", ct.astype(np.int16) if np.all(ct == np.round(ct)) else ct)
        np.save(out / f"{name}_mask.npy", mask)
        values = organ_features(ct, mask, spacing)
        cases[name] = {
            "ct": f"{name}_ct.npy",
            "mask": f"{name}_mask.npy",
            "spacing": list(spacing),
            "expected": None if values is None else [float(v) for v in values],
        }

    manifest = {"feature_version": FEATURE_VERSION, "names": list(NAMES), "axis_order": "z, y, x", "cases": cases}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def embedding_input():
    """A spine sub-volume for the embedding pipeline, [x, y, z] as the encoder expects."""
    ct, spine, spacing_zyx = phantom_spine()
    half = int(encoder.FOV_MM / 2 / spacing_zyx[2]) + 4  # a little more than the crop the encoder takes
    _, ys, xs = np.nonzero(spine)
    cy, cx = int(ys.mean()), int(xs.mean())
    z0 = ct.shape[0] // 2 - 6
    box = (slice(z0, z0 + 12), slice(max(cy - half, 0), cy + half), slice(max(cx - half, 0), cx + half))
    to_xyz = lambda a: np.ascontiguousarray(a[box].transpose(2, 1, 0))  # noqa: E731
    return to_xyz(ct), to_xyz(spine), tuple(float(s) for s in spacing_zyx[::-1])


def write_embedding(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    ct, mask, spacing = embedding_input()
    np.save(out / "ct.npy", ct)
    np.save(out / "mask.npy", mask)

    stages = encoder.stages(ct.astype(np.float32), mask, spacing, backend="torch")
    np.save(out / "slices.npy", stages["slices"])
    np.save(out / "cropped.npy", np.stack(stages["cropped"]))
    # resized images are 224x224 each; two are enough to pin the resize
    np.save(out / "resized.npy", stages["resized"][:2])
    np.save(out / "weights.npy", np.stack(stages["weights"]))
    np.save(out / "embedding.npy", stages["embedding"])

    manifest = {
        "encoder_id": encoder.ENCODER_ID,
        "preprocess_version": encoder.PREPROCESS_VERSION,
        "axis_order": "x, y, z",
        "spacing": list(spacing),
        "onnx_sha256": hashlib.sha256(ONNX.read_bytes()).hexdigest() if ONNX.is_file() else None,
        "stages": {
            "ct.npy": "1. input CT, int16 HU",
            "mask.npy": "1. input mask",
            "slices.npy": "2. selected z indices",
            "cropped.npy": "3. each slice windowed, cropped, rotated, before resizing",
            "resized.npy": "4. first two slices after resizing to 224x224",
            "weights.npy": "5. each slice's 256 patch weights",
            "embedding.npy": "6. the 384-d embedding",
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--update", action="store_true", help="overwrite existing golden files")
    args = parser.parse_args()

    for name, write in (("features", write_features), ("embedding", write_embedding)):
        out = GOLDEN / name
        if (out / "manifest.json").exists() and not args.update:
            raise SystemExit(f"{out} exists; regenerating is a decision, pass --update")
        write(out)
        size = sum(p.stat().st_size for p in out.iterdir()) / 1024
        print(f"wrote {out.relative_to(ROOT)} ({size:.0f} KB)")


if __name__ == "__main__":
    main()
