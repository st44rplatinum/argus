"""Organ embeddings from a frozen DINOv2 over axial slices.

Patch features are averaged inside the organ mask, per slice, then across
slices. Nothing is trained here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ENCODER_ID = "vit_small_patch14_dinov2.lvd142m"
# Bump when slicing, cropping or windowing changes.
PREPROCESS_VERSION = 1

IMAGE_SIZE = 224
GRID = IMAGE_SIZE // 14          # 16x16 patches
FOV_MM = 240.0                   # fixed field of view, keeps organ size visible
WINDOW = (-160.0, 240.0)         # soft tissue
SLICES = 8

# DINOv2's training normalisation (ImageNet); test_onnx_parity checks it
# against the model's own config
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

ONNX_PATH = Path(__file__).resolve().parents[3] / "weights" / "dinov2s_224.onnx"

_model = None
_session = None


def load_model(threads: int = 4):
    global _model
    if _model is None:
        import timm
        import torch

        torch.set_num_threads(threads)
        _model = timm.create_model(ENCODER_ID, pretrained=True, img_size=IMAGE_SIZE).eval()
    return _model


def select_slices(mask: np.ndarray, count: int = SLICES) -> np.ndarray:
    """Evenly spaced axial slices through the organ. mask is [x, y, z]."""
    present = np.flatnonzero(mask.any(axis=(0, 1)))
    if len(present) <= count:
        return present
    picks = np.linspace(0, len(present) - 1, count).round().astype(int)
    return present[picks]


def crop_fov(image: np.ndarray, centre: tuple[float, float], half: tuple[int, int], fill: float):
    """Square-in-mm crop around a centre, padded where it leaves the image."""
    cx, cy = (int(round(c)) for c in centre)
    hx, hy = half
    out = np.full((2 * hx, 2 * hy), fill, dtype=image.dtype)

    x0, x1 = cx - hx, cx + hx
    y0, y1 = cy - hy, cy + hy
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x1, image.shape[0]), min(y1, image.shape[1])

    if sx0 < sx1 and sy0 < sy1:
        out[sx0 - x0 : sx1 - x0, sy0 - y0 : sy1 - y0] = image[sx0:sx1, sy0:sy1]
    return out


def patch_weights(mask_2d: np.ndarray) -> np.ndarray:
    """Fraction of each patch covered by the organ, flattened to GRID*GRID."""
    import torch
    import torch.nn.functional as F

    m = torch.from_numpy(mask_2d.astype(np.float32))[None, None]
    m = F.interpolate(m, size=(IMAGE_SIZE, IMAGE_SIZE), mode="nearest")
    return F.adaptive_avg_pool2d(m, GRID).flatten().numpy()


def _prepare(ct: np.ndarray, mask: np.ndarray, spacing: tuple[float, float, float]):
    """Images and patch weights for the chosen slices."""
    half = (int(FOV_MM / 2 / spacing[0]), int(FOV_MM / 2 / spacing[1]))
    lo, hi = WINDOW

    images, weights = [], []
    for z in select_slices(mask):
        organ = mask[:, :, z]
        xs, ys = np.nonzero(organ)
        centre = (xs.mean(), ys.mean())

        image = crop_fov(ct[:, :, z], centre, half, fill=-1024.0)
        region = crop_fov(organ, centre, half, fill=False)

        # same orientation for every case
        image = np.rot90(image)
        region = np.rot90(region)

        images.append(np.clip((image - lo) / (hi - lo), 0.0, 1.0).astype(np.float32))
        weights.append(patch_weights(region))

    return images, weights


def resize(images) -> np.ndarray:
    """Each prepared slice to 224x224, bilinear, as the network wants."""
    import torch
    import torch.nn.functional as F

    out = [
        F.interpolate(torch.from_numpy(np.ascontiguousarray(image))[None, None],
                      size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)[0, 0].numpy()
        for image in images
    ]
    return np.stack(out).astype(np.float32)


def network_input(resized: np.ndarray) -> np.ndarray:
    """Grey to three channels, normalised the way DINOv2 was trained: (N, 3, 224, 224)."""
    x = np.repeat(resized[:, None], 3, axis=1)
    return ((x - np.array(MEAN, np.float32)[None, :, None, None])
            / np.array(STD, np.float32)[None, :, None, None]).astype(np.float32)


def patch_tokens(x: np.ndarray, backend: str = "torch", threads: int = 4) -> np.ndarray:
    """The network: (N, 3, 224, 224) in, (N, 256, 384) patch tokens out.

    "onnx" runs the exported file (scripts/export_dinov2_onnx.py) with ONNX
    Runtime and needs no torch; "torch" runs the timm model.
    """
    if backend == "onnx":
        return _onnx_session(threads).run(None, {"images": x})[0]

    import torch

    model = load_model(threads)
    with torch.no_grad():
        return model.forward_features(torch.from_numpy(x))[:, model.num_prefix_tokens :, :].numpy()


def _onnx_session(threads: int):
    global _session
    if _session is None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        _session = ort.InferenceSession(str(ONNX_PATH), options, providers=["CPUExecutionProvider"])
    return _session


def pool(tokens: np.ndarray, weights) -> np.ndarray | None:
    """Average the organ's patches per slice, then the slices; unit length."""
    per_slice = []
    for patches, w in zip(tokens, weights):
        # mostly-organ patches if there are any, otherwise whatever touches it
        keep = w >= 0.5
        if not keep.any():
            keep = w > 0
        if not keep.any():
            continue
        per_slice.append(np.average(patches[keep], axis=0, weights=w[keep]))

    if not per_slice:
        return None
    vector = np.mean(per_slice, axis=0)
    return vector / np.linalg.norm(vector)


def stages(ct: np.ndarray, mask: np.ndarray, spacing, backend: str = "torch", threads: int = 4) -> dict:
    """Every step of the embedding, so golden files can pin each one."""
    mask = mask.astype(bool)
    slices = select_slices(mask)
    images, weights = _prepare(ct, mask, spacing)
    if not images:
        return {"slices": slices, "cropped": [], "resized": None, "weights": [], "embedding": None}

    resized = resize(images)
    tokens = patch_tokens(network_input(resized), backend, threads)
    return {
        "slices": slices,
        "cropped": images,
        "resized": resized,
        "weights": weights,
        "embedding": pool(tokens, weights),
    }


def organ_embedding(ct: np.ndarray, mask: np.ndarray, spacing, threads: int = 4, backend: str = "torch"):
    """384-d unit vector for one organ, or None if it is absent.

    ct and mask are [x, y, z] in canonical orientation, spacing in mm.
    """
    if mask.sum() < 100:
        return None
    return stages(ct, mask, spacing, backend, threads)["embedding"]
