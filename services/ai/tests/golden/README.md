# Golden files

Saved inputs and the answers today's code gives. A port (Rust, Dart) or a
refactor has to reproduce them. Regenerate only on purpose:
`python scripts/make_golden.py --update`.

All arrays are plain `.npy`. Each folder's `manifest.json` lists the files,
spacing and axis order.

## features/

Seven cases through the 11 organ features, arrays `[z, y, x]`, CT stored as
int16 where the values are whole (load as float32). Compare at `rtol=1e-6`.
`tiny` is under 100 voxels: the answer is "no features".

numpy defaults a port gets wrong without noticing:

| where | numpy does | a port often does |
|---|---|---|
| `np.percentile` | linear interpolation | nearest, or its own method |
| `std`, skew | population (divide by n) | sample (n - 1) |
| `np.gradient` | central inside, one-sided at edges, per voxel | per mm, or zero at edges |
| surface voxels | 6-connected, border counts as outside | 26-connected |

## embedding/

One organ (the phantom's spine), arrays `[x, y, z]`, every stage saved so a
port can find where it drifts:

1. `ct.npy`, `mask.npy`: the input
2. `slices.npy`: chosen z indices
3. `cropped.npy`: each slice windowed, cropped, rotated
4. `resized.npy`: the first two slices at 224x224
5. `weights.npy`: each slice's 256 patch weights
6. `embedding.npy`: the result; match at cosine > 0.9999

The network is `weights/dinov2s_224.onnx` (`scripts/export_dinov2_onnx.py`,
sha256 in the manifest). Traps between the stages:

- `crop_fov` rounds the centre with Python's `round`, which rounds halves to
  even (`round(2.5) == 2`)
- the crop's half-width `int(FOV_MM / 2 / spacing)` truncates
- `np.rot90` defaults: one quarter turn, counter-clockwise
- image resize: bilinear, `align_corners=False`, no antialiasing (stage 4)
- mask resize: torch `nearest`, which floors; OpenCV and `nearest-exact`
  round (stage 5)
