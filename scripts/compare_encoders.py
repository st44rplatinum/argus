"""Baseline features, learned embeddings, and the two fused, on the same rows.

Neighbours come only from the shared pool and never from the same case.
The equal-influence weight comes from the spread of each method's cosines,
not from any metric here, so it is not tuned on these results. The fixed
weights are a sensitivity check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services" / "ai"))

from app.retrieval import equal_influence_weight, fuse, standardise  # noqa: E402

BASELINE = Path("data/derived/organ_features.npz")
LEARNED = Path("data/derived/organ_embeddings_dinov2s.npz")
KIDNEYS = ("kidney_left", "kidney_right")
SWEEP = (0.25, 0.5, 0.75)


def nearest(cases, organs, unit, same_organ=False):
    scores = unit @ unit.T
    scores[cases[:, None] == cases[None, :]] = -np.inf
    if same_organ:
        scores[organs[:, None] != organs[None, :]] = -np.inf
    return scores.argmax(axis=1)


def metrics(cases, organs, unit, volume, aorta_hu):
    nn = nearest(cases, organs, unit)
    hit = organs == organs[nn]

    merged = np.where(np.isin(organs, KIDNEYS), "kidney", organs)
    kidney = np.isin(organs, KIDNEYS) & np.isin(organs[nn], KIDNEYS)

    same = nearest(cases, organs, unit, same_organ=True)
    vol = np.mean([
        spearmanr(volume[r], volume[same[r]]).statistic
        for r in (np.flatnonzero(organs == o) for o in sorted(set(organs)))
    ])

    diffs = [
        abs(aorta_hu[cases[i]] - aorta_hu[cases[j]])
        for i, j in enumerate(nn)
        if cases[i] in aorta_hu and cases[j] in aorta_hu
    ]
    values = np.array(list(aorta_hu.values()))
    rng = np.random.default_rng(0)
    random = np.abs(rng.choice(values, 5000) - rng.choice(values, 5000))

    return {
        "ident": hit.mean(),
        "merged": (merged == merged[nn]).mean(),
        "side": (organs[kidney] == organs[nn][kidney]).mean(),
        "volume": vol,
        "contrast": np.median(diffs) / np.median(random),
    }


def main() -> None:
    base = np.load(BASELINE, allow_pickle=False)
    learned = np.load(LEARNED, allow_pickle=False)

    base_rows = {(str(c), str(o)): v for c, o, v in zip(base["cases"], base["organs"], base["features"])}
    learned_rows = {
        (str(c), str(o)): v for c, o, v in zip(learned["cases"], learned["organs"], learned["embeddings"])
    }

    keys = sorted(set(base_rows) & set(learned_rows))
    cases = np.array([k[0] for k in keys])
    organs = np.array([k[1] for k in keys])

    raw = np.stack([base_rows[k] for k in keys])
    std = raw.std(axis=0)
    std[std == 0] = 1.0
    base_unit = standardise(raw, raw.mean(axis=0), std)
    learned_unit = np.stack([learned_rows[k] for k in keys]).astype(np.float64)

    names = [str(n) for n in base["feature_names"]]
    volume = raw[:, names.index("volume_ml")]
    hu = names.index("mean_hu")
    aorta_hu = {c: float(v[hu]) for (c, o), v in base_rows.items() if o == "aorta" and c in set(cases)}

    equal = equal_influence_weight(base_unit, learned_unit)
    methods = [
        ("baseline", base_unit),
        ("DINOv2-S", learned_unit),
        (f"fused equal w={equal:.2f} <- headline", fuse(base_unit, learned_unit, equal)),
    ] + [(f"fused w={w}", fuse(base_unit, learned_unit, w)) for w in SWEEP]

    print(f"{len(keys)} organs over {len(set(cases))} cases, chance {1 / len(set(organs)):.1%}")
    print("w is the baseline's share of the similarity")
    print()
    print(f"{'':30}{'identify':>10}{'kidney=1':>10}{'side':>8}{'volume':>9}{'contrast':>10}")
    for label, unit in methods:
        m = metrics(cases, organs, unit, volume, aorta_hu)
        print(
            f"{label:30}{m['ident']:>10.1%}{m['merged']:>10.1%}{m['side']:>8.1%}"
            f"{m['volume']:>9.2f}{m['contrast']:>10.2f}"
        )


if __name__ == "__main__":
    main()
