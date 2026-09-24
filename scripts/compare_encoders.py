"""Baseline features against learned embeddings, on the same rows.

Both are scored on the (case, organ) pairs that have an embedding, with
neighbours drawn only from that pool. Same questions as evaluate_retrieval:
organ identification, and how much neighbours agree on contrast phase.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services" / "ai"))

from app.retrieval import standardise  # noqa: E402

BASELINE = Path("data/derived/organ_features.npz")
LEARNED = Path("data/derived/organ_embeddings_dinov2s.npz")
KIDNEYS = ("kidney_left", "kidney_right")


def nearest_other_case(cases: np.ndarray, unit: np.ndarray) -> np.ndarray:
    scores = unit @ unit.T
    scores[cases[:, None] == cases[None, :]] = -np.inf
    return scores.argmax(axis=1)


def accuracy(organs: np.ndarray, nearest: np.ndarray, merge_kidneys: bool = False) -> float:
    a, b = organs, organs[nearest]
    if merge_kidneys:
        a = np.where(np.isin(a, KIDNEYS), "kidney", a)
        b = np.where(np.isin(b, KIDNEYS), "kidney", b)
    return float((a == b).mean())


def per_organ(organs: np.ndarray, nearest: np.ndarray) -> dict[str, float]:
    hit = organs == organs[nearest]
    return {o: float(hit[organs == o].mean()) for o in sorted(set(organs))}


def left_right(organs: np.ndarray, nearest: np.ndarray) -> float:
    """Of kidneys that retrieved a kidney, how often the correct side."""
    kidney = np.isin(organs, KIDNEYS) & np.isin(organs[nearest], KIDNEYS)
    return float((organs[kidney] == organs[nearest][kidney]).mean())


def contrast_ratio(cases, nearest, aorta_hu: dict[str, float]) -> float:
    pairs = [
        abs(aorta_hu[cases[i]] - aorta_hu[cases[j]])
        for i, j in enumerate(nearest)
        if cases[i] in aorta_hu and cases[j] in aorta_hu
    ]
    values = np.array(list(aorta_hu.values()))
    rng = np.random.default_rng(0)
    random = np.abs(rng.choice(values, 5000) - rng.choice(values, 5000))
    return float(np.median(pairs) / np.median(random))


def volume_agreement(cases, organs, unit, volume) -> float:
    """Spearman correlation of volume between each organ and its nearest
    same-organ neighbour. Does "similar liver" mean a similar-sized liver?"""
    from scipy.stats import spearmanr

    scores = unit @ unit.T
    scores[cases[:, None] == cases[None, :]] = -np.inf
    scores[organs[:, None] != organs[None, :]] = -np.inf
    nearest = scores.argmax(axis=1)

    per = []
    for organ in sorted(set(organs)):
        rows = np.flatnonzero(organs == organ)
        per.append(spearmanr(volume[rows], volume[nearest[rows]]).statistic)
    return float(np.mean(per))


def main() -> None:
    base = np.load(BASELINE, allow_pickle=False)
    learned = np.load(LEARNED, allow_pickle=False)

    base_rows = {
        (str(c), str(o)): v for c, o, v in zip(base["cases"], base["organs"], base["features"])
    }
    learned_rows = {
        (str(c), str(o)): v for c, o, v in zip(learned["cases"], learned["organs"], learned["embeddings"])
    }

    keys = sorted(set(base_rows) & set(learned_rows))
    cases = np.array([k[0] for k in keys])
    organs = np.array([k[1] for k in keys])

    # restandardised over this pool, so the baseline sees no more data than the encoder
    raw = np.stack([base_rows[k] for k in keys])
    std = raw.std(axis=0)
    std[std == 0] = 1.0
    methods = {
        "baseline (11 features)": standardise(raw, raw.mean(axis=0), std),
        f"DINOv2-S ({learned['embeddings'].shape[1]}-d)": np.stack([learned_rows[k] for k in keys]),
    }

    names = [str(n) for n in base["feature_names"]]
    hu = names.index("mean_hu")
    aorta_hu = {c: float(v[hu]) for (c, o), v in base_rows.items() if o == "aorta" and c in set(cases)}

    print(f"{len(keys)} organs over {len(set(cases))} cases, chance {1 / len(set(organs)):.1%}")
    print()
    header = f"{'':28}" + "".join(f"{m:>26}" for m in methods)
    print(header)

    results = {m: nearest_other_case(cases, u) for m, u in methods.items()}

    def row(label, fn, fmt="{:.1%}"):
        print(f"{label:28}" + "".join(f"{fmt.format(fn(n)):>26}" for n in results.values()))

    row("identification", lambda n: accuracy(organs, n))
    row("  kidneys merged", lambda n: accuracy(organs, n, merge_kidneys=True))
    row("  kidney side correct", lambda n: left_right(organs, n))
    for organ in sorted(set(organs)):
        row(f"    {organ}", lambda n, o=organ: per_organ(organs, n)[o])
    row("contrast ratio (lower=worse)", lambda n: contrast_ratio(cases, n, aorta_hu), fmt="{:.2f}")

    volume = raw[:, names.index("volume_ml")]
    print(
        f"{'same-organ volume corr':28}"
        + "".join(f"{volume_agreement(cases, organs, u, volume):>26.2f}" for u in methods.values())
    )


if __name__ == "__main__":
    main()
