"""
Baseline features, learned embeddings, and the two fused, on the same rows.

Neighbours come only from the shared pool and never from the same case.

The equal-influence weight comes from the spread of each method's cosines,
not from any metric here, so it is not tuned on these results. The fixed
weights are a sensitivity check.

Each method is evaluated with and without phase deconfounding.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "services" / "ai"),
)

from app.deconfound import center_by_group  # noqa: E402
from app.retrieval import equal_influence_weight, fuse, standardise  # noqa: E402


BASELINE = Path("data/derived/organ_features.npz")
LEARNED = Path("data/derived/organ_embeddings_dinov2s.npz")
PHASES = Path("data/derived/contrast_phase.json")

KIDNEYS = ("kidney_left", "kidney_right")
SWEEP = (0.25, 0.5, 0.75)

MIN_PHASE_GROUP = 5


def nearest(cases, organs, unit, same_organ=False):
    scores = unit @ unit.T

    # Never retrieve another organ from the same case.
    scores[cases[:, None] == cases[None, :]] = -np.inf

    if same_organ:
        scores[organs[:, None] != organs[None, :]] = -np.inf

    return scores.argmax(axis=1)


def phase_leakage(phases, nn):
    """
    Fraction of nearest neighbours sharing phase, compared with the
    random-pair base rate.

    ratio = observed / sum(p_phase ** 2)

    1.0 means phase agreement is exactly at chance.
    """
    phases = np.asarray(phases, dtype=object)

    valid = np.array(
        [p is not None and p != "" for p in phases],
        dtype=bool,
    )

    if not valid.any():
        return {
            "observed": float("nan"),
            "base_rate": float("nan"),
            "ratio": float("nan"),
        }

    counts = np.unique(phases[valid], return_counts=True)[1]
    proportions = counts / counts.sum()
    base_rate = float(np.sum(proportions ** 2))

    observed = float(
        np.mean(phases[valid] == phases[nn[valid]])
    )

    ratio = (
        observed / base_rate
        if base_rate > 0
        else float("nan")
    )

    return {
        "observed": observed,
        "base_rate": base_rate,
        "ratio": ratio,
    }


def metrics(cases, organs, phases, unit, volume, aorta_hu):
    nn = nearest(cases, organs, unit)

    hit = organs == organs[nn]

    merged = np.where(
        np.isin(organs, KIDNEYS),
        "kidney",
        organs,
    )

    kidney = (
        np.isin(organs, KIDNEYS)
        & np.isin(organs[nn], KIDNEYS)
    )

    same = nearest(
        cases,
        organs,
        unit,
        same_organ=True,
    )

    volume_correlations = []

    for organ in sorted(set(organs)):
        rows = np.flatnonzero(organs == organ)

        # Spearman needs at least two observations.
        if len(rows) < 2:
            continue

        correlation = spearmanr(
            volume[rows],
            volume[same[rows]],
        ).statistic

        if np.isfinite(correlation):
            volume_correlations.append(correlation)

    vol = (
        float(np.mean(volume_correlations))
        if volume_correlations
        else float("nan")
    )

    # Existing aortic-HU leakage metric.
    diffs = [
        abs(aorta_hu[cases[i]] - aorta_hu[cases[j]])
        for i, j in enumerate(nn)
        if cases[i] in aorta_hu and cases[j] in aorta_hu
    ]

    values = np.array(list(aorta_hu.values()))

    rng = np.random.default_rng(0)
    random = np.abs(
        rng.choice(values, 5000)
        - rng.choice(values, 5000)
    )

    contrast = (
        np.median(diffs) / np.median(random)
        if diffs
        else float("nan")
    )

    phase = phase_leakage(phases, nn)

    return {
        "ident": float(hit.mean()),
        "merged": float((merged == merged[nn]).mean()),
        "side": float((organs[kidney] == organs[nn][kidney]).mean()),
        "volume": vol,
        "contrast": float(contrast),
        "phase": phase["ratio"],
        "phase_observed": phase["observed"],
        "phase_base": phase["base_rate"],
    }


def main() -> None:
    base = np.load(BASELINE, allow_pickle=False)
    learned = np.load(LEARNED, allow_pickle=False)

    with PHASES.open() as f:
        phase_data = json.load(f)

    base_rows = {
        (str(c), str(o)): v
        for c, o, v in zip(
            base["cases"],
            base["organs"],
            base["features"],
        )
    }

    learned_rows = {
        (str(c), str(o)): v
        for c, o, v in zip(
            learned["cases"],
            learned["organs"],
            learned["embeddings"],
        )
    }

    keys = sorted(set(base_rows) & set(learned_rows))

    cases = np.array([k[0] for k in keys])
    organs = np.array([k[1] for k in keys])

    raw = np.stack([
        base_rows[k]
        for k in keys
    ])

    learned_raw = np.stack([
        learned_rows[k]
        for k in keys
    ]).astype(np.float64)

    # Phase is case-level, not organ-level. A case the classifier could not
    # label counts as unknown rather than stopping the run.
    missing_phases = sorted(c for c in set(cases) if c not in phase_data)
    if missing_phases:
        print(f"no phase label for {len(missing_phases)} cases, treated as unknown")

    phases = np.array(
        [phase_data[c]["phase"] if c in phase_data else None for c in cases],
        dtype=object,
    )

    names = [
        str(n)
        for n in base["feature_names"]
    ]

    def standardised(values):
        std = values.std(axis=0)
        std[std == 0] = 1.0
        return standardise(values, values.mean(axis=0), std)

    # Centre the raw features, then standardise, so the vectors stay unit
    # length. Centring an already-normalised vector breaks the cosine.
    base_unit = standardised(raw)
    base_centred = standardised(
        center_by_group(raw, organs, phases, columns=names, min_group=MIN_PHASE_GROUP)
    )

    learned_unit = learned_raw
    learned_centred = center_by_group(
        learned_raw, organs, phases, columns=None, min_group=MIN_PHASE_GROUP
    )

    volume = raw[:, names.index("volume_ml")]

    hu = names.index("mean_hu")

    case_set = set(cases)

    aorta_hu = {
        c: float(v[hu])
        for (c, o), v in base_rows.items()
        if o == "aorta" and c in case_set
    }

    def methods(base_vectors, learned_vectors):
        # Weight comes from the representations themselves, never from the
        # metrics below. Fuse after centring each half, so the baseline's
        # shape columns are never shifted.
        equal = equal_influence_weight(base_vectors, learned_vectors)
        out = [
            ("baseline", base_vectors),
            ("DINOv2-S", learned_vectors),
            (f"fused equal w={equal:.2f}", fuse(base_vectors, learned_vectors, equal)),
        ]
        out += [(f"fused w={w}", fuse(base_vectors, learned_vectors, w)) for w in SWEEP]
        return out

    names_and_vectors = []
    for (label, before), (label_after, after) in zip(
        methods(base_unit, learned_unit), methods(base_centred, learned_centred)
    ):
        names_and_vectors.append((label, False, before))
        names_and_vectors.append((label_after, True, after))

    print(
        f"{len(keys)} organs over "
        f"{len(set(cases))} cases, "
        f"chance {1 / len(set(organs)):.1%}"
    )

    print(
        "w is the baseline's share of the similarity"
    )

    print(
        f"phase groups with min size "
        f"{MIN_PHASE_GROUP}"
    )

    print()

    print(
        f"{'':34}"
        f"{'identify':>10}"
        f"{'kidney=1':>10}"
        f"{'side':>8}"
        f"{'volume':>9}"
        f"{'aorta':>9}"
        f"{'phase':>9}"
    )

    for label, deconfounded, unit in names_and_vectors:
        m = metrics(
            cases,
            organs,
            phases,
            unit,
            volume,
            aorta_hu,
        )

        suffix = " + phase-centred" if deconfounded else ""

        print(
            f"{(label + suffix):34}"
            f"{m['ident']:>10.1%}"
            f"{m['merged']:>10.1%}"
            f"{m['side']:>8.1%}"
            f"{m['volume']:>9.2f}"
            f"{m['contrast']:>9.2f}"
            f"{m['phase']:>9.2f}"
        )

    print()
    print(
        "Phase leakage: observed neighbour phase agreement / "
        "random-pair phase agreement."
    )
    print(
        "1.00 = chance; values above 1.00 indicate phase-enriched "
        "neighbours."
    )


if __name__ == "__main__":
    main()
