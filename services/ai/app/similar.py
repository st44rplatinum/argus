"""Similar cases: fit on every indexed case, search only the openable pool.

Standardisation, the fusion weight and the phase offsets are all fitted on
the full index. Fitted on the 20 pool cases instead, almost every (organ,
phase) group is under the minimum and centring silently does nothing.

The maths matches scripts/compare_encoders.py, which is where the numbers
in the P5 results came from.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .deconfound import apply_phase_offsets, fit_phase_offsets
from .encoder import ENCODER_ID, PREPROCESS_VERSION
from .retrieval import equal_influence_weight, standardise

# from the repo root, wherever the service was started
DERIVED = Path(__file__).resolve().parents[3] / "data" / "derived"
BASELINE = DERIVED / "organ_features.npz"
LEARNED = DERIVED / "organ_embeddings_dinov2s.npz"
PHASES = DERIVED / "contrast_phase.json"
POOL = DERIVED / "pool.json"

MIN_PHASE_GROUP = 5
REASONS = 2


@dataclass(frozen=True)
class Space:
    """One way of comparing organs: plain, or phase-centred."""

    base: np.ndarray        # (n, 11) standardised, unit length
    learned: np.ndarray     # (n, 384) unit length
    mean: np.ndarray        # baseline standardisation
    std: np.ndarray
    weight: float           # baseline's share of the similarity


@dataclass(frozen=True)
class SimilarIndex:
    cases: np.ndarray
    organs: np.ndarray
    phases: np.ndarray
    names: tuple[str, ...]
    plain: Space
    centred: Space
    base_offsets: dict
    learned_offsets: dict
    pool: dict
    version: str


def _fit_space(raw: np.ndarray, learned: np.ndarray) -> Space:
    mean = raw.mean(axis=0)
    std = raw.std(axis=0)
    std[std == 0] = 1.0
    base = standardise(raw, mean, std)
    return Space(base, learned, mean, std, equal_influence_weight(base, learned))


@lru_cache(maxsize=1)
def load() -> SimilarIndex:
    base = np.load(BASELINE, allow_pickle=False)
    learned = np.load(LEARNED, allow_pickle=False)
    phase_data = json.loads(PHASES.read_text())
    pool = json.loads(POOL.read_text())

    base_rows = {(str(c), str(o)): v for c, o, v in zip(base["cases"], base["organs"], base["features"])}
    learned_rows = {
        (str(c), str(o)): v for c, o, v in zip(learned["cases"], learned["organs"], learned["embeddings"])
    }
    keys = sorted(set(base_rows) & set(learned_rows))

    cases = np.array([k[0] for k in keys])
    organs = np.array([k[1] for k in keys])
    phases = np.array([phase_data.get(c, {}).get("phase") for c in cases], dtype=object)
    names = tuple(str(n) for n in base["feature_names"])

    raw = np.stack([base_rows[k] for k in keys]).astype(np.float64)
    emb = np.stack([learned_rows[k] for k in keys]).astype(np.float64)

    # Baseline: only the intensity columns move, shape stays put.
    base_offsets = fit_phase_offsets(raw, organs, phases, columns=names, min_group=MIN_PHASE_GROUP)
    learned_offsets = fit_phase_offsets(emb, organs, phases, min_group=MIN_PHASE_GROUP)

    raw_centred = apply_phase_offsets(raw, organs, phases, base_offsets, columns=names).astype(np.float64)
    emb_centred = apply_phase_offsets(emb, organs, phases, learned_offsets).astype(np.float64)

    missing = sorted(set(pool) - set(cases))
    if missing:
        raise ValueError(f"pool cases not in the index: {missing}")

    version = f"f{int(base['feature_version'])}-{ENCODER_ID}-p{PREPROCESS_VERSION}-n{len(keys)}"

    return SimilarIndex(
        cases=cases,
        organs=organs,
        phases=phases,
        names=names,
        plain=_fit_space(raw, emb),
        centred=_fit_space(raw_centred, emb_centred),
        base_offsets=base_offsets,
        learned_offsets=learned_offsets,
        pool=pool,
        version=version,
    )


def case_for_study(study_uid: str) -> str | None:
    """The pool case a study came from, if any, so it can leave itself out."""
    for case, entry in load().pool.items():
        if entry["study_uid"] == study_uid:
            return case
    return None


def _unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def search(
    baseline: np.ndarray,
    embedding: np.ndarray,
    organ: str,
    phase: str | None,
    k: int = 5,
    phase_centred: bool = False,
    exclude_case: str | None = None,
) -> list[dict]:
    """Most similar pool organs to a query, best first.

    `baseline` is the raw 11 features, `embedding` the 384-d vector, both
    from describe(). `phase` is the query's own contrast phase.
    """
    index = load()
    raw = np.asarray(baseline, dtype=np.float64).reshape(1, -1)
    emb = np.asarray(embedding, dtype=np.float64).reshape(1, -1)

    if phase_centred:
        space = index.centred
        raw = apply_phase_offsets(raw, [organ], [phase], index.base_offsets, columns=index.names)
        emb = apply_phase_offsets(emb, [organ], [phase], index.learned_offsets)
    else:
        space = index.plain

    query_base = standardise(raw.astype(np.float64), space.mean, space.std)[0]
    query_emb = _unit(emb[0].astype(np.float64))

    keep = (index.organs == organ) & np.isin(index.cases, list(index.pool))
    if exclude_case is not None:
        keep &= index.cases != exclude_case
    rows = np.flatnonzero(keep)

    w = space.weight
    scores = w * (space.base[rows] @ query_base) + (1 - w) * (space.learned[rows] @ query_emb)

    results = []
    for j in np.argsort(-scores)[:k]:
        row = rows[j]
        case = str(index.cases[row])

        # Features that agree most. Covers the baseline half only: DINOv2's
        # 384 dimensions have no names.
        agreement = query_base * space.base[row]
        reasons = [index.names[i] for i in np.argsort(-agreement)[:REASONS] if agreement[i] > 0]

        results.append({
            "case": case,
            "similarity": float(scores[j]),
            "phase": index.phases[row],
            "study_uid": index.pool[case]["study_uid"],
            "series_uid": index.pool[case]["series_uid"],
            "reasons": reasons,
        })

    return results
