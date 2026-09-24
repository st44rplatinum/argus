"""Find similar organs by cosine over standardised features.

Flat search, no approximate index. 5372 rows against 11 features is a matrix
multiply that finishes before a network round trip would — the similarity
plugin proved flat holds well past the scale people assume, and hnswlib can
wait until N forces it.

The one thing that is not optional is standardisation. Raw, `volume_ml` has a
standard deviation of 529 and `heterogeneity` has 0.08, so a cosine over raw
vectors is a cosine over volume with rounding noise attached. Z-scoring each
feature first is what makes the other ten count.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_INDEX = Path("data/derived/organ_features.npz")


@dataclass(frozen=True)
class Index:
    cases: np.ndarray        # (n,) case ids
    organs: np.ndarray       # (n,) organ names
    raw: np.ndarray          # (n, LEN) features as computed
    unit: np.ndarray         # (n, LEN) standardised and unit length
    mean: np.ndarray         # (LEN,) corpus mean, for standardising a query
    std: np.ndarray          # (LEN,) corpus sd
    names: tuple[str, ...]
    feature_version: int

    def __len__(self) -> int:
        return len(self.cases)


def standardise(raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Z-score, then unit length so similarity is a dot product."""
    z = (np.atleast_2d(raw) - mean) / std
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    # A row exactly at the corpus mean has no direction; leave it at zero
    # rather than dividing by it.
    return np.divide(z, norms, out=np.zeros_like(z), where=norms > 0)


def load_index(path: Path = DEFAULT_INDEX) -> Index:
    data = np.load(path, allow_pickle=False)
    raw = data["features"].astype(np.float64)

    mean = raw.mean(axis=0)
    # A constant feature would divide by zero and contributes nothing anyway.
    std = raw.std(axis=0)
    std[std == 0] = 1.0

    return Index(
        cases=np.array([str(c) for c in data["cases"]]),
        organs=np.array([str(o) for o in data["organs"]]),
        raw=raw,
        unit=standardise(raw, mean, std),
        mean=mean,
        std=std,
        names=tuple(str(n) for n in data["feature_names"]),
        feature_version=int(data["feature_version"]),
    )


def search(
    index: Index,
    query: np.ndarray,
    k: int = 10,
    organ: str | None = None,
    exclude_case: str | None = None,
) -> list[tuple[str, str, float]]:
    """Nearest rows to a raw feature vector, as (case, organ, similarity).

    `organ` restricts the search — "find a similar liver" is a more useful
    question than "find something similar". `exclude_case` leaves a case out,
    which is what makes a leave-one-out evaluation honest.
    """
    vector = standardise(query, index.mean, index.std)[0]
    scores = index.unit @ vector

    keep = np.ones(len(index), dtype=bool)
    if organ is not None:
        keep &= index.organs == organ
    if exclude_case is not None:
        keep &= index.cases != exclude_case

    candidates = np.flatnonzero(keep)
    if not len(candidates):
        return []

    ordered = candidates[np.argsort(-scores[candidates])[:k]]
    return [
        (str(index.cases[i]), str(index.organs[i]), float(scores[i])) for i in ordered
    ]


def explain(index: Index, a: int, b: int, top: int = 4) -> list[tuple[str, float]]:
    """Which features made two rows similar, most to least.

    The reason for hand-written features rather than a learned vector: when a
    result looks wrong, this says why in words you can check.
    """
    left = standardise(index.raw[a], index.mean, index.std)[0]
    right = standardise(index.raw[b], index.mean, index.std)[0]
    contributions = left * right

    order = np.argsort(-np.abs(contributions))[:top]
    return [(index.names[i], float(contributions[i])) for i in order]
