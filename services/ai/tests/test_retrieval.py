"""Search behaviour, on an index small enough to reason about."""

from __future__ import annotations

import numpy as np
import pytest

from app.retrieval import Index, search, standardise

NAMES = ("volume_ml", "mean_hu")


def make_index(rows) -> Index:
    """rows: (case, organ, volume_ml, mean_hu)."""
    raw = np.array([[r[2], r[3]] for r in rows], dtype=np.float64)
    mean = raw.mean(axis=0)
    std = raw.std(axis=0)
    std[std == 0] = 1.0

    return Index(
        cases=np.array([r[0] for r in rows]),
        organs=np.array([r[1] for r in rows]),
        raw=raw,
        unit=standardise(raw, mean, std),
        mean=mean,
        std=std,
        names=NAMES,
        feature_version=1,
    )


@pytest.fixture
def index():
    return make_index([
        ("a", "liver", 1400.0, 60.0),
        ("b", "liver", 1450.0, 62.0),
        ("c", "liver", 3200.0, 61.0),   # enlarged
        ("d", "spleen", 180.0, 90.0),
        ("e", "spleen", 175.0, 88.0),
    ])


def test_standardising_gives_unit_vectors(index):
    lengths = np.linalg.norm(index.unit, axis=1)
    assert np.allclose(lengths, 1.0)


def test_a_feature_with_no_spread_does_not_divide_by_zero():
    # Every volume identical: the feature carries nothing and must not blow up.
    index = make_index([("a", "liver", 1000.0, 50.0), ("b", "liver", 1000.0, 90.0)])

    assert np.isfinite(index.unit).all()


def test_organ_filter_restricts_results(index):
    hits = search(index, index.raw[0], k=5, organ="spleen")

    assert {organ for _, organ, _ in hits} == {"spleen"}


def test_excluded_case_is_absent(index):
    hits = search(index, index.raw[0], k=5, exclude_case="a")

    assert "a" not in {case for case, _, _ in hits}


def test_similar_ranks_above_dissimilar(index):
    # b is a near twin of a; c is the enlarged one.
    hits = search(index, index.raw[0], k=3, organ="liver", exclude_case="a")
    order = [case for case, _, _ in hits]

    assert order[0] == "b"
    assert order.index("b") < order.index("c")


def test_results_are_ordered_by_descending_similarity(index):
    scores = [score for _, _, score in search(index, index.raw[0], k=5)]

    assert scores == sorted(scores, reverse=True)


def test_standardising_stops_one_feature_dominating():
    # Volume spans thousands, density spans tens. Without z-scoring, two rows
    # that differ wildly in density but share a volume would look identical.
    index = make_index([
        ("a", "liver", 1500.0, 20.0),
        ("b", "liver", 1500.0, 140.0),
        ("c", "liver", 1400.0, 22.0),
    ])

    hits = search(index, index.raw[0], k=2, organ="liver", exclude_case="a")

    # c matches on both; b matches on volume alone and must rank below it.
    assert hits[0][0] == "c"


def test_k_caps_the_result_count(index):
    assert len(search(index, index.raw[0], k=2)) == 2


def test_fused_cosine_is_the_weighted_blend():
    from app.retrieval import fuse

    rng = np.random.default_rng(0)
    a = rng.normal(size=(5, 3)); a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = rng.normal(size=(5, 8)); b /= np.linalg.norm(b, axis=1, keepdims=True)

    f = fuse(a, b, 0.3)

    assert np.allclose(np.linalg.norm(f, axis=1), 1.0)
    assert np.allclose(f @ f.T, 0.3 * (a @ a.T) + 0.7 * (b @ b.T))


def test_bunched_cosines_get_more_weight():
    from app.retrieval import equal_influence_weight

    rng = np.random.default_rng(1)
    spread = rng.normal(size=(40, 4)); spread /= np.linalg.norm(spread, axis=1, keepdims=True)
    # all rows close to one direction, so their cosines barely vary
    bunched = np.ones((40, 4)) + 0.05 * rng.normal(size=(40, 4))
    bunched /= np.linalg.norm(bunched, axis=1, keepdims=True)

    # weight is the first set's share; the spread-out set should get the smaller share
    assert equal_influence_weight(spread, bunched) < 0.5
