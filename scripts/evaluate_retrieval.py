"""Does the baseline retrieve anything, or just sort by size?

Two questions, both answerable without labels beyond what the dataset gives.

1. Leave-one-out organ identification. Hide a row, ask for its nearest
   neighbour among every other case's organs, and see whether the neighbour is
   the same organ. Chance is one in six. This is the number the learned
   embedding has to beat.

2. Whether neighbours are matched on contrast phase rather than anatomy. The
   corpus is mostly contrast enhanced, and aortic density is a good proxy for
   phase, so if neighbours agree on aortic HU far more than random pairs do,
   the index is partly a contrast detector.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services" / "ai"))

from app.retrieval import load_index  # noqa: E402


def identification(index) -> tuple[float, dict[str, float], np.ndarray]:
    """Nearest neighbour from a different case — is it the same organ?"""
    scores = index.unit @ index.unit.T

    # A row must not retrieve itself, nor anything from its own case: the
    # other five organs of the same scan share its contrast, its patient and
    # its reconstruction, so matching them would prove nothing.
    same_case = index.cases[:, None] == index.cases[None, :]
    scores[same_case] = -np.inf

    nearest = scores.argmax(axis=1)
    correct = index.organs[nearest] == index.organs

    per_organ = {}
    for organ in sorted(set(index.organs)):
        rows = index.organs == organ
        per_organ[organ] = float(correct[rows].mean())

    return float(correct.mean()), per_organ, nearest


def contrast_leak(index, nearest: np.ndarray) -> tuple[float, float]:
    """Aortic HU agreement between neighbours, against random pairs."""
    aorta_hu: dict[str, float] = {}
    mean_hu = index.names.index("mean_hu")

    for case, organ, row in zip(index.cases, index.organs, index.raw):
        if organ == "aorta":
            aorta_hu[str(case)] = float(row[mean_hu])

    pairs = [
        (aorta_hu[str(index.cases[i])], aorta_hu[str(index.cases[j])])
        for i, j in enumerate(nearest)
        if str(index.cases[i]) in aorta_hu and str(index.cases[j]) in aorta_hu
    ]
    if len(pairs) < 50:
        return float("nan"), float("nan")

    differences = np.array([abs(a - b) for a, b in pairs])

    rng = np.random.default_rng(0)
    values = np.array(list(aorta_hu.values()))
    random_differences = np.abs(
        rng.choice(values, len(differences)) - rng.choice(values, len(differences))
    )

    return float(np.median(differences)), float(np.median(random_differences))


def main() -> None:
    index = load_index()
    print(f"{len(index)} rows, feature version {index.feature_version}")
    print(f"organs: {dict(Counter(index.organs))}")
    print()

    accuracy, per_organ, nearest = identification(index)
    chance = 1.0 / len(set(index.organs))

    print("Leave-one-out organ identification")
    print(f"  chance          {chance:.1%}")
    print(f"  accuracy        {accuracy:.1%}")
    print()
    for organ, value in sorted(per_organ.items(), key=lambda kv: -kv[1]):
        print(f"    {organ:14} {value:6.1%}")
    print()

    neighbour, random = contrast_leak(index, nearest)
    print("Contrast phase leakage (aortic HU difference)")
    print(f"  between neighbours   {neighbour:6.1f} HU")
    print(f"  between random pairs {random:6.1f} HU")
    if neighbour == neighbour:  # not nan
        ratio = neighbour / random if random else float("nan")
        verdict = (
            "neighbours share contrast phase noticeably"
            if ratio < 0.7
            else "no strong contrast matching"
        )
        print(f"  ratio {ratio:.2f} — {verdict}")


if __name__ == "__main__":
    main()
