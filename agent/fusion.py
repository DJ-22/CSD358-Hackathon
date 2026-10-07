"""Rank fusion: reciprocal rank fusion and CombSUM."""
import config


def _sorted(scores: dict[int, float]) -> list[tuple[int, float]]:
    """Order fused scores descending, ties broken by lower doc_id."""
    return sorted(scores.items(), key=lambda p: (-p[1], p[0]))


def rrf(lists: list[list[tuple[int, float]]], k: int = config.RRF_K) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: score(d) = sum over lists of 1 / (k + rank of d)."""
    scores: dict[int, float] = {}
    for ranking in lists:
        for rank, (doc, _) in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return _sorted(scores)


def combsum(lists: list[list[tuple[int, float]]]) -> list[tuple[int, float]]:
    """CombSUM fusion over min-max normalised scores (all-equal scores in a list normalise to 1)."""
    scores: dict[int, float] = {}
    for ranking in lists:
        if not ranking:
            continue
        vals = [s for _, s in ranking]
        lo, hi = min(vals), max(vals)
        for doc, s in ranking:
            norm = 1.0 if hi == lo else (s - lo) / (hi - lo)
            scores[doc] = scores.get(doc, 0.0) + norm
    return _sorted(scores)
