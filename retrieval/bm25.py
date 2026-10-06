"""Zone-weighted BM25 ranking with term-at-a-time accumulators."""
import heapq
from collections import Counter

import numpy as np

import config


def _query_weights(terms) -> dict[str, float]:
    """Query term weights: query term frequency for a term list, or the given weights for a weighted query."""
    if isinstance(terms, dict):
        return {t: float(w) for t, w in terms.items() if w}
    return {t: float(c) for t, c in Counter(terms).items()}


def bm25_zone_accumulators(index, terms) -> dict[str, np.ndarray]:
    """Term-at-a-time BM25 accumulators, one dense score array per zone (idf from the body zone)."""
    k1, b = config.K1, config.B
    acc = {}
    for zone in ("title", "body"):
        z = index.zones[zone]
        scores = np.zeros(index.N, dtype=np.float64)
        avg = z.avg_len or 1.0
        for term, qw in _query_weights(terms).items():
            docs, tfs = index.posting_arrays(term, zone)
            if len(docs) == 0:
                continue
            tf = tfs.astype(np.float64)
            norm = k1 * (1.0 - b + b * z.doc_len[docs] / avg)
            scores[docs] += qw * index.idf(term) * tf * (k1 + 1.0) / (tf + norm)
        acc[zone] = scores
    return acc


def bm25_scores(index, terms) -> np.ndarray:
    """Dense zone-weighted BM25 score per document: W_TITLE*bm25_title + W_BODY*bm25_body."""
    acc = bm25_zone_accumulators(index, terms)
    return config.W_TITLE * acc["title"] + config.W_BODY * acc["body"]


def top_k(scores: np.ndarray, k: int, docs: np.ndarray) -> list[tuple[int, float]]:
    """Top-k selection with a heap; ties broken by lower doc_id."""
    pairs = zip(docs.tolist(), scores[docs].tolist())
    return [(d, s) for d, s in heapq.nlargest(k, pairs, key=lambda p: (p[1], -p[0]))]


def bm25_search(index, terms: list[str], k: int, candidates: set[int] | None = None) -> list[tuple[int, float]]:
    """BM25 ranked retrieval: W_TITLE*bm25_title + W_BODY*bm25_body, top-k via heap.

    `terms` may also be a dict term -> weight (weighted query, e.g. after feedback). Without `candidates`
    only docs matching at least one term are returned; with `candidates` every candidate is scored (zeros kept).
    """
    scores = bm25_scores(index, terms)
    if candidates is None:
        docs = np.flatnonzero(scores > 0)
    else:
        docs = np.fromiter(sorted(candidates), dtype=np.int64, count=len(candidates))
    return top_k(scores, k, docs)
