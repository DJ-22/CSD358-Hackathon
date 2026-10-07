"""Hop-2 retrieval: BM25 mode and hybrid sparse-filter -> dense-score mode."""
import numpy as np

import config
from agent.state import ResearchState
from retrieval.bm25 import bm25_scores, top_k


def hop2_retrieve(entity: str, state: ResearchState, res, mode: str, alpha: float, exclude: int | None
                  ) -> list[tuple[int, float, float, float]]:
    """Entity-conditioned second-hop retrieval returning (doc, s_sparse, s_dense, s_B)."""
    if mode == "bm25":
        return _hop2_bm25(entity, state, res, exclude)
    if mode == "hybrid":
        raise NotImplementedError("hybrid hop 2 is not implemented yet")
    raise ValueError(f"unknown hop-2 mode: {mode}")


def _hop2_bm25(entity: str, state: ResearchState, res, exclude: int | None) -> list[tuple[int, float, float, float]]:
    """BM25 over residual + entity terms on the whole index; s_B = min-max normalised score (min over all docs)."""
    index = res.index
    terms = list(state.residual_terms) + index.analyze(entity)
    if not terms:
        return []
    scores = bm25_scores(index, terms)
    if exclude is not None:
        scores[exclude] = 0.0
    hi, lo = float(scores.max()), float(scores.min())
    if hi <= 0:
        return []
    top = top_k(scores, config.HOP2_TOP, np.flatnonzero(scores > 0))
    return [(d, s, 0.0, (s - lo) / (hi - lo)) for d, s in top]
