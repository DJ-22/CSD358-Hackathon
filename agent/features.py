"""IR features for the learned bridge ranker."""
import numpy as np

from agent.entities import tokenize
from retrieval.bm25 import bm25_scores

FEATURE_NAMES = [
    "tfidf_in_source", "src_hop1_score", "src_hop1_rank", "mean_idf", "n_src_docs", "in_title_zone",
    "first_pos", "min_dist_consumed", "n_tokens", "lookahead_sparse", "lookahead_dense", "hand_score",
]
MAX_DIST = 50


def _spans(tokens: list[str], ent: list[str]) -> list[int]:
    """Start offsets of every occurrence of an entity token sequence in a token sequence (phrase match)."""
    n = len(ent)
    return [i for i in range(len(tokens) - n + 1) if tokens[i:i + n] == ent]


class QuestionContext:
    """Per-question feature context: residual BM25 scores, residual query vector and source-doc token streams."""

    def __init__(self, state, res):
        self.res = res
        index = res.index
        self.consumed = set(state.consumed_terms)
        terms = list(state.residual_terms)
        self.sparse = bm25_scores(index, terms) if terms else np.zeros(index.N)
        self.sparse_max = float(self.sparse.max()) if terms else 0.0
        self.qvec = None
        if res.emb is not None:
            from index.embed import encode_query
            self.qvec = encode_query(state.residual_text or state.question)
        self._docs: dict[int, tuple] = {}

    def doc(self, doc_id: int) -> tuple[list[str], list[str], list[bool]]:
        """Unstemmed title tokens, body tokens and a consumed-term mask over the body (positional view of a doc)."""
        if doc_id not in self._docs:
            index = self.res.index
            title = tokenize(index.doc_title(doc_id))
            body = tokenize(index.doc_text(doc_id))
            consumed = [bool(t) and t[0] in self.consumed for t in map(index.analyze, body)]
            self._docs[doc_id] = (title, body, consumed)
        return self._docs[doc_id]


def _min_dist(starts: list[int], n: int, mask: list[bool]) -> float:
    """Proximity: min token distance between an entity occurrence and a consumed query term (capped)."""
    best = MAX_DIST
    hits = [p for p, m in enumerate(mask) if m]
    for s in starts:
        for p in hits:
            if s <= p < s + n:
                continue
            d = s - p if p < s else p - (s + n - 1)
            best = min(best, d)
    return float(best)


def features_with(ctx: QuestionContext, cand: dict) -> list[float]:
    """Feature vector for one bridge candidate given a per-question context."""
    res = ctx.res
    ent = cand["entity"].split()
    src = next(s for s in cand["sources"] if s["doc"] == cand["source_doc"])
    in_title = 0.0
    for s in cand["sources"]:
        if _spans(ctx.doc(s["doc"])[0], ent):
            in_title = 1.0
            break
    _, body, mask = ctx.doc(cand["source_doc"])
    starts = _spans(body, ent)
    first_pos = starts[0] / len(body) if starts else 1.0
    e_doc = cand["entity_doc"]
    look_sparse = float(ctx.sparse[e_doc]) / ctx.sparse_max if ctx.sparse_max > 0 else 0.0
    look_dense = float(res.emb[e_doc].astype(np.float32) @ ctx.qvec) if ctx.qvec is not None else 0.0
    return [
        src["tfidf"],                        # tfidf_in_source
        src["w"],                            # src_hop1_score
        float(src["rank"]),                  # src_hop1_rank
        cand["mean_idf"],                    # mean_idf
        float(len(cand["sources"])),         # n_src_docs
        in_title,                            # in_title_zone
        first_pos,                           # first_pos
        _min_dist(starts, len(ent), mask),   # min_dist_consumed
        float(len(ent)),                     # n_tokens
        look_sparse,                         # lookahead_sparse
        look_dense,                          # lookahead_dense
        cand["hand_score"],                  # hand_score
    ]


def candidate_matrix(state, cands: list[dict], res) -> np.ndarray:
    """Feature matrix (one row per bridge candidate, columns in FEATURE_NAMES order) for one question."""
    if not cands:
        return np.zeros((0, len(FEATURE_NAMES)))
    ctx = QuestionContext(state, res)
    return np.array([features_with(ctx, c) for c in cands], dtype=np.float64)


def candidate_features(state, cand: dict, hop1_results, res) -> list[float]:
    """Feature vector for one bridge candidate (tf-idf, rank, zone, proximity, lookahead)."""
    return features_with(QuestionContext(state, res), cand)
