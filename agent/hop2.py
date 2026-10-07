"""Hop-2 retrieval: BM25 mode and hybrid sparse-filter -> dense-score mode."""
import sys, pathlib  # bootstrap: make the repo root importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from functools import lru_cache

import numpy as np

import config
from agent.entities import title_surface
from agent.state import ResearchState
from retrieval.bm25 import bm25_scores, top_k


def hop2_retrieve(entity: str, state: ResearchState, res, mode: str, alpha: float, exclude: int | None
                  ) -> list[tuple[int, float, float, float]]:
    """Entity-conditioned second-hop retrieval returning (doc, s_sparse, s_dense, s_B)."""
    if mode == "bm25":
        return _hop2_bm25(entity, state, res, exclude)
    if mode == "hybrid":
        return _hop2_hybrid(entity, state, res, alpha, exclude)
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


def _minmax(x: np.ndarray) -> np.ndarray:
    """Min-max score normalisation to [0, 1] (all-equal scores map to 1)."""
    if len(x) == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    return np.ones_like(x) if hi == lo else (x - lo) / (hi - lo)


@lru_cache(maxsize=4096)
def _query_vec(text: str) -> np.ndarray:
    """Dense query vector (cached: one residual is reused for every bridge of a question)."""
    from index.embed import encode_query
    return encode_query(text)


def sparse_filter(entity: str, state: ResearchState, res, exclude: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Sparse candidate filter: docs with the entity phrase in title or body, capped by BM25(residual), minus exclude.

    Returns (candidate doc_ids, their raw BM25(residual) scores); with an empty residual the entity terms are the query.
    """
    index = res.index
    words = entity.split()
    cand = index.phrase_docs(words, "title") | index.phrase_docs(words, "body")
    cand.discard(exclude)
    if not cand:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    scores = bm25_scores(index, list(state.residual_terms) or index.analyze(entity))
    docs = np.fromiter(sorted(cand), dtype=np.int64, count=len(cand))
    if len(docs) > config.HOP2_FILTER_CAP:
        docs = np.array(sorted(d for d, _ in top_k(scores, config.HOP2_FILTER_CAP, docs)), dtype=np.int64)
    return docs, scores[docs]


def _hop2_hybrid(entity: str, state: ResearchState, res, alpha: float, exclude: int | None
                 ) -> list[tuple[int, float, float, float]]:
    """Hybrid hop 2: s_B = alpha*minmax(BM25) + (1-alpha)*cos(residual, doc) + TITLE_BONUS*[title surface == entity]."""
    if res.emb is None:
        raise RuntimeError(f"hybrid hop 2 needs document embeddings: run python index/embed.py --corpus {res.corpus}")
    docs, sparse_raw = sparse_filter(entity, state, res, exclude)
    if len(docs) == 0:
        return []
    s_sparse = _minmax(sparse_raw)
    s_dense = res.emb[docs].astype(np.float32) @ _query_vec(state.residual_text or state.question)
    bonus = np.array([title_surface(res.index.doc_title(int(d))) == entity for d in docs], dtype=np.float64)
    s_b = alpha * s_sparse + (1.0 - alpha) * s_dense + config.TITLE_BONUS * bonus
    order = sorted(range(len(docs)), key=lambda i: (-s_b[i], int(docs[i])))[:config.HOP2_TOP]
    return [(int(docs[i]), float(s_sparse[i]), float(s_dense[i]), float(s_b[i])) for i in order]


def main() -> None:
    """Oracle diagnostic: hop 2 given the true hop-2 entity on TUNE bridge questions, BM25 vs hybrid over alpha."""
    import json
    import time
    from agent.bridge import residual_query
    from agent.parser import hop1, parse
    from agent.resources import load_resources
    from eval.run_eval import build_qrels, load_hop2_gold, load_questions, load_split

    res = load_resources("dev", with_ranker=False)
    qrecs = load_questions()
    qrels = build_qrels(qrecs, res.title2doc)
    hop2_gold = load_hop2_gold(res, qrecs, qrels)
    qids = [q for q in load_split("tune") if qrecs[q]["type"] == "bridge"]

    settings = [("bm25", None)] + [("hybrid", a) for a in config.ALPHA_GRID]
    hits = {s: [0, 0] for s in settings}
    in_filter = n_cand = 0
    t_hybrid = 0.0
    for qid in qids:
        state = parse(qrecs[qid]["question"], res, qid)
        residual_query(state, hop1(state, res), res)
        gold = hop2_gold[qid]
        entity = title_surface(res.index.doc_title(gold))
        docs, _ = sparse_filter(entity, state, res, None)
        in_filter += gold in set(docs.tolist())
        n_cand += len(docs)
        for mode, alpha in settings:
            t = time.perf_counter()
            out = [d for d, *_ in hop2_retrieve(entity, state, res, mode, alpha if alpha is not None else 0.0, None)]
            if mode == "hybrid":
                t_hybrid += time.perf_counter() - t
            hits[(mode, alpha)][0] += out[:1] == [gold]
            hits[(mode, alpha)][1] += gold in out[:3]

    n = len(qids)
    lines = ["# Hop-2 oracle diagnostic (TUNE, bridge questions)", "",
             f"Hop 2 is given the true hop-2 gold title surface as the bridge entity and the parser's residual query, "
             f"on all {n} bridge questions of TUNE (exclude = none). Recall of the hop-2 gold doc:", "",
             "| hop-2 mode | alpha | recall@1 | recall@3 |", "|---|---|---|---|"]
    for mode, alpha in settings:
        r1, r3 = hits[(mode, alpha)]
        lines.append(f"| {mode} | {'-' if alpha is None else alpha} | {r1 / n:.3f} | {r3 / n:.3f} |")
    lines += ["", f"Sparse filter: gold doc inside the phrase-filtered candidate set for {in_filter / n:.3f} of questions; "
              f"mean candidate set size {n_cand / n:.1f} (cap {config.HOP2_FILTER_CAP}). "
              f"Title bonus {config.TITLE_BONUS}.",
              f"Hybrid hop 2 runtime: {1000 * t_hybrid / (n * len(config.ALPHA_GRID)):.1f} ms per call.", "",
              "Generated by `python agent/hop2.py`."]
    out = config.RESULTS_DIR / "hop2_oracle.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
