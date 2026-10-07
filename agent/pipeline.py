"""BridgeHop pipeline: single-shot baselines and multi-hop agentic systems S0-S5."""
import config
from agent.state import ResearchState
from retrieval.bm25 import bm25_search
from retrieval.cosine import cosine_search


def single_shot(qrec: dict, cfg: dict, res) -> tuple[list[tuple[int, float]], ResearchState]:
    """Single-shot ranked retrieval over all analysed question terms (zone BM25 or lnc.ltc cosine)."""
    index = res.index
    state = ResearchState(qid=qrec.get("qid", ""), question=qrec["question"])
    terms = index.analyze(qrec["question"])
    state.query_terms = {t: index.idf(t) for t in terms}
    scorer = cfg.get("scorer", "bm25")
    if scorer == "cosine":
        ranking = cosine_search(index, terms, config.FINAL_K)
    else:
        ranking = bm25_search(index, terms, config.FINAL_K)
    state.hops.append({"query": terms, "scorer": scorer, "candidates": len(ranking), "results": ranking})
    state.visited_docs.update(d for d, _ in ranking)
    state.final_ranking = ranking
    return ranking, state


def run_system(qrec: dict, cfg: dict, res) -> tuple[list[tuple[int, float]], ResearchState]:
    """Multi-hop agentic retrieval for one question under a system configuration."""
    if cfg.get("hops") == 1:
        return single_shot(qrec, cfg, res)
    raise NotImplementedError(f"system configuration not available yet: {cfg}")
