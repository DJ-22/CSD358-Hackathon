"""BridgeHop pipeline: single-shot baselines and multi-hop agentic systems S0-S5."""
import config
from agent.bridge import bridge_candidates, hand_bridges, mean_idf, residual_query
from agent.chains import final_ranking, score_chains
from agent.hop2 import hop2_retrieve
from agent.parser import hop1, parse
from agent.state import ResearchState
from retrieval.bm25 import bm25_search
from retrieval.cosine import cosine_search
from retrieval.prf import prf_expand


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
    state.hops.append({"hop": 1, "query": terms, "scorer": scorer, "candidates": len(ranking), "results": ranking})
    state.visited_docs.update(d for d, _ in ranking)
    state.final_ranking = ranking
    return ranking, state


def prf_requery(state: ResearchState, hop1_results: list[tuple[int, float]], res) -> list[tuple[int, float]]:
    """Classic pseudo-relevance feedback: Rocchio-expand the query with the top hop-1 docs and re-query with BM25."""
    top_docs = [d for d, _ in hop1_results[:config.PRF_TOP_M]]
    expanded = prf_expand(res.index, {t: 1.0 for t in state.query_terms}, top_docs, config.PRF_N_TERMS, config.PRF_BETA)
    ranking = bm25_search(res.index, expanded, config.FINAL_K)
    state.hops.append({"hop": "prf", "query": expanded, "feedback_docs": top_docs, "results": ranking})
    state.visited_docs.update(d for d, _ in ranking)
    return ranking


def select_bridges(state: ResearchState, hop1_results, res, cfg) -> list[tuple[str, float, int]]:
    """Bridge selection: hand-scored entity feedback or the learned bridge ranker; returns (entity, p_e, source_doc)."""
    if cfg.get("bridge") == "ltr":
        from agent.bridge_ranker import rank_bridges
        return rank_bridges(state, hop1_results, res, k=config.BEAM)
    return hand_bridges(state, bridge_candidates(state, hop1_results, res), k=config.BEAM)


def comparison_slots(state: ResearchState, res) -> list[str]:
    """Slot entities of a comparison question: the most specific (highest mean idf) linked entities, in question order."""
    ents = [(surface, start) for surface, _, start in res.linker.link(state.question)]
    best = sorted(ents, key=lambda e: (-mean_idf(e[0], res), e[1]))[:config.COMPARISON_SLOTS]
    return [surface for surface, _ in sorted(best, key=lambda e: e[1])]


def hop2_step(entity: str, source_doc: int | None, state: ResearchState, res, cfg) -> None:
    """Run one entity-conditioned second hop and record it in the trace."""
    mode = cfg.get("hop2", "bm25")
    alpha = cfg.get("alpha", config.ALPHA)
    results = hop2_retrieve(entity, state, res, mode=mode, alpha=alpha, exclude=source_doc)
    state.hops.append({"hop": 2, "entity": entity, "source_doc": source_doc, "mode": mode,
                       "query": list(state.residual_terms), "results": results})
    state.visited_docs.update(d for d, _, _, _ in results)
    state.known_entities.add(entity)


def multi_hop(state: ResearchState, hop1_results, res, cfg) -> list[tuple[int, float]]:
    """Agentic two-hop search: bridge (or comparison-slot) entities drive second hops; chains build the final ranking."""
    if state.qtype == "comparison":
        residual_query(state, hop1_results, res)
        for entity in comparison_slots(state, res):
            hop2_step(entity, None, state, res, cfg)
    else:
        for entity, _, source_doc in select_bridges(state, hop1_results, res, cfg):
            hop2_step(entity, source_doc, state, res, cfg)
        score_chains(state, res, cfg)
    return final_ranking(state, cfg)


def run_system(qrec: dict, cfg: dict, res) -> tuple[list[tuple[int, float]], ResearchState]:
    """Multi-hop agentic retrieval for one question under a system configuration."""
    if cfg.get("hops") == 1:
        return single_shot(qrec, cfg, res)
    state = parse(qrec["question"], res, qid=qrec.get("qid", ""))
    ranking = hop1(state, res)
    if cfg.get("prf"):
        ranking = prf_requery(state, ranking, res)
    elif cfg.get("bridge"):
        ranking = multi_hop(state, ranking, res, cfg)
    state.final_ranking = ranking
    return ranking, state
