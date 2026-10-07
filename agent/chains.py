"""Reasoning-chain (path) scoring and the final ranking built from chains."""
import config
from agent.bridge import minmax
from agent.fusion import combsum, rrf


def hop_lists(state) -> tuple[list[tuple[int, float]], list[list[tuple[int, float]]]]:
    """The hop-1 ranking and every hop-2 ranking (doc, s_B) recorded in the trace."""
    hop1 = next((h["results"] for h in state.hops if h.get("hop") == 1), [])
    hop2 = [[(d, s_b) for d, _, _, s_b in h["results"]] for h in state.hops if h.get("hop") == 2]
    return hop1, hop2


def score_chains(state, res, cfg) -> list[dict]:
    """Path scoring S = l1*s1(d1) + l2*p(e) + l3*sB(d2), each component min-max normalised within the question."""
    lambdas = cfg.get("lambdas", config.LAMBDAS)
    hop1, _ = hop_lists(state)
    s1_of = {d: s for d, s in hop1}
    raw = []
    for b in state.bridges:
        for h in state.hops:
            if h.get("hop") != 2 or h.get("entity") != b["entity"] or h.get("source_doc") != b["source_doc"]:
                continue
            for d2, _, _, s_b in h["results"]:
                raw.append({"d1": b["source_doc"], "entity": b["entity"], "d2": d2,
                            "s1": s1_of.get(b["source_doc"], 0.0), "p_e": b["p_e"], "sB": s_b})
    if not raw:
        state.chains = []
        return []
    norm = {key: minmax([c[key] for c in raw]) for key in ("s1", "p_e", "sB")}
    for i, c in enumerate(raw):
        c["S"] = lambdas[0] * norm["s1"][i] + lambdas[1] * norm["p_e"][i] + lambdas[2] * norm["sB"][i]
    raw.sort(key=lambda c: (-c["S"], c["d1"], c["d2"]))
    state.chains = raw
    return raw


def _fill(ranking: list[int], fill: list[tuple[int, float]], final_k: int) -> list[int]:
    """Append fused docs not yet ranked until final_k docs are reached."""
    seen = set(ranking)
    for doc, _ in fill:
        if len(ranking) >= final_k:
            break
        if doc not in seen:
            ranking.append(doc)
            seen.add(doc)
    return ranking


def _with_scores(ranking: list[int]) -> list[tuple[int, float]]:
    """Attach a rank-based score 1/rank so scores decrease with position."""
    return [(d, 1.0 / r) for r, d in enumerate(ranking, start=1)]


def chain_first(chains: list[dict], lists: list[list[tuple[int, float]]], final_k: int = config.FINAL_K
                ) -> list[tuple[int, float]]:
    """Chain-first ranking: chains by S (d1 then d2, no duplicates), then RRF over all hop lists up to final_k."""
    ranking: list[int] = []
    for c in chains:
        for doc in (c["d1"], c["d2"]):
            if doc not in ranking and len(ranking) < final_k:
                ranking.append(doc)
    return _with_scores(_fill(ranking, rrf(lists), final_k))


def round_robin(slot_lists: list[list[tuple[int, float]]], lists: list[list[tuple[int, float]]],
                final_k: int = config.FINAL_K) -> list[tuple[int, float]]:
    """Comparison ranking: interleave the per-slot rankings round-robin, then RRF over all hop lists up to final_k."""
    ranking: list[int] = []
    for i in range(max((len(s) for s in slot_lists), default=0)):
        for slot in slot_lists:
            if i < len(slot) and slot[i][0] not in ranking and len(ranking) < final_k:
                ranking.append(slot[i][0])
    return _with_scores(_fill(ranking, rrf(lists), final_k))


def final_ranking(state, cfg, final_k: int = config.FINAL_K) -> list[tuple[int, float]]:
    """Final document ranking from the trace: chain-first (default), or plain RRF / CombSUM over the hop lists."""
    hop1, hop2 = hop_lists(state)
    lists = [hop1] + hop2
    fusion = cfg.get("fusion", "chain")
    if fusion == "rrf":
        return rrf(lists)[:final_k]
    if fusion == "combsum":
        return combsum(lists)[:final_k]
    if state.qtype == "comparison":
        return round_robin(hop2, lists, final_k)
    return chain_first(state.chains, lists, final_k)
