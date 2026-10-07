"""Title-anchored bridge extraction and residual query (entity-restricted feedback, query consumption)."""
import config
from agent.entities import tokenize


def minmax(values: list[float]) -> list[float]:
    """Min-max score normalisation to [0, 1] (all-equal scores map to 1)."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def entity_terms(surface: str, res) -> list[str]:
    """Analyse an entity surface into index terms (stop words removed, stemmed like the index)."""
    return res.index.analyze(surface)


def mean_idf(surface: str, res) -> float:
    """Mean idf of an entity's index terms (0 if it has none)."""
    terms = entity_terms(surface, res)
    return sum(res.index.idf(t) for t in terms) / len(terms) if terms else 0.0


def entity_doc(surface: str, doc_ids: list[int], res) -> int:
    """The entity's own paragraph: the doc whose full title equals the surface, else the lowest doc_id."""
    exact = [d for d in doc_ids if " ".join(tokenize(res.index.doc_title(d))) == surface]
    return min(exact) if exact else min(doc_ids)


def residual_query(state, hop1_results, res) -> None:
    """Query consumption: drop query terms matched by the top hop-1 doc or a question entity; fill residual fields."""
    index = res.index
    entity_vocab = {t for e in state.known_entities for t in entity_terms(e, res)}
    d_star = hop1_results[0][0] if hop1_results else None
    consumed = set()
    for t in state.query_terms:
        in_top = d_star is not None and (index.doc_term_tf(d_star, t, "title") > 0
                                         or index.doc_term_tf(d_star, t, "body") > 0)
        if in_top or t in entity_vocab:
            consumed.add(t)
    residual = [t for t in state.query_terms if t not in consumed]
    if not residual:
        residual = [t for t in state.query_terms if t not in entity_vocab]
    state.consumed_terms = consumed
    state.residual_terms = residual
    keep = set(residual)
    words = []
    for tok in tokenize(state.question):
        terms = index.analyze(tok)
        if terms and terms[0] in keep:
            words.append(tok)
    state.residual_text = " ".join(words)


def bridge_candidates(state, hop1_results, res) -> list[dict]:
    """Entity-restricted pseudo-relevance feedback; also fills residual (unconsumed) query fields in state.

    Candidates are title entities linked in the top-M hop-1 docs, scored by
    bridge(e) = sum_d w(d) * tfidf(e, d) with w(d) the min-max normalised hop-1 score.
    Returns every candidate sorted by hand score (desc); callers keep the top BEAM.
    """
    residual_query(state, hop1_results, res)
    top = hop1_results[:config.TOP_M]
    weights = minmax([s for _, s in top])
    cands: dict[str, dict] = {}
    for rank, ((doc, score), w) in enumerate(zip(top, weights), start=1):
        own = res.linker.doc2surface.get(doc)
        counts: dict[str, int] = {}
        doc_ids: dict[str, list[int]] = {}
        for surface, ids, _ in res.linker.link(res.index.doc_title(doc) + " " + res.index.doc_text(doc)):
            if surface in state.known_entities or surface == own:
                continue
            counts[surface] = counts.get(surface, 0) + 1
            doc_ids[surface] = ids
        for surface, tf in counts.items():
            c = cands.get(surface)
            if c is None:
                idf = mean_idf(surface, res)
                c = cands[surface] = {"entity": surface, "doc_ids": doc_ids[surface],
                                      "entity_doc": entity_doc(surface, doc_ids[surface], res),
                                      "mean_idf": idf, "sources": [], "hand_score": 0.0,
                                      "source_doc": None, "p_e": None, "features": None}
            tfidf = tf * c["mean_idf"]
            contrib = w * tfidf
            c["sources"].append({"doc": doc, "rank": rank, "w": w, "tf": tf, "tfidf": tfidf})
            c["hand_score"] += contrib
            best = c["source_doc"]
            if best is None or contrib > c["_best"]:
                c["source_doc"], c["_best"] = doc, contrib
    out = sorted(cands.values(), key=lambda c: (-c["hand_score"], c["entity"]))
    for c in out:
        del c["_best"]
    return out


def hand_bridges(state, cands: list[dict], k: int = config.BEAM) -> list[tuple[str, float, int]]:
    """Rule-based bridge selection: top-k candidates by hand score, p_e = hand score share; recorded in state.bridges."""
    total = sum(c["hand_score"] for c in cands)
    out = []
    for c in cands[:k]:
        p_e = c["hand_score"] / total if total > 0 else 1.0 / len(cands)
        state.bridges.append({"entity": c["entity"], "source_doc": c["source_doc"],
                              "hand_score": c["hand_score"], "p_e": p_e, "features": c["features"]})
        out.append((c["entity"], p_e, c["source_doc"]))
    return out
