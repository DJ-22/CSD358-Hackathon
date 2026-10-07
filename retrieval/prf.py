"""Rocchio-style pseudo-relevance feedback."""
from collections import Counter


def prf_expand(index, query_terms: dict[str, float], top_docs: list[int], n_terms: int, beta: float) -> dict[str, float]:
    """Pseudo-relevance feedback: add the top-n tf-idf terms of the top docs with weight beta.

    `query_terms` are query weights (use 1.0 per term for an unweighted query); they are kept unchanged.
    Feedback terms are ranked by their mean tf-idf over the feedback docs (title + body tf, body idf).
    """
    expanded = dict(query_terms)
    if not top_docs or n_terms <= 0:
        return expanded
    centroid = Counter()
    for d in top_docs:
        tf = Counter(index.doc_terms(d, "body"))
        tf.update(index.doc_terms(d, "title"))
        for term, c in tf.items():
            centroid[term] += c * index.idf(term) / len(top_docs)
    ranked = sorted((t for t in centroid if t not in expanded), key=lambda t: (-centroid[t], t))
    for term in ranked[:n_terms]:
        expanded[term] = beta
    return expanded
