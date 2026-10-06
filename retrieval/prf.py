"""Rocchio-style pseudo-relevance feedback (PLAN.md §3)."""


def prf_expand(index, query_terms: dict[str, float], top_docs: list[int], n_terms: int, beta: float) -> dict[str, float]:
    """Pseudo-relevance feedback: add top-n tf-idf terms of the top docs with weight beta."""
    return dict(query_terms)
