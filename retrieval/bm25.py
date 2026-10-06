"""Zone-weighted BM25 ranking with term-at-a-time accumulators (PLAN.md §3)."""


def bm25_search(index, terms: list[str], k: int, candidates: set[int] | None = None) -> list[tuple[int, float]]:
    """BM25 ranked retrieval: W_TITLE*bm25_title + W_BODY*bm25_body, top-k via heap."""
    return []
