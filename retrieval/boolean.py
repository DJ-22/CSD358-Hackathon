"""Boolean retrieval: df-ordered AND, OR union and phrase queries."""
import numpy as np


def _any_zone_docs(index, term: str) -> np.ndarray:
    """Postings list of a term over title ∪ body (sorted doc_ids)."""
    title = index.posting_arrays(term, "title")[0]
    body = index.posting_arrays(term, "body")[0]
    return np.union1d(title, body)


def intersect(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Postings intersection: binary-search each doc of the shorter list in the longer one."""
    if len(a) > len(b):
        a, b = b, a
    if len(a) == 0:
        return a
    i = np.searchsorted(b, a)
    i[i == len(b)] = 0
    return a[b[i] == a]


def boolean_and(index, terms: list[str]) -> tuple[list[int], list[str]]:
    """Boolean AND: intersect postings in increasing df order; returns (docs, df-order used)."""
    uniq = list(dict.fromkeys(terms))
    if not uniq:
        return [], []
    order = sorted(uniq, key=lambda t: (_df_any(index, t), t))
    result = _any_zone_docs(index, order[0])
    for term in order[1:]:
        if len(result) == 0:
            break
        result = intersect(result, _any_zone_docs(index, term))
    return result.tolist(), order


def _df_any(index, term: str) -> int:
    """Document frequency over title ∪ body."""
    t = index.term_id(term)
    return int(index.df_any[t]) if t >= 0 else 0


def boolean_or(index, terms: list[str]) -> list[int]:
    """Boolean OR: union of postings lists."""
    result = np.zeros(0, dtype=np.int32)
    for term in dict.fromkeys(terms):
        result = np.union1d(result, _any_zone_docs(index, term))
    return result.tolist()


def phrase_docs(index, tokens: list[str], zone: str) -> set[int]:
    """Phrase query over the positional index (tokens are raw words; stop words keep their positions)."""
    return index.phrase_docs(tokens, zone)
