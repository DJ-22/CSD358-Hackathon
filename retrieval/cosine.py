"""lnc.ltc cosine vector-space baseline over the title + body bag of words."""
import math
from collections import Counter

import numpy as np

from retrieval.bm25 import top_k


def cosine_search(index, terms: list[str], k: int) -> list[tuple[int, float]]:
    """Vector space model ranking with lnc.ltc weighting and cosine similarity."""
    qtf = Counter(terms)
    q = {}
    for term, c in qtf.items():
        t = index.term_id(term)
        if t < 0 or index.df_any[t] == 0:
            continue
        q[term] = (1.0 + math.log(c)) * math.log(index.N / index.df_any[t])   # ltc: log tf x idf
    q_norm = math.sqrt(sum(w * w for w in q.values()))
    if q_norm == 0:
        return []
    scores = np.zeros(index.N, dtype=np.float64)
    tf_any = np.zeros(index.N, dtype=np.float64)
    for term, wq in q.items():
        tf_any[:] = 0.0
        for zone in ("title", "body"):
            docs, tfs = index.posting_arrays(term, zone)
            tf_any[docs] += tfs
        docs = np.flatnonzero(tf_any)
        scores[docs] += (wq / q_norm) * (1.0 + np.log(tf_any[docs])) / index.cos_norm[docs]  # lnc: log tf, cosine norm
    return top_k(scores, k, np.flatnonzero(scores > 0))
