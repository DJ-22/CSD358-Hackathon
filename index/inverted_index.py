"""Positional zone inverted index with title/body zones, stored as compressed-sparse-row arrays."""
import math
import pickle

import numpy as np

import config
from index.preprocess import analyze, analyze_with_positions

ZONES = ("title", "body")


class _Zone:
    """One zone of the index: positional postings (term -> docs, tf, positions) plus a forward index (doc -> terms, tf)."""

    def __init__(self, ptr, docs, tfs, pos_ptr, pos, fptr, fterms, ftfs, doc_len):
        self.ptr, self.docs, self.tfs = ptr, docs, tfs              # postings of term t: [ptr[t], ptr[t+1])
        self.pos_ptr, self.pos = pos_ptr, pos                       # positions of posting i: [pos_ptr[i], pos_ptr[i+1])
        self.fptr, self.fterms, self.ftfs = fptr, fterms, ftfs      # forward entries of doc d: [fptr[d], fptr[d+1])
        self.doc_len = doc_len                                      # # index terms per doc
        self.avg_len = float(doc_len.mean()) if len(doc_len) else 0.0


class InvertedIndex:
    """Positional inverted index over title and body zones with df, N and length statistics."""

    def __init__(self, stem: bool = True):
        self.stem = stem
        self.N = 0
        self.vocab: dict[str, int] = {}
        self.terms: list[str] = []
        self.titles: list[str] = []
        self.texts: list[str] = []
        self.zones: dict[str, _Zone] = {}
        self.df_any = np.zeros(0, dtype=np.int32)       # df over title ∪ body
        self.idf_body = np.zeros(0, dtype=np.float64)   # BM25 idf per term id
        self.cos_norm = np.zeros(0, dtype=np.float64)   # lnc document vector lengths (title + body)

    # --- analysis -------------------------------------------------------------
    def analyze(self, text: str) -> list[str]:
        """Analyse query text the same way the index was built (stemmed or not)."""
        return analyze(text, self.stem)

    def term_id(self, term: str) -> int:
        """Vocabulary lookup: term -> term id, or -1 if the term is not indexed."""
        return self.vocab.get(term, -1)

    # --- postings -------------------------------------------------------------
    def posting_arrays(self, term: str, zone: str) -> tuple[np.ndarray, np.ndarray]:
        """Postings list as parallel arrays (doc_ids, tfs), sorted by doc_id."""
        t = self.term_id(term)
        if t < 0:
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
        z = self.zones[zone]
        a, b = z.ptr[t], z.ptr[t + 1]
        return z.docs[a:b], z.tfs[a:b]

    def postings(self, term: str, zone: str) -> list[tuple[int, int, list[int]]]:
        """Postings list (doc_id, tf, positions) sorted by doc_id."""
        t = self.term_id(term)
        if t < 0:
            return []
        z = self.zones[zone]
        out = []
        for i in range(z.ptr[t], z.ptr[t + 1]):
            positions = z.pos[z.pos_ptr[i]:z.pos_ptr[i + 1]].tolist()
            out.append((int(z.docs[i]), int(z.tfs[i]), positions))
        return out

    def positions(self, doc_id: int, term: str, zone: str) -> np.ndarray:
        """Positions of a term in one document zone (binary search in the postings list)."""
        t = self.term_id(term)
        if t < 0:
            return np.zeros(0, dtype=np.int32)
        z = self.zones[zone]
        a, b = z.ptr[t], z.ptr[t + 1]
        i = a + int(np.searchsorted(z.docs[a:b], doc_id))
        if i < b and z.docs[i] == doc_id:
            return z.pos[z.pos_ptr[i]:z.pos_ptr[i + 1]]
        return np.zeros(0, dtype=np.int32)

    def df(self, term: str, zone: str) -> int:
        """Document frequency of a term in a zone."""
        t = self.term_id(term)
        if t < 0:
            return 0
        z = self.zones[zone]
        return int(z.ptr[t + 1] - z.ptr[t])

    def idf(self, term: str) -> float:
        """BM25 idf on the body zone: log((N - df + 0.5)/(df + 0.5) + 1)."""
        t = self.term_id(term)
        if t < 0:
            return math.log((self.N + 0.5) / 0.5 + 1)
        return float(self.idf_body[t])

    def phrase_docs(self, tokens: list[str], zone: str) -> set[int]:
        """Phrase query via positional postings intersection; tokens are raw words, stop words keep their slot."""
        analysed = analyze_with_positions(" ".join(tokens), self.stem)
        if not analysed:
            return set()
        first_pos = analysed[0][1]
        offsets = [(term, pos - first_pos) for term, pos in analysed]
        lists = [self.posting_arrays(term, zone)[0] for term, _ in offsets]
        if any(len(docs) == 0 for docs in lists):
            return set()
        cand = min(lists, key=len)
        for docs in lists:          # positional match is only checked on docs containing every term
            idx = np.searchsorted(docs, cand)
            idx[idx == len(docs)] = 0
            cand = cand[docs[idx] == cand]
        out = set()
        for d in cand.tolist():
            starts = set(self.positions(d, offsets[0][0], zone).tolist())
            for term, off in offsets[1:]:
                starts &= {p - off for p in self.positions(d, term, zone).tolist()}
                if not starts:
                    break
            if starts:
                out.add(d)
        return out

    # --- forward index --------------------------------------------------------
    def doc_terms(self, doc_id: int, zone: str) -> dict[str, int]:
        """Forward index: term -> tf for one document zone."""
        z = self.zones[zone]
        a, b = z.fptr[doc_id], z.fptr[doc_id + 1]
        return {self.terms[t]: int(tf) for t, tf in zip(z.fterms[a:b].tolist(), z.ftfs[a:b].tolist())}

    def doc_term_tf(self, doc_id: int, term: str, zone: str) -> int:
        """Term frequency of a term in one document zone (forward index lookup)."""
        t = self.term_id(term)
        if t < 0:
            return 0
        z = self.zones[zone]
        a, b = z.fptr[doc_id], z.fptr[doc_id + 1]
        i = a + int(np.searchsorted(z.fterms[a:b], t))
        return int(z.ftfs[i]) if i < b and z.fterms[i] == t else 0

    def doc_len(self, doc_id: int, zone: str) -> int:
        """Document length (# index terms) in a zone."""
        return int(self.zones[zone].doc_len[doc_id])

    def avg_len(self, zone: str) -> float:
        """Average document length in a zone."""
        return self.zones[zone].avg_len

    def doc_text(self, doc_id: int) -> str:
        """Raw body text of a document."""
        return self.texts[doc_id]

    def doc_title(self, doc_id: int) -> str:
        """Raw title of a document."""
        return self.titles[doc_id]


def _build_zone(entries_term, entries_doc, entries_tf, entries_pos_start, all_pos, N: int, V: int) -> _Zone:
    """Invert (doc, term, tf, positions) entries into CSR postings and a term-sorted CSR forward index."""
    term = np.asarray(entries_term, dtype=np.int32)
    doc = np.asarray(entries_doc, dtype=np.int32)
    tf = np.asarray(entries_tf, dtype=np.int32)
    pstart = np.asarray(entries_pos_start, dtype=np.int64)
    pos = np.asarray(all_pos, dtype=np.int32)

    def gather_positions(order):
        """Reorder the flat position array to follow the given entry order."""
        lens = tf[order].astype(np.int64)
        new_ptr = np.zeros(len(order) + 1, dtype=np.int64)
        np.cumsum(lens, out=new_ptr[1:])
        idx = np.repeat(pstart[order], lens) + (np.arange(new_ptr[-1]) - np.repeat(new_ptr[:-1], lens))
        return new_ptr, pos[idx]

    # postings: entries grouped by term, doc order kept (stable sort; entries were produced in doc order)
    order = np.argsort(term, kind="stable")
    ptr = np.zeros(V + 1, dtype=np.int64)
    np.cumsum(np.bincount(term, minlength=V), out=ptr[1:])
    pos_ptr, pos_sorted = gather_positions(order)

    # forward index: entries grouped by doc, term ids ascending inside each doc (binary-searchable)
    forder = np.lexsort((term, doc))
    fptr = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(np.bincount(doc, minlength=N), out=fptr[1:])
    doc_len = np.bincount(doc, weights=tf, minlength=N).astype(np.int32)
    return _Zone(ptr, doc[order], tf[order], pos_ptr, pos_sorted, fptr, term[forder], tf[forder], doc_len)


def build(docs: list[dict], stem: bool = True) -> InvertedIndex:
    """Index construction: analyse every doc zone, invert into positional postings, compute idf and lnc norms."""
    idx = InvertedIndex(stem)
    docs = sorted(docs, key=lambda d: d["doc_id"])
    assert [d["doc_id"] for d in docs] == list(range(len(docs))), "doc_ids must be 0..N-1"
    idx.N = len(docs)
    idx.titles = [d["title"] for d in docs]
    idx.texts = [d["text"] for d in docs]
    raw = {z: ([], [], [], [], []) for z in ZONES}      # term, doc, tf, pos_start, positions
    vocab = idx.vocab
    for d in docs:
        for zone, text in (("title", d["title"]), ("body", d["text"])):
            e_term, e_doc, e_tf, e_start, e_pos = raw[zone]
            grouped: dict[str, list[int]] = {}
            for term, p in analyze_with_positions(text, stem):
                grouped.setdefault(term, []).append(p)
            for term, plist in grouped.items():
                t = vocab.setdefault(term, len(vocab))
                e_term.append(t)
                e_doc.append(d["doc_id"])
                e_tf.append(len(plist))
                e_start.append(len(e_pos))
                e_pos.extend(plist)
    idx.terms = [None] * len(vocab)
    for term, t in vocab.items():
        idx.terms[t] = term
    V = len(vocab)
    for zone in ZONES:
        idx.zones[zone] = _build_zone(*raw[zone], N=idx.N, V=V)
        raw[zone] = None

    body = idx.zones["body"]
    df_body = np.diff(body.ptr)
    idx.idf_body = np.log((idx.N - df_body + 0.5) / (df_body + 0.5) + 1)

    # title ∪ body bag of words: df over either zone and lnc vector lengths (for the cosine baseline)
    t_z, b_z = idx.zones["title"], idx.zones["body"]
    docs_cat = np.concatenate([np.repeat(np.arange(idx.N), np.diff(t_z.fptr)), np.repeat(np.arange(idx.N), np.diff(b_z.fptr))])
    keys = docs_cat.astype(np.int64) * V + np.concatenate([t_z.fterms, b_z.fterms])
    uniq, inv = np.unique(keys, return_inverse=True)
    tf_any = np.bincount(inv, weights=np.concatenate([t_z.ftfs, b_z.ftfs]))
    idx.df_any = np.bincount(uniq % V, minlength=V).astype(np.int32)
    w = 1.0 + np.log(tf_any)
    idx.cos_norm = np.sqrt(np.bincount(uniq // V, weights=w * w, minlength=idx.N))
    return idx


def index_path(corpus: str, variant: str = ""):
    """Path of a pickled index file."""
    suffix = f"_{variant}" if variant else ""
    return config.INDEX_DIR / f"index_{corpus}{suffix}.pkl"


def save_index(idx: InvertedIndex, corpus: str, variant: str = "") -> None:
    """Persist an index to disk."""
    with open(index_path(corpus, variant), "wb") as f:
        pickle.dump(idx, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_index(corpus: str, variant: str = "") -> InvertedIndex:
    """Load a pickled inverted index (variant "" or "nostem")."""
    with open(index_path(corpus, variant), "rb") as f:
        return pickle.load(f)
