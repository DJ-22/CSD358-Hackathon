"""Tests for Boolean retrieval, BM25 and cosine ranking on a toy corpus."""
import sys, pathlib  # bootstrap: make the repo root importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from index.inverted_index import build
from index.preprocess import analyze
from retrieval.bm25 import bm25_search
from retrieval.boolean import boolean_and, boolean_or, intersect, phrase_docs
from retrieval.cosine import cosine_search
from retrieval.prf import prf_expand
from tests.test_index import DOCS

import numpy as np


@pytest.fixture(scope="module")
def idx():
    return build(DOCS, stem=True)


def test_and_processes_lowest_df_first(idx):
    temple, actress = analyze("temple actress")
    docs, order = boolean_and(idx, [temple, actress])
    assert order == [actress, temple]          # df 1 before df 2
    assert docs == [1]


def test_and_with_missing_term_is_empty(idx):
    docs, order = boolean_and(idx, analyze("temple zzzunseen"))
    assert docs == [] and order[0] == "zzzunseen"


def test_and_covers_both_zones(idx):
    docs, _ = boolean_and(idx, analyze("diplomat"))
    assert docs == [1, 4]


def test_or_union(idx):
    assert boolean_or(idx, analyze("actress film")) == [0, 1, 3]
    assert boolean_or(idx, []) == []


def test_intersect():
    a, b = np.array([1, 3, 5, 9]), np.array([0, 3, 4, 9, 12])
    assert intersect(a, b).tolist() == [3, 9]
    assert intersect(a, np.array([], dtype=np.int64)).tolist() == []


def test_phrase_wrapper(idx):
    assert phrase_docs(idx, ["Shirley", "Temple"], "title") == {1}
    assert phrase_docs(idx, ["Shirley", "Temple"], "body") == {0, 1}


def test_bm25_ranks_obvious_doc_first(idx):
    res = bm25_search(idx, analyze("Shirley Temple actress"), k=3)
    assert res[0][0] == 1
    assert [s for _, s in res] == sorted((s for _, s in res), reverse=True)
    assert len(res) == 2                       # only docs matching a term


def test_bm25_title_zone_weight(idx):
    res = dict(bm25_search(idx, analyze("film"), k=5))
    assert res[3] > res[0]                     # title match on "Film" beats a body-only match


def test_bm25_candidates_keep_zero_scores(idx):
    res = bm25_search(idx, analyze("actress"), k=10, candidates={1, 2})
    assert res[0][0] == 1 and res[1] == (2, 0.0)


def test_bm25_weighted_query(idx):
    temple, actress = analyze("temple actress")
    assert bm25_search(idx, {temple: 1.0, actress: 0.0}, k=5) == bm25_search(idx, [temple], k=5)


def test_prf_adds_feedback_terms_with_beta(idx):
    q = {t: 1.0 for t in analyze("actress")}
    exp = prf_expand(idx, q, top_docs=[1], n_terms=2, beta=0.5)
    assert exp[analyze("actress")[0]] == 1.0                   # original weight kept
    added = {t: w for t, w in exp.items() if t not in q}
    assert len(added) == 2 and set(added.values()) == {0.5}
    assert analyze("temple")[0] in added                       # tf 3 in doc 1 (title + body): highest tf-idf
    assert prf_expand(idx, q, top_docs=[], n_terms=2, beta=0.5) == q


def test_cosine_ranks_obvious_doc_first(idx):
    res = cosine_search(idx, analyze("Shirley Temple actress"), k=3)
    assert res[0][0] == 1
    assert all(0 < s <= 1.0 + 1e-9 for _, s in res)
    assert cosine_search(idx, ["zzzunseen"], k=3) == []
