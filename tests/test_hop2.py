"""Tests for hop-2 retrieval (BM25 and hybrid modes) and the question-type rules on a toy corpus."""
import sys, pathlib  # bootstrap: make the repo root importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

import config
from agent import hop2
from agent.hop2 import hop2_retrieve
from agent.parser import classify_qtype
from agent.resources import Resources
from agent.state import ResearchState
from index.inverted_index import build
from retrieval.bm25 import bm25_search
from tests.test_index import DOCS


@pytest.fixture(scope="module")
def res():
    return Resources(corpus="toy", index=build(DOCS, stem=True), linker=None, emb=None, ranker=None, title2doc={})


def state_with_residual(res, words: str) -> ResearchState:
    s = ResearchState(qid="q", question=words)
    s.residual_terms = res.index.analyze(words)
    s.residual_text = words
    return s


def test_bm25_mode_shape_and_scores(res):
    out = hop2_retrieve("Shirley Temple", state_with_residual(res, "actress diplomat"), res, "bm25", 0.5, None)
    assert 1 <= len(out) <= 3
    assert out[0][0] == 1
    for doc, s_sparse, s_dense, s_b in out:
        assert 0.0 <= s_b <= 1.0 and s_dense == 0.0 and s_sparse > 0
    assert out[0][3] == pytest.approx(1.0)
    assert [o[3] for o in out] == sorted((o[3] for o in out), reverse=True)


def test_bm25_mode_respects_exclude(res):
    st = state_with_residual(res, "actress diplomat")
    out = hop2_retrieve("Shirley Temple", st, res, "bm25", 0.5, exclude=1)
    assert 1 not in [d for d, *_ in out]
    assert len(out) <= 3 and out[0][3] == pytest.approx(1.0)


def test_bm25_mode_no_terms_returns_empty(res):
    assert hop2_retrieve("the", state_with_residual(res, "of"), res, "bm25", 0.5, None) == []
    assert hop2_retrieve("zzzunseen", state_with_residual(res, ""), res, "bm25", 0.5, None) == []


E = np.eye(4, dtype=np.float32)
TOY_EMB = np.stack([E[0], 0.9 * E[0] + np.sqrt(1 - 0.81) * E[1], E[2], E[3], E[1]]).astype(np.float16)


@pytest.fixture
def hres(res, monkeypatch):
    """Toy resources with hand-made unit embeddings; every residual encodes to E[0]."""
    monkeypatch.setattr(hop2, "_query_vec", lambda text: E[0])
    return Resources(corpus="toy", index=res.index, linker=None, emb=TOY_EMB, ranker=None, title2doc={})


def test_hybrid_filter_keeps_only_phrase_docs(hres):
    docs, _ = hop2.sparse_filter("shirley temple", state_with_residual(hres, "actress"), hres, None)
    assert docs.tolist() == [0, 1]                 # doc 0 body and doc 1 title/body contain the phrase
    out = hop2_retrieve("shirley temple", state_with_residual(hres, "actress"), hres, "hybrid", 0.5, None)
    assert {d for d, *_ in out} <= {0, 1}
    assert hop2_retrieve("temple shirley", state_with_residual(hres, "actress"), hres, "hybrid", 0.5, None) == []


def test_hybrid_respects_exclude(hres):
    out = hop2_retrieve("shirley temple", state_with_residual(hres, "actress"), hres, "hybrid", 0.5, exclude=1)
    assert [d for d, *_ in out] == [0]


def test_hybrid_title_bonus(hres, monkeypatch):
    monkeypatch.setattr(config, "TITLE_BONUS", 0.2)
    st = state_with_residual(hres, "actress")
    out = {d: (sp, de, sb) for d, sp, de, sb in hop2_retrieve("shirley temple", st, hres, "hybrid", 0.0, None)}
    assert out[0][1] == pytest.approx(1.0, abs=1e-3) and out[1][1] == pytest.approx(0.9, abs=1e-3)
    assert out[1][2] - out[1][1] == pytest.approx(0.2)        # doc 1's title surface equals the entity
    assert out[0][2] - out[0][1] == pytest.approx(0.0)
    assert out[1][2] > out[0][2]                               # the bonus flips the dense order
    monkeypatch.setattr(config, "TITLE_BONUS", 0.0)
    first = hop2_retrieve("shirley temple", st, hres, "hybrid", 0.0, None)[0][0]
    assert first == 0


def test_hybrid_alpha_one_matches_bm25_on_candidates(hres, monkeypatch):
    monkeypatch.setattr(config, "TITLE_BONUS", 0.0)
    for entity, residual in (("shirley temple", "american diplomat"), ("film", "motion picture starring")):
        st = state_with_residual(hres, residual)
        docs, _ = hop2.sparse_filter(entity, st, hres, None)
        expected = [d for d, _ in bm25_search(hres.index, st.residual_terms, 3, candidates=set(docs.tolist()))]
        got = [d for d, *_ in hop2_retrieve(entity, st, hres, "hybrid", 1.0, None)]
        assert got == expected
        assert all(0.0 <= sp <= 1.0 for _, sp, _, _ in hop2_retrieve(entity, st, hres, "hybrid", 1.0, None))


def test_hybrid_filter_cap_keeps_best_bm25(hres, monkeypatch):
    monkeypatch.setattr(config, "HOP2_FILTER_CAP", 1)
    docs, _ = hop2.sparse_filter("shirley temple", state_with_residual(hres, "actress"), hres, None)
    assert docs.tolist() == [1]                    # only doc 1 mentions "actress"


def test_hybrid_needs_embeddings(res):
    with pytest.raises(RuntimeError):
        hop2_retrieve("shirley temple", state_with_residual(res, "actress"), res, "hybrid", 0.5, None)


def test_unknown_mode_rejected(res):
    with pytest.raises(ValueError):
        hop2_retrieve("Temple", state_with_residual(res, "actress"), res, "nope", 0.5, None)


@pytest.mark.parametrize("question, entities, expected", [
    ("Were Scott Derrickson and Ed Wood of the same nationality?", [], "comparison"),
    ("Are both Kiss and Tell and Film American?", [], "comparison"),
    ("Which was released first, Kiss and Tell or New York?", [], "comparison"),
    ("Is Temple Bar or Shirley Temple older than 50?", [("temple bar", [0], 1), ("shirley temple", [1], 4)], "comparison"),
    ("Is it Shirley Temple or not?", [("shirley temple", [1], 2)], "bridge"),
    ("Who directed the film starring Shirley Temple?", [("shirley temple", [1], 5)], "bridge"),
])
def test_qtype_rules(question, entities, expected):
    assert classify_qtype(question, entities) == expected
