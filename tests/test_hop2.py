"""Tests for hop-2 retrieval and the question-type rules on a toy corpus."""
import sys, pathlib  # bootstrap: make the repo root importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from agent.hop2 import hop2_retrieve
from agent.parser import classify_qtype
from agent.resources import Resources
from agent.state import ResearchState
from index.inverted_index import build
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
