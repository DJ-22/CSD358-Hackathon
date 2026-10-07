"""Tests for bridge candidate extraction, hand score and the residual query on a toy corpus."""
import sys, pathlib  # bootstrap: make the repo root importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
from types import SimpleNamespace

import pytest

import config
from agent.bridge import bridge_candidates, hand_bridges, minmax
from agent.entities import EntityLinker
from agent.state import ResearchState
from index.inverted_index import build

DOCS = [
    ("Kiss and Tell (1945 film)", "Kiss and Tell is a 1945 comedy film starring Shirley Temple as Corliss Archer."),
    ("Shirley Temple", "Shirley Temple was an American actress who served as Chief of Protocol of the United States."),
    ("Corliss Archer", "Corliss Archer is a fictional teenage girl. Shirley Temple played her in a film."),
    ("Chief of Protocol", "The Chief of Protocol of the United States is an official of the State Department."),
    ("Meet Corliss Archer", "Meet Corliss Archer is a radio sitcom about a teenage girl."),
    ("United States", "The United States is a country in North America."),
]
QUESTION = "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?"


@pytest.fixture
def res(tmp_path, monkeypatch):
    """Toy resources: positional index and title linker over six paragraphs."""
    docs = [{"doc_id": i, "title": t, "text": x} for i, (t, x) in enumerate(DOCS)]
    with open(tmp_path / "corpus_toy.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    (tmp_path / "title2doc_toy.json").write_text(json.dumps({d["title"]: d["doc_id"] for d in docs}))
    monkeypatch.setattr(config, "DATA_PROCESSED", tmp_path)
    return SimpleNamespace(index=build(docs), linker=EntityLinker("toy"))


def make_state(question: str, entities: set[str], res) -> ResearchState:
    """Research state with idf-weighted query terms and the given question entities."""
    terms = res.index.analyze(question)
    return ResearchState(qid="q", question=question, query_terms={t: res.index.idf(t) for t in terms},
                         known_entities=set(entities))


def test_minmax():
    assert minmax([10.0, 5.0, 0.0]) == [1.0, 0.5, 0.0]
    assert minmax([3.0, 3.0]) == [1.0, 1.0]
    assert minmax([]) == []


def test_candidates_exclude_known_entities_and_own_title(res):
    state = make_state(QUESTION, {"corliss archer", "kiss and tell"}, res)
    cands = bridge_candidates(state, [(0, 10.0), (2, 8.0), (4, 5.0)], res)
    names = [c["entity"] for c in cands]
    assert "shirley temple" in names
    assert "corliss archer" not in names and "kiss and tell" not in names   # question entities
    assert "meet corliss archer" not in names                               # doc 4's own title
    st = next(c for c in cands if c["entity"] == "shirley temple")
    assert st["source_doc"] == 0 and st["entity_doc"] == 1


def test_entity_title_of_another_doc_is_a_candidate(res):
    state = make_state(QUESTION, set(), res)
    cands = bridge_candidates(state, [(2, 9.0), (0, 8.0)], res)
    names = [c["entity"] for c in cands]
    assert "corliss archer" in names        # doc 2's own title, but linked from doc 0
    assert next(c for c in cands if c["entity"] == "corliss archer")["source_doc"] == 0


def test_residual_removes_terms_in_top_doc(res):
    state = make_state(QUESTION, {"corliss archer", "kiss and tell"}, res)
    bridge_candidates(state, [(0, 10.0), (2, 8.0)], res)
    film = res.index.analyze("film")[0]
    assert film in state.consumed_terms and film not in state.residual_terms
    assert state.residual_terms == res.index.analyze("government position held woman portrayed")
    assert state.residual_text == "government position held woman portrayed"


def test_empty_residual_falls_back_to_non_entity_terms(res):
    question = "Which comedy film starring Shirley Temple?"
    state = make_state(question, {"shirley temple"}, res)
    bridge_candidates(state, [(0, 10.0)], res)
    assert state.residual_terms == res.index.analyze("comedy film starring")
    assert state.residual_text == "comedy film starring"


def test_hand_score_higher_in_two_top_docs_than_one(res):
    s1 = make_state(QUESTION, {"corliss archer", "kiss and tell"}, res)
    two = bridge_candidates(s1, [(0, 10.0), (2, 10.0)], res)       # Shirley Temple in both docs
    s2 = make_state(QUESTION, {"corliss archer", "kiss and tell"}, res)
    one = bridge_candidates(s2, [(0, 10.0), (3, 10.0)], res)       # only in doc 0
    score = lambda cs: next(c["hand_score"] for c in cs if c["entity"] == "shirley temple")
    assert score(two) > score(one) > 0


def test_hand_bridges_records_top_k(res):
    state = make_state(QUESTION, {"corliss archer", "kiss and tell"}, res)
    cands = bridge_candidates(state, [(0, 10.0), (1, 8.0), (2, 6.0)], res)
    beam = hand_bridges(state, cands, k=2)
    assert len(beam) == len(state.bridges) == min(2, len(cands))
    assert [b[0] for b in beam] == [c["entity"] for c in cands[:2]]
    assert all(0.0 <= p <= 1.0 for _, p, _ in beam)
