"""Tests for the title-dictionary entity linker on a toy corpus."""
import sys, pathlib  # bootstrap: make the repo root importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json

import pytest

import config
from agent.entities import EntityLinker, title_surface, tokenize

TITLES = ["Kiss and Tell (1945 film)", "Kiss and Tell (play)", "Shirley Temple", "Temple", "New York",
          "New York Times", "Times Square", "The", "1945", "Film", "Hollywood"]


@pytest.fixture
def linker(tmp_path, monkeypatch):
    """Linker over a toy corpus where 'film' is a high-df token."""
    docs = [{"doc_id": i, "title": t, "text": f"{t} is a film article."} for i, t in enumerate(TITLES)]
    with open(tmp_path / "corpus_toy.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    (tmp_path / "title2doc_toy.json").write_text(json.dumps({d["title"]: d["doc_id"] for d in docs}))
    monkeypatch.setattr(config, "DATA_PROCESSED", tmp_path)
    monkeypatch.setattr(config, "MAX_ENTITY_DF", 5)
    return EntityLinker("toy")


def test_parenthetical_stripping():
    assert title_surface("Kiss and Tell (1945 film)") == "kiss and tell"
    assert title_surface("Temple") == "temple"
    assert tokenize("Kiss-and-Tell, 1945!") == ["kiss", "and", "tell", "1945"]


def test_surface_maps_to_all_disambiguated_docs(linker):
    assert linker.surface2docs[("kiss", "and", "tell")] == [0, 1]
    assert linker.doc2surface[0] == "kiss and tell"


def test_longest_match_beats_shorter_overlap(linker):
    ents = linker.link("The New York Times Square office")
    surfaces = [e[0] for e in ents]
    assert "new york times" in surfaces
    assert "new york" not in surfaces and "times square" not in surfaces


def test_longest_match_starting_later_wins(linker):
    ents = linker.link("Actress Shirley Temple starred")
    assert [e[0] for e in ents] == ["shirley temple"]


def test_no_overlapping_spans(linker):
    ents = linker.link("Shirley Temple and Kiss and Tell in New York Times Square, Temple")
    spans = sorted((start, start + len(s.split())) for s, _, start in ents)
    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
        assert a1 <= b0


def test_stop_words_numbers_and_high_df_ignored(linker):
    ents = linker.link("The 1945 film")
    assert ents == []  # 'the' is a stop word, '1945' a pure number, 'film' exceeds MAX_ENTITY_DF


def test_positions_are_token_offsets(linker):
    ents = linker.link("In 1945, Shirley Temple starred in Kiss and Tell.")
    assert ents == [("shirley temple", [2], 2), ("kiss and tell", [0, 1], 6)]
