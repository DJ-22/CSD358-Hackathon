"""Tests for text analysis and the positional zone index on a toy corpus."""
import sys, pathlib  # bootstrap: make the repo root importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import math

import pytest

from index.inverted_index import build
from index.preprocess import analyze, analyze_with_positions

DOCS = [
    {"doc_id": 0, "title": "Kiss and Tell", "text": "Kiss and Tell is a 1945 film starring Shirley Temple."},
    {"doc_id": 1, "title": "Shirley Temple", "text": "Shirley Temple was an American actress. Temple later served as a diplomat."},
    {"doc_id": 2, "title": "New York", "text": "New York is a city. The city of New York is large."},
    {"doc_id": 3, "title": "Film", "text": "A film is a motion picture."},
    {"doc_id": 4, "title": "Diplomat", "text": "A diplomat represents a state."},
]


@pytest.fixture(scope="module")
def idx():
    return build(DOCS, stem=True)


def term(word: str) -> str:
    return analyze(word)[0]


def test_analyze_stops_stems_and_casefolds():
    assert analyze("The Films were Starring") == ["film", "star"]
    assert analyze("The Films", stem=False) == ["films"]


def test_positions_counted_before_stop_word_removal():
    assert analyze_with_positions("Kiss and Tell") == [("kiss", 0), ("tell", 2)]


def test_postings_sorted_by_doc_id(idx):
    for zone in ("title", "body"):
        for t in idx.terms:
            docs = [d for d, _, _ in idx.postings(t, zone)]
            assert docs == sorted(set(docs))


def test_tf_and_positions(idx):
    temple = term("temple")
    post = {d: (tf, pos) for d, tf, pos in idx.postings(temple, "body")}
    assert post == {0: (1, [9]), 1: (2, [1, 6])}
    assert idx.doc_term_tf(1, temple, "body") == 2
    assert idx.doc_terms(1, "title") == {term("shirley"): 1, temple: 1}


def test_df_idf_and_lengths(idx):
    assert idx.N == 5
    assert idx.df(term("temple"), "body") == 2
    assert idx.df(term("temple"), "title") == 1
    assert idx.idf(term("temple")) == pytest.approx(math.log((5 - 2 + 0.5) / (2 + 0.5) + 1))
    assert idx.idf("zzzunseen") == pytest.approx(math.log(5.5 / 0.5 + 1))
    assert idx.doc_len(4, "body") == 3          # diplomat, represents, state
    assert idx.avg_len("title") == pytest.approx((2 + 2 + 2 + 1 + 1) / 5)


def test_zone_separation(idx):
    dip = term("diplomat")
    assert idx.df(dip, "title") == 1 and idx.df(dip, "body") == 2
    assert idx.doc_term_tf(1, dip, "title") == 0
    assert idx.doc_term_tf(1, dip, "body") == 1


def test_phrase_match_across_stop_word(idx):
    assert idx.phrase_docs(["Kiss", "and", "Tell"], "body") == {0}
    assert idx.phrase_docs(["Kiss", "and", "Tell"], "title") == {0}
    assert idx.phrase_docs(["kiss", "tell"], "body") == set()   # the stop word keeps its slot


def test_phrase_order_and_missing_terms(idx):
    assert idx.phrase_docs(["new", "york"], "body") == {2}
    assert idx.phrase_docs(["york", "new"], "body") == set()
    assert idx.phrase_docs(["new", "jersey"], "body") == set()
    assert idx.phrase_docs(["the", "of"], "body") == set()


def test_doc_text_and_title(idx):
    assert idx.doc_title(3) == "Film"
    assert idx.doc_text(3) == "A film is a motion picture."
