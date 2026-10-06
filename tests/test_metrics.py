"""Tests for the retrieval evaluation metrics (hand-computed toy cases)."""
import sys, pathlib  # bootstrap: make the repo root importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import math

import pytest

from eval.metrics import (bridge_hit_at_k, compute_hop2_gold, evaluate, hop2_recall_at_k, joint_at_k, mrr,
                          p_at_k, per_question, recall_at_k)

GOLD = [7, 9]


def ranking_with(golds_at: dict[int, int], length: int = 20) -> list[int]:
    """Build a ranking of `length` docs with gold doc ids placed at the given 1-based ranks."""
    filler = iter(range(100, 200))
    return [golds_at.get(r) or next(filler) for r in range(1, length + 1)]


def test_golds_at_rank_1_and_15():
    r = ranking_with({1: 7, 15: 9})
    assert recall_at_k(r, GOLD, 2) == 0.5
    assert recall_at_k(r, GOLD, 10) == 0.5
    assert recall_at_k(r, GOLD, 20) == 1.0
    assert joint_at_k(r, GOLD, 10) == 0.0
    assert joint_at_k(r, GOLD, 20) == 1.0
    assert p_at_k(r, GOLD, 2) == 0.5
    assert p_at_k(r, GOLD, 10) == 0.1
    assert mrr(r, GOLD) == 1.0


def test_both_golds_in_top_2():
    r = [9, 7, 1, 2]
    assert recall_at_k(r, GOLD, 2) == 1.0
    assert joint_at_k(r, GOLD, 2) == 1.0
    assert p_at_k(r, GOLD, 2) == 1.0
    assert mrr(r, GOLD) == 1.0


def test_no_gold_retrieved():
    r = list(range(100, 120))
    for k in (2, 5, 10, 20):
        assert recall_at_k(r, GOLD, k) == 0.0
    assert joint_at_k(r, GOLD, 10) == 0.0
    assert p_at_k(r, GOLD, 10) == 0.0
    assert mrr(r, GOLD) == 0.0


def test_k_larger_than_list():
    r = [1, 7]
    assert recall_at_k(r, GOLD, 20) == 0.5
    assert joint_at_k(r, GOLD, 20) == 0.0
    assert p_at_k(r, GOLD, 10) == pytest.approx(0.1)  # missing positions count as non-relevant
    assert mrr(r, GOLD) == 0.5


def test_empty_ranking():
    assert recall_at_k([], GOLD, 10) == 0.0
    assert joint_at_k([], GOLD, 10) == 0.0
    assert p_at_k([], GOLD, 2) == 0.0
    assert mrr([], GOLD) == 0.0


def test_mrr_first_gold_at_rank_4():
    assert mrr([1, 2, 3, 9, 7], GOLD) == 0.25


def test_hop2_recall():
    r = ranking_with({1: 7, 12: 9})
    assert hop2_recall_at_k(r, 9, 10) == 0.0
    assert hop2_recall_at_k(r, 9, 20) == 1.0
    assert hop2_recall_at_k(r, 7, 10) == 1.0


def test_bridge_hit():
    bridges = ["kiss and tell", "shirley temple", "the beatles", "london"]
    assert bridge_hit_at_k(bridges, "shirley temple", 3) == 1.0
    assert bridge_hit_at_k(bridges, "london", 3) == 0.0
    assert bridge_hit_at_k([], "london", 3) == 0.0


def test_compute_hop2_gold():
    s0 = {"q1": [(7, 12.0), (9, 3.5)],          # 9 scores lower
          "q2": [(9, 8.0), (1, 5.0)],           # 7 absent -> -inf
          "q3": [(3, 1.0)],                     # both absent -> tie -> later gold
          "q4": [(7, 2.0), (9, 2.0)]}           # equal scores -> later gold
    gold = {"q1": [7, 9], "q2": [7, 9], "q3": [7, 9], "q4": [9, 7], "q5": [7, 9]}
    assert compute_hop2_gold(s0, gold) == {"q1": 9, "q2": 7, "q3": 9, "q4": 7, "q5": 9}


def test_evaluate_overall_and_per_type():
    qrels = {"a": [1, 2], "b": [3, 4], "c": [5, 6]}
    rankings = {"a": [1, 2, 9], "b": [9, 3], "c": [8, 9]}
    meta = {"a": {"type": "bridge", "hop2_gold": 2, "bridges": ["x", "y"], "hop2_gold_surface": "y"},
            "b": {"type": "bridge", "hop2_gold": 4, "bridges": ["z"], "hop2_gold_surface": "y"},
            "c": {"type": "comparison", "hop2_gold": 6}}
    pq = per_question(rankings, qrels, meta)
    assert list(pq["qid"]) == ["a", "b", "c"]
    assert math.isnan(pq.loc[2, "bridge_hit@3"])          # not defined for comparison questions
    df = evaluate(rankings, qrels, meta).set_index("type")
    assert list(df.index) == ["all", "bridge", "comparison"]
    assert df.loc["all", "n"] == 3
    assert df.loc["all", "recall@2"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert df.loc["bridge", "joint@2"] == pytest.approx(0.5)
    assert df.loc["bridge", "mrr"] == pytest.approx((1.0 + 0.5) / 2)
    assert df.loc["bridge", "hop2_recall@10"] == pytest.approx(0.5)
    assert df.loc["bridge", "bridge_hit@3"] == pytest.approx(0.5)
    assert df.loc["comparison", "recall@20"] == 0.0
    assert math.isnan(df.loc["comparison", "bridge_hit@3"])


def test_evaluate_missing_ranking_counts_as_empty():
    df = evaluate({}, {"a": [1, 2]}, {"a": {"type": "bridge"}}).set_index("type")
    assert df.loc["all", "joint@10"] == 0.0
    assert df.loc["comparison", "n"] == 0
