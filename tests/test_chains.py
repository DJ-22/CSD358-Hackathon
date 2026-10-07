"""Tests for chain scoring, rank fusion and the final ranking on synthetic traces."""
import sys, pathlib  # bootstrap: make the repo root importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from agent.chains import chain_first, final_ranking, round_robin, score_chains
from agent.fusion import combsum, rrf
from agent.state import ResearchState


def trace(qtype: str = "bridge") -> ResearchState:
    """Synthetic trace: hop 1 over docs 1-4, two bridges with their hop-2 results."""
    s = ResearchState(qid="q", question="?", qtype=qtype)
    s.hops = [
        {"hop": 1, "results": [(1, 10.0), (2, 6.0), (3, 2.0), (4, 1.0)]},
        {"hop": 2, "entity": "a", "source_doc": 1, "results": [(10, 9.0, 0.0, 1.0), (11, 5.0, 0.0, 0.5)]},
        {"hop": 2, "entity": "b", "source_doc": 2, "results": [(20, 8.0, 0.0, 0.8), (1, 3.0, 0.0, 0.2)]},
    ]
    s.bridges = [{"entity": "a", "source_doc": 1, "hand_score": 3.0, "p_e": 0.6, "features": None},
                 {"entity": "b", "source_doc": 2, "hand_score": 2.0, "p_e": 0.4, "features": None}]
    return s


def test_rrf():
    fused = rrf([[(1, 9.0), (2, 8.0)], [(2, 5.0), (3, 4.0)]], k=60)
    assert [d for d, _ in fused] == [2, 1, 3]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)
    assert rrf([]) == []


def test_rrf_ties_broken_by_doc_id():
    assert [d for d, _ in rrf([[(5, 1.0)], [(3, 1.0)]], k=60)] == [3, 5]


def test_combsum_minmax():
    fused = dict(combsum([[(1, 10.0), (2, 5.0), (3, 0.0)], [(3, 4.0), (2, 2.0)]]))
    # list 1 -> {1: 1, 2: .5, 3: 0}; list 2 -> {3: 1, 2: 0}
    assert fused == {1: pytest.approx(1.0), 2: pytest.approx(0.5), 3: pytest.approx(1.0)}
    assert combsum([[(7, 2.0), (8, 2.0)]]) == [(7, 1.0), (8, 1.0)]


def test_chain_components_minmax_normalised():
    s = trace()
    chains = score_chains(s, None, {"lambdas": (1.0, 0.0, 0.0)})
    assert len(chains) == 4
    by = {(c["d1"], c["d2"]): c["S"] for c in chains}
    assert by[(1, 10)] == pytest.approx(1.0) and by[(2, 20)] == pytest.approx(0.0)   # s1: 10 -> 1, 6 -> 0


def test_lambda_weighting():
    s = trace()
    chains = score_chains(s, None, {"lambdas": (0.3, 0.3, 0.4)})
    by = {(c["d1"], c["d2"]): c["S"] for c in chains}
    # normalised: s1 {1: 1, 2: 0}; p_e {a: 1, b: 0}; sB {10: 1, 11: .375, 20: .75, 1: 0}
    assert by[(1, 10)] == pytest.approx(1.0)
    assert by[(1, 11)] == pytest.approx(0.3 + 0.3 + 0.4 * 0.375)
    assert by[(2, 20)] == pytest.approx(0.4 * 0.75)
    assert by[(2, 1)] == pytest.approx(0.0)
    assert [(c["d1"], c["d2"]) for c in chains] == [(1, 10), (1, 11), (2, 20), (2, 1)]
    assert s.chains == chains


def test_chain_first_order_without_duplicates():
    chains = [{"d1": 1, "d2": 10, "S": 0.9}, {"d1": 1, "d2": 11, "S": 0.8}, {"d1": 2, "d2": 1, "S": 0.5}]
    ranking = [d for d, _ in chain_first(chains, [], final_k=20)]
    assert ranking == [1, 10, 11, 2]


def test_chain_first_rrf_fill_up_to_final_k():
    chains = [{"d1": 1, "d2": 10, "S": 0.9}]
    hop1 = [(d, 1.0 / d) for d in range(1, 30)]
    ranking = chain_first(chains, [hop1], final_k=8)
    docs = [d for d, _ in ranking]
    assert docs == [1, 10, 2, 3, 4, 5, 6, 7]
    assert len(set(docs)) == 8
    assert all(a[1] > b[1] for a, b in zip(ranking, ranking[1:]))      # scores decrease with rank


def test_round_robin_for_comparison():
    slot_a = [(10, 1.0), (11, 0.5), (12, 0.1)]
    slot_b = [(20, 1.0), (10, 0.7)]
    hop1 = [(5, 3.0), (20, 2.0), (6, 1.0)]
    docs = [d for d, _ in round_robin([slot_a, slot_b], [hop1, slot_a, slot_b], final_k=7)]
    assert docs == [10, 20, 11, 12, 5, 6]                  # only 6 distinct docs exist
    docs = [d for d, _ in round_robin([slot_a, slot_b], [hop1, slot_a, slot_b], final_k=3)]
    assert docs == [10, 20, 11]


def test_final_ranking_dispatch():
    s = trace()
    score_chains(s, None, {})
    chain = [d for d, _ in final_ranking(s, {}, final_k=6)]
    assert chain[:2] == [1, 10]
    fused = [d for d, _ in final_ranking(s, {"fusion": "rrf"}, final_k=6)]
    assert fused[0] == 1                                    # in hop 1 and in a hop-2 list
    s.qtype = "comparison"
    rr = [d for d, _ in final_ranking(s, {}, final_k=6)]
    assert rr[:4] == [10, 20, 11, 1]
