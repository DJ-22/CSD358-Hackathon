"""Retrieval evaluation metrics: recall@k, joint@k, P@k, MRR, hop-2 recall, bridge hit rate."""
import math

import pandas as pd

RECALL_KS = (2, 5, 10, 20)
JOINT_KS = (2, 10)
P_KS = (2, 10)
HOP2_K = 10
BRIDGE_K = 3
TYPES = ("bridge", "comparison")


def recall_at_k(ranking: list[int], gold: list[int], k: int) -> float:
    """Recall@k: fraction of relevant documents found in the top k."""
    if not gold:
        return 0.0
    return len(set(gold) & set(ranking[:k])) / len(gold)


def joint_at_k(ranking: list[int], gold: list[int], k: int) -> float:
    """Joint recall@k: 1 if every relevant document is in the top k (all supporting docs retrieved together)."""
    if not gold:
        return 0.0
    return float(set(gold) <= set(ranking[:k]))


def p_at_k(ranking: list[int], gold: list[int], k: int) -> float:
    """Precision@k: fraction of the top k positions holding a relevant document (missing positions count as non-relevant)."""
    return len(set(gold) & set(ranking[:k])) / k


def mrr(ranking: list[int], gold: list[int]) -> float:
    """Reciprocal rank of the first relevant document (0 if none retrieved)."""
    gold = set(gold)
    for rank, doc in enumerate(ranking, start=1):
        if doc in gold:
            return 1.0 / rank
    return 0.0


def hop2_recall_at_k(ranking: list[int], hop2_gold: int, k: int = HOP2_K) -> float:
    """Recall@k of the second-hop document (the gold doc that single-shot retrieval ranks lower)."""
    return float(hop2_gold in ranking[:k])


def bridge_hit_at_k(bridges: list[str], gold_surface: str, k: int = BRIDGE_K) -> float:
    """Bridge hit rate: 1 if the hop-2 gold title surface is among the top-k extracted bridge entities."""
    return float(gold_surface in bridges[:k])


def compute_hop2_gold(s0_scores_by_qid: dict[str, list[tuple[int, float]]],
                      gold_by_qid: dict[str, list[int]]) -> dict[str, int]:
    """Identify the hop-2 gold doc per question: the relevant doc with the lower single-shot BM25 score.

    A gold doc absent from the single-shot list scores -inf; on a tie the later gold doc is chosen.
    """
    out = {}
    for qid, gold in gold_by_qid.items():
        scores = {int(d): float(s) for d, s in s0_scores_by_qid.get(qid, [])}
        best_doc, best_score = None, math.inf
        for doc in gold:
            score = scores.get(int(doc), -math.inf)
            if score <= best_score:
                best_doc, best_score = int(doc), score
        out[qid] = best_doc
    return out


def per_question(rankings: dict[str, list[int]], qrels: dict[str, list[int]], meta: dict) -> pd.DataFrame:
    """Per-question IR metrics; meta[qid] holds 'type' and optionally 'hop2_gold', 'bridges', 'hop2_gold_surface'."""
    rows = []
    for qid, gold in qrels.items():
        ranking = [int(d) for d in rankings.get(qid, [])]
        m = meta.get(qid, {})
        row = {"qid": qid, "type": m.get("type", "")}
        for k in RECALL_KS:
            row[f"recall@{k}"] = recall_at_k(ranking, gold, k)
        for k in JOINT_KS:
            row[f"joint@{k}"] = joint_at_k(ranking, gold, k)
        for k in P_KS:
            row[f"p@{k}"] = p_at_k(ranking, gold, k)
        row["mrr"] = mrr(ranking, gold)
        hop2_gold = m.get("hop2_gold")
        row[f"hop2_recall@{HOP2_K}"] = math.nan if hop2_gold is None else hop2_recall_at_k(ranking, hop2_gold)
        bridges = m.get("bridges")
        surface = m.get("hop2_gold_surface")
        has_bridge = row["type"] == "bridge" and bridges is not None and surface is not None
        row[f"bridge_hit@{BRIDGE_K}"] = bridge_hit_at_k(bridges, surface) if has_bridge else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(pq: pd.DataFrame) -> pd.DataFrame:
    """Macro-average per-question metrics overall and split by question type (NaN entries are skipped)."""
    metric_cols = [c for c in pq.columns if c not in ("qid", "type")]
    rows = []
    for typ in ("all",) + TYPES:
        part = pq if typ == "all" else pq[pq["type"] == typ]
        row = {"type": typ, "n": len(part)}
        for c in metric_cols:
            row[c] = part[c].mean() if len(part) else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate(rankings: dict[str, list[int]], qrels: dict[str, list[int]], meta: dict) -> pd.DataFrame:
    """Compute IR metrics for a set of rankings against qrels: one row overall plus one per question type."""
    return aggregate(per_question(rankings, qrels, meta))
