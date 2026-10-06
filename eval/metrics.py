"""Retrieval evaluation metrics: recall@k, joint@k, P@k, MRR (PLAN.md §12)."""
import pandas as pd


def evaluate(rankings: dict[str, list[int]], qrels: dict[str, list[int]], meta: dict) -> pd.DataFrame:
    """Compute per-question IR metrics for a set of rankings against qrels."""
    return pd.DataFrame()
