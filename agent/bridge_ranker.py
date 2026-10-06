"""Inference with the learned bridge ranker (PLAN.md §8)."""


def rank_bridges(state, hop1_results, res, k: int = 3) -> list[tuple[str, float, int]]:
    """Learning-to-rank inference: (entity, p_e softmax, source_doc) for the top-k bridges."""
    return []
