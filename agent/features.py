"""IR features for the learned bridge ranker."""

FEATURE_NAMES = [
    "tfidf_in_source", "src_hop1_score", "src_hop1_rank", "mean_idf", "n_src_docs", "in_title_zone",
    "first_pos", "min_dist_consumed", "n_tokens", "lookahead_sparse", "lookahead_dense", "hand_score",
]


def candidate_features(state, cand: dict, hop1_results, res) -> list[float]:
    """Feature vector for one bridge candidate (tf-idf, rank, zone, proximity, lookahead)."""
    return [0.0] * len(FEATURE_NAMES)
