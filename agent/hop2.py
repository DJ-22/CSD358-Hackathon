"""Hop-2 retrieval: BM25 mode and hybrid sparse-filter -> dense-score mode."""
from agent.state import ResearchState


def hop2_retrieve(entity: str, state: ResearchState, res, mode: str, alpha: float, exclude: int | None
                  ) -> list[tuple[int, float, float, float]]:
    """Entity-conditioned second-hop retrieval returning (doc, s_sparse, s_dense, s_B)."""
    return []
