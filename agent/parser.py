"""Question parser (qtype, entities, query terms) and hop-1 retrieval (PLAN.md §5)."""
from agent.state import ResearchState


def parse(question: str, res, qid: str = "") -> ResearchState:
    """Query analysis: question type, idf-weighted query terms and linked question entities."""
    return ResearchState(qid=qid, question=question)


def hop1(state: ResearchState, res) -> list[tuple[int, float]]:
    """Hop 1: Boolean AND over top-idf terms union free-text BM25, ranked by BM25."""
    return []
