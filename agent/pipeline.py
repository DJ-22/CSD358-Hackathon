"""BridgeHop pipeline: systems S0-S5."""
from agent.state import ResearchState


def run_system(qrec: dict, cfg: dict, res) -> tuple[list[tuple[int, float]], ResearchState]:
    """Multi-hop agentic retrieval for one question under a system configuration."""
    return [], ResearchState(qid=qrec.get("qid", ""), question=qrec.get("question", ""))
