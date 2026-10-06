"""Text analysis: tokenisation, case-folding, stop-word removal, Porter stemming (PLAN.md §1)."""


def analyze(text: str, stem: bool = True) -> list[str]:
    """Analyse text into index terms (tokenise, case-fold, stop, stem)."""
    return []


def analyze_with_positions(text: str) -> list[tuple[str, int]]:
    """Analyse text into (term, position) pairs; positions counted before stop-word removal (positional index)."""
    return []
