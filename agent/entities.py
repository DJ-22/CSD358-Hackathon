"""Dictionary-based entity linker over corpus titles (PLAN.md §4)."""


class EntityLinker:
    """Longest-match n-gram entity linker mapping title surfaces to doc_ids."""

    def __init__(self, corpus: str):
        self.corpus = corpus
        self.surface2docs = {}

    def link(self, text: str) -> list[tuple[str, list[int], int]]:
        """Entity linking: (surface, doc_ids, start) via longest-match sliding n-grams, no overlaps."""
        return []
