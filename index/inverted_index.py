"""Positional zone inverted index with title/body zones."""


class InvertedIndex:
    """Positional inverted index over title and body zones with df, N and length statistics."""

    def __init__(self):
        self.N = 0
        self.zones = {"title": {}, "body": {}}
        self.doc_len = {"title": [], "body": []}
        self.avg_len = {"title": 0.0, "body": 0.0}

    def postings(self, term: str, zone: str) -> list[tuple[int, int, list[int]]]:
        """Postings list (doc_id, tf, positions) sorted by doc_id."""
        return []

    def df(self, term: str, zone: str) -> int:
        """Document frequency of a term in a zone."""
        return 0

    def idf(self, term: str) -> float:
        """BM25 idf on the body zone: log((N - df + 0.5)/(df + 0.5) + 1)."""
        return 0.0

    def phrase_docs(self, tokens: list[str], zone: str) -> set[int]:
        """Phrase query via positional postings intersection (consecutive positions)."""
        return set()

    def doc_term_tf(self, doc_id: int, term: str, zone: str) -> int:
        """Term frequency of a term in one document zone (forward index lookup)."""
        return 0

    def doc_terms(self, doc_id: int, zone: str) -> dict[str, int]:
        """Forward index: term -> tf for one document zone."""
        return {}

    def doc_text(self, doc_id: int) -> str:
        """Raw body text of a document."""
        return ""

    def doc_title(self, doc_id: int) -> str:
        """Raw title of a document."""
        return ""


def load_index(corpus: str, variant: str = "") -> InvertedIndex:
    """Load a pickled inverted index (variant "" or "nostem")."""
    return InvertedIndex()
