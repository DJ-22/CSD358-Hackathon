"""Shared retrieval resources for one corpus: index, linker, embeddings, ranker."""
from dataclasses import dataclass
from typing import Any


@dataclass
class Resources:
    """Bundle of loaded IR resources (inverted index, entity linker, dense embeddings, bridge ranker)."""
    corpus: str
    index: Any
    linker: Any
    emb: Any
    ranker: Any | None
    title2doc: dict


def load_resources(corpus: str = "dev", with_ranker: bool = True) -> Resources:
    """Load every retrieval resource for a corpus."""
    from index.inverted_index import InvertedIndex
    from index.embed import load_embeddings
    from agent.entities import EntityLinker
    return Resources(corpus=corpus, index=InvertedIndex(), linker=EntityLinker(corpus),
                     emb=load_embeddings(corpus), ranker=None, title2doc={})
