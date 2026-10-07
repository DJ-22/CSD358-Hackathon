"""Shared retrieval resources for one corpus: index, linker, embeddings, ranker."""
import json
from dataclasses import dataclass
from typing import Any

import config


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
    """Load every retrieval resource for a corpus (embeddings only if built, ranker only if requested)."""
    from index.inverted_index import load_index
    from index.embed import load_embeddings
    from agent.entities import EntityLinker
    emb_path = config.CACHE_DIR / f"doc_emb_{corpus}.npy"
    emb = load_embeddings(corpus) if emb_path.exists() else None
    title2doc = json.loads((config.DATA_PROCESSED / f"title2doc_{corpus}.json").read_text(encoding="utf-8"))
    ranker = None
    if with_ranker:
        from agent.bridge_ranker import load_ranker
        ranker = load_ranker()
    return Resources(corpus=corpus, index=load_index(corpus), linker=EntityLinker(corpus), emb=emb,
                     ranker=ranker, title2doc=title2doc)
