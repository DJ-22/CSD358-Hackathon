"""Dense document embeddings with MiniLM for hybrid hop 2."""
import sys, pathlib  # bootstrap: make the repo root importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

EMB_DIM = 384


def load_embeddings(corpus: str) -> np.ndarray:
    """Load L2-normalised float16 document embeddings aligned with doc_id (dense retrieval)."""
    return np.zeros((0, EMB_DIM), dtype=np.float16)


def encode_query(text: str) -> np.ndarray:
    """Encode a query into the same L2-normalised dense space (dense retrieval)."""
    return np.zeros(EMB_DIM, dtype=np.float32)


def main() -> None:
    """Embed every document: --corpus dev|train."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
