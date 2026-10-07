"""Dense document embeddings with MiniLM for hybrid hop 2."""
import sys, pathlib  # bootstrap: make the repo root importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import json
import time

import numpy as np

import config

EMB_DIM = 384
_MODEL = None


def _model():
    """Load the sentence encoder once per process (CPU)."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        try:  # offline once cached; download only on first use
            _MODEL = SentenceTransformer(config.EMB_MODEL, device="cpu", local_files_only=True)
        except OSError:
            _MODEL = SentenceTransformer(config.EMB_MODEL, device="cpu")
    return _MODEL


def emb_path(corpus: str) -> pathlib.Path:
    """Path of a corpus' document embedding matrix."""
    return config.CACHE_DIR / f"doc_emb_{corpus}.npy"


def doc_inputs(docs: list[dict]) -> list[str]:
    """Encoder input per document: title + '. ' + body text."""
    return [d["title"] + ". " + d["text"] for d in docs]


def embed_texts(texts: list[str], progress: bool = True) -> np.ndarray:
    """Encode texts into L2-normalised float16 vectors (dense representation)."""
    vecs = _model().encode(texts, batch_size=config.EMB_BATCH, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=progress)
    return vecs.astype(np.float16)


def load_embeddings(corpus: str) -> np.ndarray:
    """Load L2-normalised float16 document embeddings aligned with doc_id (dense retrieval)."""
    return np.load(emb_path(corpus))


def encode_query(text: str) -> np.ndarray:
    """Encode a query into the same L2-normalised dense space (dense retrieval)."""
    return _model().encode([text], normalize_embeddings=True, convert_to_numpy=True,
                           show_progress_bar=False)[0].astype(np.float32)


def main() -> None:
    """Embed every document: --corpus dev|train [--limit N to verify on the first N docs without writing]."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["dev", "train"], required=True)
    ap.add_argument("--limit", type=int, default=None, help="encode only the first N docs and print checks (no file written)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing embedding file")
    args = ap.parse_args()

    out = emb_path(args.corpus)
    if args.limit is None and out.exists() and not args.force:
        print(f"[skip] {out.name} already exists (use --force to rebuild)")
        return
    with open(config.DATA_PROCESSED / f"corpus_{args.corpus}.jsonl", encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]
    assert [d["doc_id"] for d in docs] == list(range(len(docs))), "corpus must be in doc_id order"
    if args.limit is not None:
        docs = docs[:args.limit]

    t0 = time.time()
    emb = embed_texts(doc_inputs(docs))
    secs = time.time() - t0
    norms = np.linalg.norm(emb.astype(np.float32), axis=1)
    print(f"[{args.corpus}] shape={emb.shape} dtype={emb.dtype} norm min={norms.min():.4f} max={norms.max():.4f} "
          f"in {secs:.1f} s ({len(docs) / secs:.0f} docs/s)")
    if args.limit is not None:
        return
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".part.npy")
    np.save(tmp, emb)
    tmp.replace(out)
    print(f"wrote {out.name} ({out.stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
