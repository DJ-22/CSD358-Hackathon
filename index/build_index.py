"""Build and pickle the positional zone index for a corpus."""
import sys, pathlib  # bootstrap: make the repo root importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import json
import time

import config
from index.inverted_index import build, index_path, save_index


def load_corpus(corpus: str) -> list[dict]:
    """Read the document collection for a corpus."""
    with open(config.DATA_PROCESSED / f"corpus_{corpus}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    """Index construction: --corpus dev|train [--no-stem]."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["dev", "train"], required=True)
    ap.add_argument("--no-stem", action="store_true", help="build the unstemmed variant for the stemming ablation")
    args = ap.parse_args()
    stem = config.USE_STEMMING and not args.no_stem
    variant = "" if stem else "nostem"

    t0 = time.time()
    docs = load_corpus(args.corpus)
    idx = build(docs, stem=stem)
    save_index(idx, args.corpus, variant)
    secs = time.time() - t0
    path = index_path(args.corpus, variant)
    n_post = {z: len(idx.zones[z].docs) for z in idx.zones}
    print(f"[{args.corpus}{'/' + variant if variant else ''}] N={idx.N} vocab={len(idx.terms)} "
          f"postings title={n_post['title']} body={n_post['body']} "
          f"avg_len title={idx.avg_len('title'):.2f} body={idx.avg_len('body'):.2f}")
    print(f"wrote {path.name} ({path.stat().st_size / 2**20:.1f} MB) in {secs:.1f} s")


if __name__ == "__main__":
    main()
