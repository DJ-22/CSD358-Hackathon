"""Download HotpotQA distractor Parquet and build corpora, splits and LTR questions."""
import sys, pathlib  # bootstrap: make the repo root importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import random
from collections import Counter

import pandas as pd
import requests
from tqdm import tqdm

import config


def download(url: str, dest: pathlib.Path) -> None:
    """Fetch a raw collection file by streaming it to disk (skipped if already present)."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} already present")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.replace(dest)


def normalise(row) -> dict:
    """Normalise one Parquet row into a question record with qrels (gold titles) and its 10 context paragraphs."""
    sf_titles = [str(t) for t in row["supporting_facts"]["title"]]
    gold_titles = list(dict.fromkeys(sf_titles))  # unique, order kept
    ctx = row["context"]
    context = [(str(t), [str(s) for s in sents]) for t, sents in zip(ctx["title"], ctx["sentences"])]
    return {"qid": str(row["id"]), "question": str(row["question"]), "answer": str(row["answer"]),
            "type": str(row["type"]), "gold_titles": gold_titles, "context": context}


def build_corpus(records: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Build the document collection: one doc per paragraph, dedupe by exact title (keep longest), ids in title order."""
    best = {}
    for rec in records:
        for title, sents in rec["context"]:
            text = " ".join(sents)
            if title not in best or len(text) > len(best[title]):
                best[title] = text
    docs = [{"doc_id": i, "title": t, "text": best[t]} for i, t in enumerate(sorted(best))]
    return docs, {d["title"]: d["doc_id"] for d in docs}


def question_record(rec: dict) -> dict:
    """Question record with its relevance judgements (gold titles)."""
    return {k: rec[k] for k in ("qid", "question", "type", "gold_titles", "answer")}


def write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    """Write rows as JSON lines."""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: pathlib.Path, obj) -> None:
    """Write one JSON object."""
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def load_records(path: pathlib.Path) -> list[dict]:
    """Read a Parquet shard and normalise every row."""
    df = pd.read_parquet(path)
    return [normalise(row) for row in tqdm(df.to_dict("records"), desc=f"normalise {path.name}")]


def main() -> None:
    """Build the document collections, qrels and EVAL/TUNE splits."""
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = config.DATA_PROCESSED
    download(config.DEV_URL, config.DEV_PARQUET)
    download(config.TRAIN_URL, config.TRAIN_PARQUET)

    # --- dev: retrieval corpus + evaluation questions ---
    dev = load_records(config.DEV_PARQUET)
    dev_docs, dev_t2d = build_corpus(dev)
    write_jsonl(out / "corpus_dev.jsonl", dev_docs)
    write_json(out / "title2doc_dev.json", dev_t2d)

    kept, dropped = [], 0
    for rec in dev:
        if len(rec["gold_titles"]) == 2 and all(t in dev_t2d for t in rec["gold_titles"]):
            kept.append(question_record(rec))
        else:
            dropped += 1
    print(f"dev questions: {len(dev)} total, {len(kept)} kept, {dropped} dropped (gold titles not 2 or not in corpus)")
    write_jsonl(out / "questions_dev.jsonl", kept)

    qids = sorted(q["qid"] for q in kept)
    random.Random(config.SEED).shuffle(qids)
    split_eval = qids[:config.EVAL_N]
    split_tune = qids[config.EVAL_N:config.EVAL_N + config.TUNE_N]
    assert len(split_eval) == config.EVAL_N and len(split_tune) == config.TUNE_N
    assert not set(split_eval) & set(split_tune), "EVAL and TUNE overlap"
    write_json(out / "split_eval.json", split_eval)
    write_json(out / "split_tune.json", split_tune)

    # -- train: LTR questions and their own corpus (never mixed with dev) ---
    train = [r for r in load_records(config.TRAIN_PARQUET) if r["type"] == "bridge"]
    train.sort(key=lambda r: r["qid"])
    train = random.Random(config.SEED).sample(train, config.LTR_TRAIN_N)
    train_docs, train_t2d = build_corpus(train)
    train_q = [question_record(r) for r in train]
    assert all(t in train_t2d for q in train_q for t in q["gold_titles"])
    write_jsonl(out / "questions_train.jsonl", train_q)
    write_jsonl(out / "corpus_train.jsonl", train_docs)
    write_json(out / "title2doc_train.json", train_t2d)

    by_qid = {q["qid"]: q for q in kept}
    types = Counter(by_qid[q]["type"] for q in split_eval)
    print("=== summary ===")
    print(f"#docs dev corpus:   {len(dev_docs)}")
    print(f"#docs train corpus: {len(train_docs)}")
    print(f"#dev questions kept: {len(kept)} (EVAL {len(split_eval)}, TUNE {len(split_tune)}, disjoint)")
    print(f"#train LTR questions: {len(train_q)} (bridge only)")
    print(f"EVAL type distribution: {dict(types)}")


if __name__ == "__main__":
    main()
