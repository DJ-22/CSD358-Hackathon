"""Evaluation harness: run systems over EVAL/TUNE and write results."""
import sys, pathlib  # bootstrap: make the repo root importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import json
import time

import pandas as pd
from tqdm import tqdm

import config
from eval.metrics import aggregate, compute_hop2_gold, per_question

HOP2_GOLD_PATH = config.DATA_PROCESSED / "hop2_gold.json"
METRICS_PATH = config.RESULTS_DIR / "metrics.csv"


def load_questions() -> dict[str, dict]:
    """Load dev question records (query text, type, gold titles) keyed by qid."""
    with open(config.DATA_PROCESSED / "questions_dev.jsonl", encoding="utf-8") as f:
        return {q["qid"]: q for q in map(json.loads, f)}


def load_split(split: str) -> list[str]:
    """Load the qids of the EVAL or TUNE split."""
    return json.loads((config.DATA_PROCESSED / f"split_{split}.json").read_text(encoding="utf-8"))


def build_qrels(qrecs: dict[str, dict], title2doc: dict) -> dict[str, list[int]]:
    """Relevance judgements: gold supporting-paragraph titles mapped to doc_ids."""
    return {qid: [title2doc[t] for t in q["gold_titles"]] for qid, q in qrecs.items()}


def load_hop2_gold(res, qrecs: dict[str, dict], qrels: dict[str, list[int]]) -> dict[str, int]:
    """Hop-2 gold doc per EVAL/TUNE question from single-shot BM25 scores of the two gold docs (cached on disk)."""
    if HOP2_GOLD_PATH.exists():
        return json.loads(HOP2_GOLD_PATH.read_text(encoding="utf-8"))
    from retrieval.bm25 import bm25_search
    qids = load_split("eval") + load_split("tune")
    s0 = {qid: bm25_search(res.index, res.index.analyze(qrecs[qid]["question"]), 2, candidates=set(qrels[qid]))
          for qid in tqdm(qids, desc="hop-2 gold")}
    gold = compute_hop2_gold(s0, {qid: qrels[qid] for qid in qids})
    HOP2_GOLD_PATH.write_text(json.dumps(gold), encoding="utf-8")
    return gold


def run_one(label: str, cfg: dict, qids: list[str], qrecs: dict, qrels: dict, hop2_gold: dict, res,
            run_fn=None) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Batch evaluation of one system: returns (per-question metrics, aggregated metrics, ms per question)."""
    if run_fn is None:
        from agent.pipeline import run_system as run_fn
    rankings, meta = {}, {}
    t0 = time.perf_counter()
    for qid in tqdm(qids, desc=label):
        q = qrecs[qid]
        ranking, state = run_fn(q, cfg, res)
        rankings[qid] = [int(d) for d, _ in ranking]
        m = {"type": q["type"]}
        if qid in hop2_gold:
            m["hop2_gold"] = hop2_gold[qid]
            m["hop2_gold_surface"] = res.linker.doc2surface[hop2_gold[qid]]
        if cfg.get("bridge"):
            m["bridges"] = [b["entity"] for b in state.bridges]
        meta[qid] = m
    ms = 1000 * (time.perf_counter() - t0) / max(len(qids), 1)
    pq = per_question(rankings, {qid: qrels[qid] for qid in qids}, meta)
    return pq, aggregate(pq), ms


def write_results(label: str, split: str, pq: pd.DataFrame, agg: pd.DataFrame, ms: float) -> None:
    """Write per-question metrics and replace this system's rows in the metrics table."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if split == "eval" else f"_{split}"
    pq.to_csv(config.RESULTS_DIR / f"per_question_{label}{suffix}.csv", index=False)
    agg = agg.copy()
    agg.insert(0, "split", split)
    agg.insert(0, "system", label)
    agg["ms_per_q"] = round(ms, 2)
    if METRICS_PATH.exists():
        old = pd.read_csv(METRICS_PATH)
        old = old[~((old["system"] == label) & (old["split"] == split))]
        agg = pd.concat([old, agg], ignore_index=True)
    order = {s: i for i, s in enumerate(config.SYSTEMS)}
    agg["_o"] = agg["system"].map(lambda s: (order.get(s.split("+")[0], len(order)), s))
    agg = agg.sort_values(["_o", "split"], kind="stable").drop(columns="_o")
    agg.to_csv(METRICS_PATH, index=False, float_format="%.4f")


def main() -> None:
    """Batch evaluation of retrieval systems."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["all"], help="system ids (S0 S0c S1 ...) or 'all'")
    ap.add_argument("--split", choices=["eval", "tune"], default="eval")
    ap.add_argument("--n", type=int, default=None, help="evaluate only the first n questions of the split")
    ap.add_argument("--index-variant", default="", choices=["", "nostem"])
    ap.add_argument("--fusion", default=None, choices=["rrf", "combsum", "chain"])
    args = ap.parse_args()

    systems = list(config.SYSTEMS) if args.systems == ["all"] else args.systems
    unknown = [s for s in systems if s not in config.SYSTEMS]
    if unknown:
        ap.error(f"unknown systems: {unknown}")

    from agent.resources import load_resources
    needs_ranker = any(config.SYSTEMS[s].get("bridge") == "ltr" for s in systems)
    res = load_resources("dev", with_ranker=needs_ranker)
    qrecs = load_questions()
    qrels = build_qrels(qrecs, res.title2doc)
    hop2_gold = load_hop2_gold(res, qrecs, qrels)
    if args.index_variant:
        from index.inverted_index import load_index
        res.index = load_index("dev", args.index_variant)

    qids = load_split(args.split)[:args.n]
    for name in systems:
        cfg = dict(config.SYSTEMS[name])
        label = name
        if args.fusion:
            cfg["fusion"] = args.fusion
            label += f"+{args.fusion}"
        if args.index_variant:
            label += f"+{args.index_variant}"
        pq, agg, ms = run_one(label, cfg, qids, qrecs, qrels, hop2_gold, res)
        write_results(label, args.split, pq, agg, ms)
        row = agg.set_index("type").loc["all"]
        print(f"{label} [{args.split}, n={len(qids)}] joint@10={row['joint@10']:.3f} recall@10={row['recall@10']:.3f} "
              f"mrr={row['mrr']:.3f} hop2_recall@10={row['hop2_recall@10']:.3f} ({ms:.1f} ms/q)")


if __name__ == "__main__":
    main()
