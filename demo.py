"""BridgeHop live demo: prints the full agent trace for one question."""
import sys, pathlib  # bootstrap: make the repo root importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import argparse
import hashlib
import json
import time

import config

SYSTEM_NAMES = {
    "S0": "single-shot zone BM25", "S0c": "single-shot lnc.ltc cosine", "S1": "parser + Boolean/free-text hop 1",
    "S2": "S1 + Rocchio PRF", "S3": "hand-scored bridges + BM25 hop 2 + chains",
    "S4": "learned bridge ranker + BM25 hop 2 + chains", "S3B": "hand-scored bridges + hybrid hop 2 + chains",
    "S5": "learned bridge ranker + hybrid hop 2 + chains",
}
SHOW_HOP1 = 5
SHOW_FINAL = 10
SHOW_CHAINS = 3
SHOW_FEATURES = 3
TITLE_WIDTH = 42


# ---------------------------------------------------------------- running a question

def load_qrec(qid: str) -> dict:
    """Question record (text, type, gold titles) of a dev qid."""
    with open(config.DATA_PROCESSED / "questions_dev.jsonl", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            if q["qid"] == qid:
                return q
    raise SystemExit(f"unknown qid: {qid}")


def free_text_qrec(text: str) -> dict:
    """Question record for a free-text query (no gold judgements); the qid is a hash of the text."""
    return {"qid": "q-" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:10], "question": text,
            "type": None, "gold_titles": []}


def _hop2_gold(qrec: dict, gold: list[int], res) -> int | None:
    """Second-hop gold doc: the gold doc that single-shot BM25 scores lower."""
    from eval.metrics import compute_hop2_gold
    from retrieval.bm25 import bm25_search
    if len(gold) < 2:
        return None
    s0 = bm25_search(res.index, res.index.analyze(qrec["question"]), 2, candidates=set(gold))
    return compute_hop2_gold({"q": s0}, {"q": gold})["q"]


def _feature_contribs(bridges: list[dict], res) -> list[list[list]]:
    """Per bridge, the top features by absolute ranker contribution: [name, value, contribution]."""
    import numpy as np
    from agent.features import FEATURE_NAMES
    out = []
    for b in bridges:
        feats = b.get("features")
        if not feats or res.ranker is None or "ltr_score" not in b:
            out.append([])
            continue
        x = np.array([[feats[n] for n in FEATURE_NAMES]], dtype=np.float64)
        contrib = res.ranker.predict(x, pred_contrib=True)[0][:len(FEATURE_NAMES)]
        top = sorted(range(len(FEATURE_NAMES)), key=lambda i: (-abs(contrib[i]), i))[:SHOW_FEATURES]
        out.append([[FEATURE_NAMES[i], float(x[0, i]), float(contrib[i])] for i in top])
    return out


def build_trace(qrec: dict, system: str, res, compare: str | None = None) -> dict:
    """Run one system on one question and collect everything the trace view needs (JSON-safe)."""
    from agent.hop2 import sparse_filter
    from agent.pipeline import comparison_slots, run_system
    from eval.metrics import bridge_hit_at_k, joint_at_k, recall_at_k
    from retrieval.bm25 import bm25_zone_accumulators

    index = res.index
    cfg = dict(config.SYSTEMS[system])
    t0 = time.perf_counter()
    ranking, state = run_system(qrec, cfg, res)
    ms = 1000 * (time.perf_counter() - t0)
    docs = [int(d) for d, _ in ranking]
    gold = [res.title2doc[t] for t in qrec.get("gold_titles", []) if t in res.title2doc]

    hop1 = next(h for h in state.hops if h.get("hop") == 1)
    acc = bm25_zone_accumulators(index, hop1["query"])
    hop1_split = [[float(config.W_TITLE * acc["title"][d]), float(config.W_BODY * acc["body"][d])]
                  for d, _ in hop1["results"][:SHOW_HOP1]]
    df_any = {t: int(index.df_any[index.term_id(t)]) if index.term_id(t) >= 0 else 0
              for t in hop1.get("df_order", [])}

    hop2_filter = []
    for h in state.hops:
        if h.get("hop") == 2:
            n = len(sparse_filter(h["entity"], state, res, h["source_doc"])[0]) if h["mode"] == "hybrid" else None
            hop2_filter.append(n)

    metrics = {}
    if gold:
        hop2_gold = _hop2_gold(qrec, gold, res)
        metrics = {"joint@10": joint_at_k(docs, gold, 10), "recall@10": recall_at_k(docs, gold, 10),
                   "hop2_gold": hop2_gold, "hop2_gold_surface": res.linker.doc2surface.get(hop2_gold)}
        if cfg.get("bridge") and state.qtype == "bridge" and qrec.get("type") == "bridge":
            metrics["bridge_hit@3"] = bridge_hit_at_k([b["entity"] for b in state.bridges],
                                                      metrics["hop2_gold_surface"])

    comp = None
    if compare:
        base, _ = run_system(qrec, dict(config.SYSTEMS[compare]), res)
        base_docs = [int(d) for d, _ in base]
        rank = lambda lst, d: lst.index(d) + 1 if d in lst else None
        comp = {"system": compare, "ranks": [[g, rank(docs, g), rank(base_docs, g)] for g in gold]}

    slots = comparison_slots(state, res) if state.qtype == "comparison" and cfg.get("bridge") else []
    seen = set(gold) | set(docs) | {d for d, _ in hop1["results"]} | {b["source_doc"] for b in state.bridges}
    seen |= {d for h in state.hops if h.get("hop") == 2 for d, *_ in h["results"]}
    seen |= {d for h in state.hops if h.get("hop") == "prf" for d in h["feedback_docs"]}
    seen |= {d for c in state.chains for d in (c["d1"], c["d2"])}
    seen.discard(None)

    return {
        "qid": qrec["qid"], "question": qrec["question"], "system": system, "gold_type": qrec.get("type"),
        "gold": gold, "ms": round(ms, 1),
        "params": {"W_TITLE": config.W_TITLE, "ALPHA": cfg.get("alpha", config.ALPHA),
                   "TITLE_BONUS": config.TITLE_BONUS, "LAMBDAS": list(config.LAMBDAS)},
        "titles": {str(d): index.doc_title(int(d)) for d in sorted(seen)},
        "entities": [[s, start, len(ids)] for s, ids, start in sorted(res.linker.link(qrec["question"]),
                                                                       key=lambda e: e[2])],
        "df_any": df_any, "hop1_split": hop1_split, "hop2_filter": hop2_filter, "slots": slots,
        "bridge_features": _feature_contribs(state.bridges, res),
        "metrics": metrics, "compare": comp, "state": state.to_json(),
    }


def run_path(qid: str) -> pathlib.Path:
    """Saved trace location of a question."""
    return config.RUNS_DIR / f"{qid}.json"


def save_trace(trace: dict) -> pathlib.Path:
    """Write a trace to runs/<qid>.json."""
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = run_path(trace["qid"])
    path.write_text(json.dumps(trace, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------- rendering a trace

def _short(text: str, width: int = TITLE_WIDTH) -> str:
    return text if len(text) <= width else text[:width - 1] + "…"


def render(trace: dict, console=None) -> None:
    """Print the trace sections [PARSE] [HOP 1] [STATE] [BRIDGE] [HOP 2] [CHAINS] [FINAL] from a saved trace."""
    from rich import box
    from rich.console import Console
    from rich.table import Table

    if console is None:
        console = Console(highlight=False)
        if not console.is_terminal:
            console = Console(highlight=False, width=120)
    con = console
    st = trace["state"]
    gold = set(trace["gold"])
    titles = trace["titles"]
    star = lambda d: "★" if d in gold else ""
    doc = lambda d: f"{_short(titles.get(str(d), '?'))} [{d}]"

    def table(*cols):
        t = Table(box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False)
        for c in cols:
            t.add_column(c, justify="right" if c not in ("doc", "chain", "entity", "features", "") else "left")
        return t

    def head(name):
        con.print()
        con.print(f"[bold cyan]\\[{name}][/]")

    con.print(f"[bold]{trace['question']}[/]")
    con.print(f"qid {trace['qid']} · system {trace['system']} ({SYSTEM_NAMES.get(trace['system'], '')}) · "
              f"W_TITLE {trace['params']['W_TITLE']} · α {trace['params']['ALPHA']} · "
              f"title bonus {trace['params']['TITLE_BONUS']} · λ {tuple(trace['params']['LAMBDAS'])}")

    # PARSE
    head("PARSE")
    hop1 = next(h for h in st["hops"] if h.get("hop") == 1)
    single = "boolean_terms" not in hop1
    qtype = "n/a (single-shot)" if single else st["qtype"]
    label = f" · HotpotQA label: {trace['gold_type']}" if trace["gold_type"] else ""
    con.print(f"type: {qtype}{label}")
    ents = ", ".join(f"'{s}' (tok {start}, {n} doc{'s' if n != 1 else ''})" for s, start, n in trace["entities"])
    con.print(f"entities: {ents or '–'}")
    terms = sorted(st["query_terms"].items(), key=lambda kv: -kv[1])
    con.print("terms (idf): " + " · ".join(f"{t} {idf:.2f}" for t, idf in terms))

    # HOP 1
    head("HOP 1")
    if single:
        con.print(f"single-shot {hop1['scorer']} over all {len(hop1['query'])} question terms")
    else:
        order = " → ".join(f"{t} (df {trace['df_any'].get(t, 0)})" for t in hop1["df_order"])
        mode = "AND" if hop1["boolean_mode"] == "and" else "AND < TOP_K → OR fallback"
        con.print(f"Boolean over top-idf terms in df order: {order}")
        con.print(f"  {mode}: {hop1['boolean_count']} docs ∪ free-text BM25 top {hop1['free_text_count']} "
                  f"= {hop1['candidates']} candidates → ranked by zone BM25")
    t = table("#", "doc", "score", "title BM25", "body BM25", "")
    for i, ((d, s), (ts, bs)) in enumerate(zip(hop1["results"], trace["hop1_split"]), start=1):
        t.add_row(str(i), doc(d), f"{s:.2f}", f"{ts:.2f}", f"{bs:.2f}", star(d))
    con.print(t)
    for h in st["hops"]:
        if h.get("hop") == "prf":
            added = [f"{w_t} {w:.2f}" for w_t, w in h["query"].items() if w_t not in st["query_terms"]]
            con.print(f"PRF (Rocchio) from top {len(h['feedback_docs'])} docs, added terms: {' · '.join(added) or '–'}")

    # STATE
    head("STATE")
    if st["residual_terms"] or st["consumed_terms"]:
        con.print(f"consumed: {' · '.join(st['consumed_terms']) or '–'}")
        con.print(f"residual: {' · '.join(st['residual_terms']) or '–'}   (text: \"{st['residual_text']}\")")
    else:
        con.print("no query consumption in this system")

    # BRIDGE
    head("BRIDGE")
    if trace["slots"]:
        con.print("comparison question: one hop 2 per slot entity → " + ", ".join(f"'{s}'" for s in trace["slots"]))
    elif not st["bridges"]:
        con.print("no bridge step in this system" if single or not any(h.get("hop") == 2 for h in st["hops"])
                  else "no bridge candidates")
    else:
        t = table("#", "entity", "doc", "hand", "p_e", "features")
        for i, (b, feats) in enumerate(zip(st["bridges"], trace["bridge_features"]), start=1):
            ftxt = "\n".join(f"{n}={v:.2f} ({c:+.2f})" for n, v, c in feats) or "– (hand score)"
            t.add_row(str(i), b["entity"], doc(b["source_doc"]), f"{b['hand_score']:.2f}", f"{b['p_e']:.3f}", ftxt)
        con.print("source doc = hop-1 doc the bridge was extracted from · features: value (ranker contribution)")
        con.print(t)

    # HOP 2
    head("HOP 2")
    hop2 = [h for h in st["hops"] if h.get("hop") == 2]
    if not hop2:
        con.print("no second hop in this system")
    for h, n_filter in zip(hop2, trace["hop2_filter"]):
        src = f" · excluding source {h['source_doc']}" if h["source_doc"] is not None else ""
        filt = f"phrase filter → {n_filter} docs (cap {config.HOP2_FILTER_CAP})" if n_filter is not None \
            else "BM25(residual + entity) over the whole index"
        con.print(f"'{h['entity']}' · {h['mode']} · {filt}{src}")
        t = table("#", "doc", "s_sparse", "s_dense", "s_B", "")
        for i, (d, sp, de, sb) in enumerate(h["results"], start=1):
            t.add_row(str(i), doc(d), f"{sp:.3f}", f"{de:.3f}", f"{sb:.3f}", star(d))
        if h["results"]:
            con.print(t)
        else:
            con.print("  (no results)")

    # CHAINS
    head("CHAINS")
    if not st["chains"]:
        con.print("no chains (single-hop system or comparison question)")
    else:
        t = table("#", "chain", "s1", "p_e", "s_B", "S")
        for i, c in enumerate(st["chains"][:SHOW_CHAINS], start=1):
            t.add_row(str(i), f"{doc(c['d1'])} → '{c['entity']}' → {doc(c['d2'])}",
                      f"{c['s1']:.2f}", f"{c['p_e']:.3f}", f"{c['sB']:.3f}", f"{c['S']:.3f}")
        con.print(t)

    # FINAL
    head("FINAL")
    t = table("#", "doc", "score", "")
    for i, (d, s) in enumerate(st["final_ranking"][:SHOW_FINAL], start=1):
        t.add_row(str(i), doc(d), f"{s:.3f}", star(d))
    con.print(t)
    m = trace["metrics"]
    if not m:
        con.print("no gold judgements for this query")
    else:
        line = f"joint@10 {m['joint@10']:.0f} · recall@10 {m['recall@10']:.2f}"
        if m.get("hop2_gold") is not None:
            line += f" · hop-2 gold {doc(m['hop2_gold'])}"
        if "bridge_hit@3" in m:
            line += f" · bridge hit@3 {m['bridge_hit@3']:.0f} (gold bridge '{m['hop2_gold_surface']}')"
        con.print(line)
    if trace["compare"]:
        c = trace["compare"]
        fmt = lambda r: str(r) if r is not None else f">{config.FINAL_K}"
        con.print(f"gold ranks {trace['system']} vs {c['system']}: "
                  + " · ".join(f"{_short(titles[str(g)], 32)} {fmt(a)} vs {fmt(b)}" for g, a, b in c["ranks"]))
    con.print(f"[dim]{trace['ms']:.0f} ms for this question[/]")


# ---------------------------------------------------------------- command line

def main() -> None:
    """Render an inspectable multi-hop retrieval trace: --qid | --q | --replay."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="BridgeHop demo: full agent trace for one question.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--qid", help="HotpotQA dev question id")
    src.add_argument("--q", help="free-text question")
    src.add_argument("--replay", metavar="QID", help="render a saved runs/<qid>.json without running anything")
    ap.add_argument("--system", default="S5", choices=list(config.SYSTEMS))
    ap.add_argument("--compare", choices=list(config.SYSTEMS), help="also report each gold doc's rank under this system")
    args = ap.parse_args()

    if args.replay:
        path = run_path(args.replay)
        if not path.exists():
            raise SystemExit(f"no saved trace: {path}")
        render(json.loads(path.read_text(encoding="utf-8")))
        return

    from agent.resources import load_resources
    needs_ranker = any(config.SYSTEMS[s].get("bridge") == "ltr" for s in (args.system, args.compare) if s)
    res = load_resources("dev", with_ranker=needs_ranker)
    if config.SYSTEMS[args.system].get("hop2") == "hybrid":
        from agent.hop2 import _query_vec
        _query_vec("warm up")  # load the sentence encoder before timing
    qrec = load_qrec(args.qid) if args.qid else free_text_qrec(args.q)
    trace = build_trace(qrec, args.system, res, args.compare)
    path = save_trace(trace)
    render(json.loads(path.read_text(encoding="utf-8")))
    print(f"\nsaved {path.relative_to(config.ROOT)} · replay: python demo.py --replay {trace['qid']}")


if __name__ == "__main__":
    main()
