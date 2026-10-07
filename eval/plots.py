"""Evaluation plots and summary tables for the report (matplotlib, one figure per file)."""
import sys, pathlib  # bootstrap: make the repo root importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json

import numpy as np
import pandas as pd

import config

# categorical series colours in fixed order (colour-blind-checked), recessive greys for text and axes
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
MAIN_SYSTEMS = ["S0", "S0c", "S1", "S2", "S3", "S4", "S3B", "S5"]
SYSTEM_LABELS = {"S0": "S0\nBM25", "S0c": "S0c\ncosine", "S1": "S1\nparse+hop1", "S2": "S2\n+PRF",
                 "S3": "S3\nbridge", "S4": "S4\n+LTR", "S3B": "S3B\n+hybrid", "S5": "S5\nfull"}


def style_axes(ax, grid_axis: str = "y") -> None:
    """Recessive axes: no top/right spines, light grid behind the marks, muted tick labels."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fig(w: float = 8, h: float = 4.5):
    """New figure on the chart surface."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(w, h), facecolor=SURFACE)
    return plt, fig, ax


def _save(plt, fig, name: str) -> None:
    """Write one figure into the plots folder."""
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / name, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote plots/{name}")


def _legend(ax, **kw) -> None:
    """Frameless legend in primary text colour."""
    leg = ax.legend(frameon=False, fontsize=9, **kw)
    for t in leg.get_texts():
        t.set_color(TEXT)


def _grouped_bars(ax, groups: list[str], series: dict[str, list[float]], width: float = 0.26,
                  label_series: str | None = None) -> None:
    """Grouped vertical bars, one colour per series in fixed order, small gaps between adjacent bars."""
    x = np.arange(len(groups))
    n = len(series)
    for i, (name, vals) in enumerate(series.items()):
        pos = x + (i - (n - 1) / 2) * width
        ax.bar(pos, vals, width * 0.9, color=SERIES[i], label=name)
        if name == label_series:
            for p, v in zip(pos, vals):
                if not np.isnan(v):
                    ax.text(p, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=TEXT)
    ax.set_xticks(x, groups)


def metric(df: pd.DataFrame, system: str, typ: str, col: str) -> float:
    """One metric value from metrics.csv (EVAL split), NaN if missing."""
    row = df[(df["system"] == system) & (df["type"] == typ) & (df["split"] == "eval")]
    return float(row[col].iloc[0]) if len(row) else float("nan")


def plot_joint10(df: pd.DataFrame) -> None:
    """joint@10 per system, overall and by question type."""
    systems = [s for s in MAIN_SYSTEMS if not np.isnan(metric(df, s, "all", "joint@10"))]
    plt, fig, ax = _fig(9, 4.8)
    series = {t: [metric(df, s, t, "joint@10") for s in systems] for t in ("all", "bridge", "comparison")}
    _grouped_bars(ax, [SYSTEM_LABELS[s] for s in systems], series, label_series="all")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("joint@10 (both gold paragraphs in top 10)")
    ax.set_title("Joint recall@10 by system and question type (EVAL, n=1000)", loc="left")
    style_axes(ax)
    _legend(ax, loc="upper left", ncol=3)
    _save(plt, fig, "joint10_by_system_type.png")


def plot_recall_at_k(df: pd.DataFrame) -> None:
    """Recall@k curves for single-shot BM25, the rule-based agent and the full system."""
    ks = [2, 5, 10, 20]
    plt, fig, ax = _fig(7, 4.5)
    for i, s in enumerate(["S0", "S3", "S5"]):
        vals = [metric(df, s, "all", f"recall@{k}") for k in ks]
        ax.plot(ks, vals, color=SERIES[i], linewidth=2, marker="o", markersize=6, label=s)
        ax.annotate(f"{s} {vals[-1]:.2f}", (ks[-1], vals[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=TEXT)
    ax.set_xticks(ks)
    ax.set_xlim(1, 23)
    ax.set_xlabel("cut-off k")
    ax.set_ylabel("recall@k")
    ax.set_title("Recall@k: single-shot BM25 vs rule-based agent vs full system (EVAL)", loc="left")
    style_axes(ax)
    _legend(ax, loc="lower right")
    _save(plt, fig, "recall_at_k.png")


def plot_bridge_hit(df: pd.DataFrame) -> None:
    """Bridge hit rate@3 for the four bridge systems (diagnostic of bridge entity selection)."""
    systems = ["S3", "S4", "S3B", "S5"]
    vals = [metric(df, s, "bridge", "bridge_hit@3") for s in systems]
    plt, fig, ax = _fig(6.5, 4.2)
    ax.bar([SYSTEM_LABELS[s] for s in systems], vals, 0.55, color=SERIES[0])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, color=TEXT)
    ax.set_ylim(0, 1)
    ax.set_ylabel("bridge_hit@3 (hop-2 gold entity in top-3 bridges)")
    ax.set_title("Bridge hit rate@3 on EVAL bridge questions (n=809)", loc="left")
    style_axes(ax)
    _save(plt, fig, "bridge_hit.png")


def plot_alpha_sweep(sweeps: dict, tuned: dict) -> None:
    """TUNE joint@10 of S3B over the sparse/dense interpolation weight alpha."""
    pts = sweeps["ALPHA"]
    xs, ys = [p["value"] for p in pts], [p["joint@10"] for p in pts]
    plt, fig, ax = _fig(6.5, 4.2)
    ax.plot(xs, ys, color=SERIES[0], linewidth=2, marker="o", markersize=7)
    best = tuned["ALPHA"]
    yb = ys[xs.index(best)]
    ax.annotate(f"chosen α={best} ({yb:.3f})", (best, yb), xytext=(0, -14), textcoords="offset points",
                ha="center", va="top", fontsize=8.5, color=TEXT)
    pad = (max(ys) - min(ys)) * 0.15 or 0.01
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_xlabel("α  (1 = sparse BM25 only, 0 = dense MiniLM only)")
    ax.set_ylabel("S3B joint@10 on TUNE")
    ax.set_title(f"Hybrid hop 2: α sweep (TUNE, n={sweeps['n']})", loc="left")
    style_axes(ax)
    _save(plt, fig, "alpha_sweep.png")


def plot_lambda_heatmap(sweeps: dict, tuned: dict) -> None:
    """TUNE joint@10 of S5 over the chain weights (l1, l2), with l3 = 1 - l1 - l2."""
    from matplotlib.colors import LinearSegmentedColormap
    step = config.LAMBDA_STEP
    n = round(1 / step)
    grid = np.full((n + 1, n + 1), np.nan)
    for p in sweeps["LAMBDAS"]:
        l1, l2, _ = p["value"]
        grid[round(l2 / step), round(l1 / step)] = p["joint@10"]
    cmap = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"])
    cmap.set_bad(SURFACE)
    plt, fig, ax = _fig(6.8, 5.6)
    im = ax.imshow(grid, origin="lower", cmap=cmap, extent=(-step / 2, 1 + step / 2, -step / 2, 1 + step / 2))
    b1, b2, b3 = tuned["LAMBDAS"]
    ax.plot(b1, b2, marker="s", markersize=13, markerfacecolor="none", markeredgecolor=TEXT, markeredgewidth=1.5)
    ax.annotate(f"chosen ({b1}, {b2}, {b3})", (b1, b2), xytext=(10, 10), textcoords="offset points",
                fontsize=8.5, color=TEXT)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("S5 joint@10 on TUNE", color=TEXT)
    cb.outline.set_visible(False)
    ax.set_xlabel("λ1 (hop-1 score of d1)")
    ax.set_ylabel("λ2 (bridge probability p(e))")
    lo, hi = np.nanmin(grid), np.nanmax(grid)
    ax.set_title(f"Chain-score weights (λ3 = 1 − λ1 − λ2 on hop-2 score)\n"
                 f"joint@10 spans only {lo:.3f}–{hi:.3f} across the grid", loc="left", fontsize=10.5)
    style_axes(ax, grid_axis="both")
    ax.grid(False)
    _save(plt, fig, "lambda_heatmap.png")


def plot_fusion(df: pd.DataFrame) -> None:
    """Final-ranking ablation for the full system: chain-first vs RRF vs CombSUM."""
    labels = {"S5": "chain-first\n(default)", "S5+rrf": "RRF only", "S5+combsum": "CombSUM only"}
    systems = [s for s in labels if not np.isnan(metric(df, s, "all", "joint@10"))]
    plt, fig, ax = _fig(7, 4.3)
    series = {t: [metric(df, s, t, "joint@10") for s in systems] for t in ("all", "bridge", "comparison")}
    _grouped_bars(ax, [labels[s] for s in systems], series, label_series="all")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("joint@10")
    ax.set_title("Final-ranking fusion ablation for S5 (EVAL)", loc="left")
    style_axes(ax)
    _legend(ax, loc="upper right", ncol=3)
    _save(plt, fig, "fusion_ablation.png")


def plot_stemming(df: pd.DataFrame) -> None:
    """Stemming ablation: Porter-stemmed vs unstemmed index for S0 and S5."""
    systems = ["S0", "S5"]
    series = {"Porter stemming": [metric(df, s, "all", "joint@10") for s in systems],
              "no stemming": [metric(df, f"{s}+nostem", "all", "joint@10") for s in systems]}
    plt, fig, ax = _fig(6, 4.2)
    _grouped_bars(ax, systems, series, width=0.32)
    for i, (name, vals) in enumerate(series.items()):
        for j, v in enumerate(vals):
            ax.text(j + (i - 0.5) * 0.32, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=TEXT)
    ax.set_ylim(0, 1)
    ax.set_ylabel("joint@10")
    ax.set_title("Stemming ablation (EVAL)", loc="left")
    style_axes(ax)
    _legend(ax, loc="upper left")
    _save(plt, fig, "stemming_ablation.png")


def write_ablation_2x2(df: pd.DataFrame) -> None:
    """2x2 ablation table: bridge selection (hand / learned) x hop-2 scoring (BM25 / hybrid)."""
    cell = lambda s: (f"{metric(df, s, 'all', 'joint@10'):.3f} / {metric(df, s, 'bridge', 'joint@10'):.3f} / "
                      f"{metric(df, s, 'bridge', 'bridge_hit@3'):.3f}")
    lines = ["# 2x2 ablation: bridge selection x hop-2 scoring (EVAL, n=1000)", "",
             "Each cell: joint@10 (all) / joint@10 (bridge questions) / bridge_hit@3 (bridge questions).", "",
             "| bridge selection \\ hop 2 | BM25 (residual + entity) | hybrid (phrase filter → sparse+dense) |",
             "|---|---|---|",
             f"| hand score (rule-based) | S3: {cell('S3')} | S3B: {cell('S3B')} |",
             f"| learned ranker (LambdaMART) | S4: {cell('S4')} | S5: {cell('S5')} |", "",
             "Generated by `python eval/plots.py` from `results/metrics.csv`."]
    (config.RESULTS_DIR / "ablation_2x2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote ablation_2x2.md")


def write_summary(df: pd.DataFrame, tuned: dict | None) -> None:
    """Script-generated headline numbers per system for the report."""
    cols = ["joint@10", "joint@2", "recall@10", "mrr", "hop2_recall@10", "bridge_hit@3"]
    lines = ["# Results summary (EVAL, n=1000)", "",
             "| system | " + " | ".join(cols) + " | joint@10 bridge | joint@10 comparison | ms/q |",
             "|---|" + "---|" * (len(cols) + 3)]
    systems = [s for s in df[df["split"] == "eval"]["system"].unique()]
    for s in systems:
        vals = [metric(df, s, "all", c) for c in cols]
        cells = ["–" if np.isnan(v) else f"{v:.3f}" for v in vals]
        ms = df[(df["system"] == s) & (df["type"] == "all") & (df["split"] == "eval")]["ms_per_q"].iloc[0]
        lines.append(f"| {s} | " + " | ".join(cells) + f" | {metric(df, s, 'bridge', 'joint@10'):.3f} | "
                     f"{metric(df, s, 'comparison', 'joint@10'):.3f} | {ms:.1f} |")
    s0, s5 = metric(df, "S0", "all", "joint@10"), metric(df, "S5", "all", "joint@10")
    lines += ["", f"Headline: joint@10 {s0:.3f} (S0, single-shot BM25) → {s5:.3f} (S5, full system), "
              f"{s5 - s0:+.3f} absolute."]
    if tuned:
        lines += ["", "Tuned on TUNE only (300 questions, disjoint from EVAL): "
                  + ", ".join(f"{k}={v}" for k, v in tuned.items()) + "."]
    lines += ["", "Systems: S0 single-shot zone BM25 · S0c lnc.ltc cosine · S1 parser + Boolean/free-text hop 1 · "
              "S2 S1 + Rocchio PRF · S3 bridge (hand score) + residual + chains, hop-2 BM25 · S4 S3 with the "
              "learned bridge ranker · S3B S3 with hybrid hop 2 · S5 learned ranker + hybrid hop 2. "
              "`+rrf` / `+combsum` replace the chain-first final ranking; `+nostem` uses the unstemmed index.", "",
              "Generated by `python eval/plots.py` from `results/metrics.csv`."]
    (config.RESULTS_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote summary.md")


def main() -> None:
    """Render every evaluation plot and the summary tables from the results files."""
    df = pd.read_csv(config.RESULTS_DIR / "metrics.csv")
    sweeps_path = config.RESULTS_DIR / "tune_sweeps.json"
    tuned_path = config.RESULTS_DIR / "tuned_params.json"
    tuned = json.loads(tuned_path.read_text(encoding="utf-8")) if tuned_path.exists() else None
    plot_joint10(df)
    plot_recall_at_k(df)
    plot_bridge_hit(df)
    if sweeps_path.exists() and tuned:
        sweeps = json.loads(sweeps_path.read_text(encoding="utf-8"))
        plot_alpha_sweep(sweeps, tuned)
        plot_lambda_heatmap(sweeps, tuned)
    plot_fusion(df)
    plot_stemming(df)
    write_ablation_2x2(df)
    write_summary(df, tuned)


if __name__ == "__main__":
    main()
