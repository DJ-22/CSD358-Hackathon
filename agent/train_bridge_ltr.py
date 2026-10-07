"""Train the learned bridge ranker (LambdaMART over IR features) and a logistic-regression baseline."""
import sys, pathlib  # bootstrap: make the repo root importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import json
import pickle
import time

import numpy as np
from tqdm import tqdm

import config
from agent.features import FEATURE_NAMES, candidate_matrix

FEATURES_CACHE = config.CACHE_DIR / "ltr_features_train.npz"
MODEL_PATH = config.MODELS_DIR / "bridge_ltr.txt"
LR_PATH = config.MODELS_DIR / "lr_baseline.pkl"


def generate_features() -> dict:
    """Run parse -> hop 1 -> bridge candidates -> features on every training question; label gold bridge entities."""
    from agent.bridge import bridge_candidates
    from agent.entities import title_surface
    from agent.parser import hop1, parse
    from agent.resources import load_resources

    res = load_resources("train", with_ranker=False)
    with open(config.DATA_PROCESSED / "questions_train.jsonl", encoding="utf-8") as f:
        qrecs = [json.loads(line) for line in f]
    X, y, sizes, qids = [], [], [], []
    no_cands = no_pos = 0
    for q in tqdm(qrecs, desc="LTR features", mininterval=5):
        state = parse(q["question"], res, q["qid"])
        cands = bridge_candidates(state, hop1(state, res), res)
        if not cands:
            no_cands += 1
            continue
        gold = {title_surface(t) for t in q["gold_titles"]} - state.known_entities
        labels = [int(c["entity"] in gold) for c in cands]
        if not any(labels):
            no_pos += 1
            continue
        X.append(candidate_matrix(state, cands, res))
        y.extend(labels)
        sizes.append(len(cands))
        qids.append(q["qid"])
    data = {"X": np.vstack(X), "y": np.array(y, dtype=np.int32), "sizes": np.array(sizes, dtype=np.int32),
            "qids": np.array(qids), "n_questions": len(qrecs), "no_cands": no_cands, "no_pos": no_pos}
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(FEATURES_CACHE, **data)
    return data


def split_groups(sizes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """90/10 train/held-out split by question (group), seeded."""
    rng = np.random.default_rng(config.SEED)
    order = rng.permutation(len(sizes))
    n_hold = int(round(config.LTR_HOLDOUT * len(sizes)))
    return np.sort(order[n_hold:]), np.sort(order[:n_hold])


def rows_of(groups: np.ndarray, offsets: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Row indices of the given question groups in the stacked feature matrix."""
    return np.concatenate([np.arange(offsets[g], offsets[g] + sizes[g]) for g in groups])


def hit_at_k(scores: np.ndarray, y: np.ndarray, groups: np.ndarray, offsets: np.ndarray, sizes: np.ndarray,
             k: int) -> float:
    """Bridge hit@k: fraction of questions whose gold bridge entity is among the top-k scored candidates."""
    hits = 0
    for g in groups:
        a, b = offsets[g], offsets[g] + sizes[g]
        order = np.argsort(-scores[a:b], kind="stable")[:k]
        hits += int(y[a:b][order].any())
    return hits / len(groups)


def plot_importance(names: list[str], gain: np.ndarray, path: pathlib.Path) -> None:
    """Horizontal bar chart of LambdaMART split-gain feature importance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from eval.plots import style_axes, SERIES, SURFACE, TEXT
    order = np.argsort(gain)
    share = 100 * gain[order] / gain.sum()
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    ax.barh([names[i] for i in order], share, color=SERIES[0], height=0.6)
    for yi, v in enumerate(share):
        ax.text(v + 0.5, yi, f"{v:.1f}%", va="center", fontsize=8, color=TEXT)
    ax.set_xlabel("Share of total split gain (%)")
    ax.set_title("Learned bridge ranker: feature importance (gain)", loc="left")
    style_axes(ax, grid_axis="x")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    """Learning to rank for bridge entities: features, LambdaMART + LR training, held-out hit@k report."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", action="store_true", help="regenerate the cached training features")
    args = ap.parse_args()

    t0 = time.time()
    if FEATURES_CACHE.exists() and not args.regen:
        data = dict(np.load(FEATURES_CACHE))
        print(f"[cache] loaded {FEATURES_CACHE.name}")
    else:
        data = generate_features()
    X, y, sizes = data["X"], data["y"], data["sizes"]
    n_q, no_cands, no_pos = int(data["n_questions"]), int(data["no_cands"]), int(data["no_pos"])
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    tr, ho = split_groups(sizes)
    tr_rows, ho_rows = rows_of(tr, offsets, sizes), rows_of(ho, offsets, sizes)
    print(f"features {X.shape} in {time.time() - t0:.0f} s; groups train {len(tr)} held-out {len(ho)}")

    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ranker = lgb.LGBMRanker(**config.LTR_PARAMS, verbose=-1)
    ranker.fit(X[tr_rows], y[tr_rows], group=sizes[tr])
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=config.SEED))
    lr.fit(X[tr_rows], y[tr_rows])

    scorers = {
        "hand_score (rule-based)": X[:, FEATURE_NAMES.index("hand_score")],
        "Logistic regression": lr.decision_function(X),
        "LambdaMART": ranker.predict(X),
    }
    ceiling = (n_q - no_cands - no_pos) / n_q
    rows = []
    for name, s in scorers.items():
        h1, h3 = (hit_at_k(s, y, ho, offsets, sizes, k) for k in (1, 3))
        rows.append((name, h1, h3))
        print(f"{name:26s} hit@1 {h1:.3f}  hit@3 {h3:.3f}")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ranker.booster_.save_model(str(MODEL_PATH))
    with open(LR_PATH, "wb") as f:
        pickle.dump(lr, f)
    gain = ranker.booster_.feature_importance(importance_type="gain")
    plot_importance(FEATURE_NAMES, gain, config.PLOTS_DIR / "ltr_feature_importance.png")

    lines = ["# Learned bridge ranker: held-out evaluation", "",
             f"Training questions: {n_q} bridge questions from the HotpotQA train shard (own train corpus and index).",
             f"- No bridge candidate at all: {no_cands}",
             f"- Candidates but no gold bridge entity among them (dropped): {no_pos}",
             f"- Kept question groups: {len(sizes)}; candidate-recall ceiling **{ceiling:.3f}**",
             f"- Split by question (seed {config.SEED}): {len(tr)} train / {len(ho)} held-out groups, "
             f"{len(tr_rows)} / {len(ho_rows)} candidates; {int(y.sum())} positives overall", "",
             "Hit@k on held-out groups (a gold bridge entity is among the top-k ranked candidates). "
             "The absolute column multiplies by the candidate-recall ceiling.", "",
             "| Ranker | hit@1 | hit@3 | hit@3 × ceiling |", "|---|---|---|---|"]
    for name, h1, h3 in rows:
        lines.append(f"| {name} | {h1:.3f} | {h3:.3f} | {h3 * ceiling:.3f} |")
    lines += ["", "LambdaMART: `LGBMRanker` " + ", ".join(f"{k}={v}" for k, v in config.LTR_PARAMS.items()) + ".",
              "Logistic regression: standardised features, pointwise binary labels.", "",
              "Feature importance (split gain): see `plots/ltr_feature_importance.png`.", "",
              "| Feature | gain share |", "|---|---|"]
    for i in np.argsort(-gain):
        lines.append(f"| {FEATURE_NAMES[i]} | {100 * gain[i] / gain.sum():.1f}% |")
    lines += ["", "Generated by `python agent/train_bridge_ltr.py`."]
    (config.RESULTS_DIR / "ltr_heldout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {MODEL_PATH.name}, {LR_PATH.name}, ltr_heldout.md, ltr_feature_importance.png "
          f"({time.time() - t0:.0f} s total)")


if __name__ == "__main__":
    main()
