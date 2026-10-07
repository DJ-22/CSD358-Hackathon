"""Hyperparameter tuning on the TUNE split only: W_TITLE (S0), ALPHA x TITLE_BONUS (S3B), chain lambdas (S5)."""
import sys, pathlib  # bootstrap: make the repo root importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import copy
import json
from itertools import product

from tqdm import tqdm

import config
from eval.metrics import joint_at_k

TUNED_PATH = config.RESULTS_DIR / "tuned_params.json"
SWEEPS_PATH = config.RESULTS_DIR / "tune_sweeps.json"


def lambda_grid(step: float) -> list[tuple[float, float, float]]:
    """All (l1, l2, l3) on a simplex grid with the given step (each >= 0, sum = 1)."""
    n = round(1 / step)
    return [(round(i * step, 4), round(j * step, 4), round((n - i - j) * step, 4))
            for i, j in product(range(n + 1), repeat=2) if i + j <= n]


def mean_joint10(rankings: dict[str, list[int]], qrels: dict[str, list[int]]) -> float:
    """Mean joint@10 over the tuning questions."""
    return sum(joint_at_k(rankings[q], qrels[q], 10) for q in qrels) / len(qrels)


def main() -> None:
    """Re-tune from the config defaults: set any existing tuned_params.json aside (restored if tuning fails)."""
    if not TUNED_PATH.exists():
        tune_all()
        return
    import importlib
    backup = config.CACHE_DIR / "tuned_params.prev.json"
    backup.parent.mkdir(parents=True, exist_ok=True)
    TUNED_PATH.replace(backup)
    importlib.reload(config)                                  # config without tuned overrides = the defaults
    try:
        tune_all()
    except BaseException:
        backup.replace(TUNED_PATH)
        raise


def tune_all() -> None:
    """Grid-search W_TITLE, ALPHA x TITLE_BONUS and the chain lambdas on TUNE; write tuned_params.json and sweeps."""
    from agent.chains import final_ranking, score_chains
    from agent.pipeline import run_system
    from agent.resources import load_resources
    from eval.run_eval import build_qrels, load_questions, load_split

    tune = load_split("tune")
    eval_qids = set(load_split("eval"))
    assert not eval_qids & set(tune), "TUNE overlaps EVAL"
    all_q = load_questions()
    qrecs = {q: all_q[q] for q in tune}                      # only TUNE questions are ever read below
    del all_q
    res = load_resources("dev", with_ranker=True)
    qrels = build_qrels(qrecs, res.title2doc)
    assert not eval_qids & set(qrels), "an EVAL question reached the tuner"

    def run(cfg: dict, desc: str) -> tuple[dict, dict]:
        """Run a system over TUNE: (rankings, states)."""
        rankings, states = {}, {}
        for qid in tqdm(tune, desc=desc, leave=False):
            ranking, state = run_system(qrecs[qid], cfg, res)
            rankings[qid], states[qid] = [d for d, _ in ranking], state
        return rankings, states

    sweeps = {}
    # 1. zone weight of the title field, single-shot BM25
    sweeps["W_TITLE"] = []
    for w in config.W_TITLE_GRID:
        config.W_TITLE = w
        score = mean_joint10(run(config.SYSTEMS["S0"], f"S0 W_TITLE={w}")[0], qrels)
        sweeps["W_TITLE"].append({"value": w, "joint@10": score})
        print(f"W_TITLE={w}: S0 joint@10 {score:.4f}")
    best_w = max(sweeps["W_TITLE"], key=lambda r: (r["joint@10"], -abs(r["value"] - 2.0)))["value"]
    config.W_TITLE = best_w

    # 2. hybrid hop 2: sparse/dense interpolation alpha x exact-title bonus, searched jointly
    sweeps["ALPHA"] = []
    for bonus, a in product(config.TITLE_BONUS_GRID, config.ALPHA_GRID):
        config.TITLE_BONUS = bonus
        cfg = dict(config.SYSTEMS["S3B"], alpha=a)
        score = mean_joint10(run(cfg, f"S3B alpha={a} bonus={bonus}")[0], qrels)
        sweeps["ALPHA"].append({"value": a, "title_bonus": bonus, "joint@10": score})
        print(f"ALPHA={a} TITLE_BONUS={bonus}: S3B joint@10 {score:.4f}")
    best = max(sweeps["ALPHA"], key=lambda r: (r["joint@10"], -abs(r["value"] - 0.5) - abs(r["title_bonus"] - 0.2)))
    best_a, best_bonus = best["value"], best["title_bonus"]
    config.ALPHA, config.TITLE_BONUS = best_a, best_bonus

    # 3. chain-score weights: hops run once, chains and final ranking re-scored per lambda
    _, states = run(dict(config.SYSTEMS["S5"], alpha=best_a), "S5 hops")
    sweeps["LAMBDAS"] = []
    for lam in lambda_grid(config.LAMBDA_STEP):
        cfg = dict(config.SYSTEMS["S5"], lambdas=lam)
        rankings = {}
        for qid, st in states.items():
            st = copy.deepcopy(st)
            score_chains(st, res, cfg)
            rankings[qid] = [d for d, _ in final_ranking(st, cfg)]
        sweeps["LAMBDAS"].append({"value": list(lam), "joint@10": mean_joint10(rankings, qrels)})
    best_l = max(sweeps["LAMBDAS"], key=lambda r: (r["joint@10"], -sum(abs(x - y) for x, y in
                                                                       zip(r["value"], (0.3, 0.3, 0.4)))))["value"]
    print(f"LAMBDAS={best_l}: S5 joint@10 {max(r['joint@10'] for r in sweeps['LAMBDAS']):.4f}")

    tuned = {"W_TITLE": best_w, "ALPHA": best_a, "TITLE_BONUS": best_bonus, "LAMBDAS": best_l}
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TUNED_PATH.write_text(json.dumps(tuned, indent=2) + "\n", encoding="utf-8")
    SWEEPS_PATH.write_text(json.dumps({"split": "tune", "n": len(tune), "metric": "joint@10", **sweeps}, indent=1)
                           + "\n", encoding="utf-8")
    print(f"wrote {TUNED_PATH.name}: {tuned}")


if __name__ == "__main__":
    main()
