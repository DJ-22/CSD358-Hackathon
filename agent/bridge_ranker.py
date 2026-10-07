"""Inference with the learned bridge ranker."""
import numpy as np

import config
from agent.bridge import bridge_candidates
from agent.features import FEATURE_NAMES, candidate_matrix

MODEL_PATH = config.MODELS_DIR / "bridge_ltr.txt"


def load_ranker():
    """Load the trained LambdaMART bridge ranker (None if it has not been trained yet)."""
    if not MODEL_PATH.exists():
        return None
    import lightgbm as lgb
    return lgb.Booster(model_file=str(MODEL_PATH))


def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax over ranker scores: a probability distribution over a question's bridge candidates."""
    z = np.exp(x - x.max())
    return z / z.sum()


def rank_bridges(state, hop1_results, res, k: int = 3) -> list[tuple[str, float, int]]:
    """Learning-to-rank inference: (entity, p_e softmax, source_doc) for the top-k bridges."""
    if res.ranker is None:
        raise RuntimeError("no bridge ranker loaded: run python agent/train_bridge_ltr.py")
    cands = bridge_candidates(state, hop1_results, res)
    if not cands:
        return []
    X = candidate_matrix(state, cands, res)
    scores = res.ranker.predict(X)
    p = softmax(scores)
    order = sorted(range(len(cands)), key=lambda i: (-scores[i], i))[:k]
    out = []
    for i in order:
        c = cands[i]
        state.bridges.append({"entity": c["entity"], "source_doc": c["source_doc"], "hand_score": c["hand_score"],
                              "p_e": float(p[i]), "ltr_score": float(scores[i]),
                              "features": dict(zip(FEATURE_NAMES, X[i].tolist()))})
        out.append((c["entity"], float(p[i]), c["source_doc"]))
    return out
