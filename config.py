"""BridgeHop configuration: every hyperparameter lives here."""
import json
from pathlib import Path

# ===== INDEX & RETRIEVAL =====
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
INDEX_DIR = ROOT / "index"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = ROOT / "models"
RUNS_DIR = ROOT / "runs"
LOGS_DIR = ROOT / "logs"

SEED = 42

DEV_URL = ("https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/"
           "distractor/validation-00000-of-00001.parquet")
TRAIN_URL = ("https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/"
             "distractor/train-00000-of-00002.parquet")
DEV_PARQUET = DATA_RAW / "dev_distractor.parquet"
TRAIN_PARQUET = DATA_RAW / "train_distractor_0.parquet"

EVAL_N = 1000
TUNE_N = 300
LTR_TRAIN_N = 8000

USE_STEMMING = True
K1 = 1.2
B = 0.75
W_TITLE = 2.0
W_BODY = 1.0
TOP_K = 20
BOOL_TOP_TERMS = 3
FREE_TEXT_K = 100
PRF_TOP_M = 5
PRF_N_TERMS = 10
PRF_BETA = 0.5

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMB_BATCH = 128
HOP2_FILTER_CAP = 50
HOP2_TOP = 3
ALPHA = 0.5
TITLE_BONUS = 0.2

# ===== AGENT & EVALUATION =====
MAX_ENTITY_DF = 2000
MAX_NGRAM = 6
TOP_M = 5
BEAM = 3
LAMBDAS = (0.3, 0.3, 0.4)
FINAL_K = 20
RRF_K = 60
LTR_PARAMS = {
    "objective": "lambdarank",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": SEED,
}
LTR_HOLDOUT = 0.1
ALPHA_GRID = [0, 0.25, 0.5, 0.75, 1.0]
LAMBDA_STEP = 0.1
W_TITLE_GRID = [1.0, 2.0, 3.0]
SYSTEMS = {
    "S0": {"hops": 1, "scorer": "bm25"},
    "S0c": {"hops": 1, "scorer": "cosine"},
    "S1": {"decompose": True},
    "S2": {"decompose": True, "prf": True},
    "S3": {"decompose": True, "bridge": "hand", "hop2": "bm25"},
    "S4": {"decompose": True, "bridge": "ltr", "hop2": "bm25"},
    "S3B": {"decompose": True, "bridge": "hand", "hop2": "hybrid"},
    "S5": {"decompose": True, "bridge": "ltr", "hop2": "hybrid"},
}

# ===== TUNED OVERRIDES (do not edit) =====
_TUNED = RESULTS_DIR / "tuned_params.json"
if _TUNED.exists():
    for _name, _value in json.loads(_TUNED.read_text(encoding="utf-8")).items():
        if _name in globals():
            globals()[_name] = tuple(_value) if isinstance(globals()[_name], tuple) else _value
