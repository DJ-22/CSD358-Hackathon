# BridgeHop

Multi-hop retrieval over HotpotQA paragraphs using a readable chain of classic IR operations.

A multi-hop question such as *"When was the producer of the film Betrayal born?"* needs two paragraphs. The first,
*Betrayal (1983 film)*, shares words with the question. The second, *Sam Spiegel*, connects only through an
intermediate entity. Single-shot retrieval finds the first and usually misses the second. BridgeHop handles this in
steps:

1. Parse the question.
2. Retrieve a first hop with a self-built positional zone index, Boolean filtering and BM25.
3. Extract bridge entities from the hop-1 paragraphs and rank them, by hand score or with a learned ranker.
4. For each bridge, retrieve a second hop: a phrase filter, then hybrid sparse + dense scoring.
5. Score the resulting reasoning chains and return a ranked list, plus a JSON trace of every decision.

Everything in the retrieval path is built from scratch: the inverted index, Boolean / phrase / BM25 / cosine
retrieval, PRF and fusion. No search libraries and no LLMs are used. The only downloads are the HotpotQA data and the
MiniLM sentence encoder.

## Pipeline

```
question
   │
   ▼
[PARSE]   question type (bridge / comparison) · idf-weighted query terms · title-surface entity linking
   │
   ▼
[HOP 1]   Boolean AND over the top-idf terms in df order (OR fallback)  ∪  free-text BM25 top 100
   │      → ranked by zone BM25 (W_TITLE·title + W_BODY·body)
   ▼
[STATE]   query consumption: terms matched by the top doc or a question entity are consumed;
   │      the rest form the residual query
   ▼
[BRIDGE]  entity-restricted feedback over the top hop-1 docs → bridge candidates
   │      → hand score  or  learned LambdaMART ranker (12 IR features) → top 3 bridges, p(e)
   │      (comparison questions: one second hop per slot entity instead)
   ▼
[HOP 2]   per bridge e:  BM25 mode   BM25(residual + entity) over the whole index
   │                  or hybrid mode phrase filter "e" in title ∪ body → cap 50 by BM25(residual) → minus source doc
   │                                 s_B = α·minmax(BM25) + (1−α)·cos(MiniLM(residual), MiniLM(doc)) + title bonus
   ▼
[CHAINS]  S = λ1·s1(d1) + λ2·p(e) + λ3·s_B(d2)   (each term min-max normalised per question)
   │
   ▼
[FINAL]   chain-first ranking (or RRF / CombSUM over the hop lists) → top 20 + runs/<qid>.json trace
```

| System | What it adds |
|---|---|
| S0 / S0c | single-shot zone BM25 / lnc.ltc cosine |
| S1 | parser + Boolean / free-text hop 1 |
| S2 | S1 + Rocchio pseudo-relevance feedback |
| S3 | hand-scored bridges + BM25 hop 2 + chains |
| S4 | S3 with the learned bridge ranker |
| S3B | S3 with hybrid hop 2 |
| S5 | learned bridge ranker + hybrid hop 2 (full system) |

## Results

All numbers below are copied from the files in `results/` by a script and come from the 1,000 EVAL questions of the
HotpotQA distractor dev set. Parameters were tuned on a disjoint set of 300 TUNE questions only.

**joint@10 0.600 (S0, single-shot BM25) → 0.772 (S5, full system), +0.172 absolute.**

| system | joint@10 | joint@2 | recall@10 | mrr | hop2_recall@10 | bridge_hit@3 | joint@10 bridge | joint@10 comparison | ms/q |
|---|---|---|---|---|---|---|---|---|---|
| S0 | 0.600 | 0.262 | 0.786 | 0.849 | 0.601 | – | 0.522 | 0.932 | 4.6 |
| S0+nostem | 0.601 | 0.259 | 0.787 | 0.848 | 0.608 | – | 0.522 | 0.937 | 3.8 |
| S0c | 0.572 | 0.211 | 0.768 | 0.797 | 0.588 | – | 0.510 | 0.833 | 7.2 |
| S1 | 0.600 | 0.262 | 0.786 | 0.849 | 0.601 | – | 0.522 | 0.932 | 6.9 |
| S2 | 0.649 | 0.128 | 0.807 | 0.707 | 0.653 | – | 0.597 | 0.869 | 13.3 |
| S3 | 0.705 | 0.340 | 0.817 | 0.802 | 0.732 | 0.459 | 0.661 | 0.890 | 11.0 |
| S4 | 0.773 | 0.494 | 0.849 | 0.798 | 0.796 | 0.569 | 0.747 | 0.885 | 36.3 |
| S3B | 0.697 | 0.417 | 0.802 | 0.804 | 0.727 | 0.459 | 0.653 | 0.885 | 42.0 |
| S5 | 0.772 | 0.537 | 0.850 | 0.795 | 0.795 | 0.569 | 0.747 | 0.880 | 30.3 |
| S5+chain | 0.772 | 0.537 | 0.850 | 0.795 | 0.795 | 0.569 | 0.747 | 0.880 | 42.0 |
| S5+combsum | 0.835 | 0.215 | 0.900 | 0.711 | 0.839 | 0.569 | 0.811 | 0.937 | 49.4 |
| S5+nostem | 0.767 | 0.525 | 0.843 | 0.792 | 0.792 | 0.569 | 0.742 | 0.874 | 38.4 |
| S5+rrf | 0.787 | 0.178 | 0.868 | 0.672 | 0.805 | 0.569 | 0.758 | 0.911 | 53.1 |

`joint@k` is 1 when both gold paragraphs are in the top k. `hop2_recall@10` is the recall of the gold paragraph that
single-shot BM25 ranks lower, i.e. the one that needs the second hop. `bridge_hit@3` is 1 when the hop-2 gold entity is
among the three selected bridges.

### Bridge selection × hop-2 scoring

Each cell: joint@10 (all) / joint@10 (bridge questions) / bridge_hit@3 (bridge questions).

| bridge selection \ hop 2 | BM25 (residual + entity) | hybrid (phrase filter → sparse+dense) |
|---|---|---|
| hand score (rule-based) | S3: 0.705 / 0.661 / 0.459 | S3B: 0.697 / 0.653 / 0.459 |
| learned ranker (LambdaMART) | S4: 0.773 / 0.747 / 0.569 | S5: 0.772 / 0.747 / 0.569 |

### Learned bridge ranker (held-out training questions)

Training questions: 8000 bridge questions from the HotpotQA train shard (own train corpus and index).
- No bridge candidate at all: 2
- Candidates but no gold bridge entity among them (dropped): 2395
- Kept question groups: 5603; candidate-recall ceiling **0.700**
- Split by question (seed 42): 5043 train / 560 held-out groups, 94238 / 10651 candidates; 5793 positives overall

Hit@k on held-out groups (a gold bridge entity is among the top-k ranked candidates). The absolute column multiplies by the candidate-recall ceiling.

| Ranker | hit@1 | hit@3 | hit@3 × ceiling |
|---|---|---|---|
| hand_score (rule-based) | 0.362 | 0.746 | 0.523 |
| Logistic regression | 0.807 | 0.936 | 0.655 |
| LambdaMART | 0.832 | 0.952 | 0.667 |

LambdaMART: `LGBMRanker` objective=lambdarank, n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42.
Logistic regression: standardised features, pointwise binary labels.

Feature importance (split gain): see `plots/ltr_feature_importance.png`.

| Feature | gain share |
|---|---|
| lookahead_sparse | 49.6% |
| hand_score | 19.1% |
| n_tokens | 10.0% |
| mean_idf | 8.1% |
| src_hop1_score | 4.5% |
| lookahead_dense | 2.9% |
| min_dist_consumed | 2.5% |
| first_pos | 1.5% |
| tfidf_in_source | 1.0% |
| src_hop1_rank | 0.4% |
| n_src_docs | 0.2% |
| in_title_zone | 0.1% |

### Diagnostics

Question-type parser on EVAL:

Rule-based qtype vs HotpotQA `type` on 1000 EVAL questions: **accuracy 0.934** (934/1000).

| gold \ predicted | bridge | comparison | total | recall |
|---|---|---|---|---|
| bridge | 762 | 47 | 809 | 0.942 |
| comparison | 19 | 172 | 191 | 0.901 |
| precision | 0.976 | 0.785 | | |

Hop 2 in isolation, given the true hop-2 entity, on the TUNE bridge questions:

Hop 2 is given the true hop-2 gold title surface as the bridge entity and the parser's residual query, on all 236 bridge questions of TUNE (exclude = none). Recall of the hop-2 gold doc:

| hop-2 mode | alpha | recall@1 | recall@3 |
|---|---|---|---|
| bm25 | - | 0.877 | 0.936 |
| hybrid | 0 | 0.941 | 0.966 |
| hybrid | 0.25 | 0.936 | 0.966 |
| hybrid | 0.5 | 0.881 | 0.966 |
| hybrid | 0.75 | 0.826 | 0.936 |
| hybrid | 1.0 | 0.814 | 0.932 |

Sparse filter: gold doc inside the phrase-filtered candidate set for 0.966 of questions; mean candidate set size 8.8 (cap 50). Title bonus 0.5.
Hybrid hop 2 runtime: 13.8 ms per call.

Plots are in `results/plots/`: joint@10 by system and question type, recall@k, bridge hit rate, the α sweep, the λ
heat map, LTR feature importance, and the fusion and stemming ablations. `results/own_queries.md` reports P@2 / P@10 of
S0 vs S5 on our own questions.

### Measured runtimes

Measured on a laptop CPU (no GPU).

| step | time |
|---|---|
| build index: dev / train / dev unstemmed | 11.3 s / 11.7 s / 7.8 s |
| MiniLM embeddings: dev (66581 docs) / train (71269 docs) | 26 min / 27 min (43 docs/s) |
| hybrid hop 2 per call (TUNE oracle) | 13.8 ms |
| demo, S5 + `--compare S0`, per question after warm-up | mean 105 ms, max 285 ms (startup about 30 s) |

Per-question retrieval times of every system are in the `ms/q` column of the results table.

## What works

- **The full system beats single-shot BM25 by a wide margin.** The gain comes from bridge questions, where the
  second paragraph is reachable only through an intermediate entity. Compare `joint@10 bridge` and `hop2_recall@10`
  for S0 and S5 above.
- **The learned bridge ranker is the largest single improvement.** It chooses a gold bridge entity far more often than
  the hand score (S3 → S4, and the held-out table above).
- **Hybrid hop 2 puts the right pair at the very top.** With the tuned title bonus it ranks both gold paragraphs at
  positions 1–2 more often than BM25 hop 2 does (`joint@2` of S4 vs S5), while joint@10 stays level.
- **Every answer is inspectable.** `demo.py` prints and saves each decision: Boolean df-order, consumed and residual
  terms, bridge features, hop-2 scores and chain scores. `runs/` holds five committed traces, including a failure case.

## What is planned

- **Hop 1 adds nothing over S0 yet.** S1 equals S0 because the Boolean candidate set always contains the BM25 top 100,
  and most questions fall back from AND to OR. A planned change runs one sub-query per linked question entity and fuses
  the lists with RRF. This gave higher joint@10 on comparison questions in a TUNE trial.
- **Bridge candidate recall caps the ranker.** Many training questions have no gold bridge among the extracted
  candidates at all (the candidate-recall ceiling in the ranker table above). Wider extraction is the next step:
  entities from body text beyond the top hop-1 documents, and fuzzy title matching.
- **Comparison questions lose a little against S0.** S0 already finds both entities' paragraphs for most comparison
  questions (`joint@10 comparison`), and the slot hops sometimes push one of them down. A fallback to the hop-1 list
  when both slot entities are already in its top ranks is planned.
- **The ranking cut-off for fused lists.** CombSUM gives the best joint@10, but the chain-first ranking gives far better
  joint@2. A learned final-ranking step could combine both.
- **Answer extraction is out of scope.** BridgeHop retrieves the supporting paragraphs; it does not produce an answer
  string.

## Setup

Python 3.12. CPU only; a GPU is not needed.

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on Linux / macOS
pip install -r requirements.txt
```

## Reproduce

Run from the repository root.

```bash
# 1. Data: downloads the HotpotQA distractor dev set + one train shard, writes corpora, splits and LTR questions
python data/prepare.py

# 2. Indexes and dense embeddings (see the runtimes above; embeddings are the slow step)
python index/build_index.py --corpus dev
python index/build_index.py --corpus train
python index/build_index.py --corpus dev --no-stem      # only for the stemming ablation
python index/embed.py --corpus dev
python index/embed.py --corpus train

# 3. Learned bridge ranker (writes models/ and the feature-importance plot)
python agent/train_bridge_ltr.py

# 4. Optional: re-tune on TUNE only (results/tuned_params.json is committed and loaded by config.py)
python eval/tune.py

# 5. Evaluation on EVAL, ablations, plots and summary tables
python eval/run_eval.py --systems all
python eval/run_eval.py --systems S5 --fusion rrf
python eval/run_eval.py --systems S5 --fusion combsum
python eval/run_eval.py --systems S5 --fusion chain
python eval/run_eval.py --systems S0 S5 --index-variant nostem
python eval/plots.py

# Diagnostics
python agent/parser.py          # question-type accuracy -> results/parser_accuracy.md
python agent/hop2.py            # hop-2 oracle on TUNE -> results/hop2_oracle.md

# Own questions: S0 vs S5 P@2 / P@10 -> results/own_queries.md
python eval/run_own_queries.py

# Tests
pytest tests/
```

The embedding step is the slow one. To skip it, copy `cache/doc_emb_dev.npy` and `cache/doc_emb_train.npy` from a
machine that has already built them.

## Demo

```bash
python demo.py --qid 5a73595055429901807dafd6 --system S5 --compare S0   # a HotpotQA dev question
python demo.py --q "Which country is the director of the film Inception from?"   # free text (no gold)
python demo.py --replay 5abbf64055429965836003bc                         # re-render a saved trace
```

The demo prints `[PARSE]`, `[HOP 1]`, `[STATE]`, `[BRIDGE]`, `[HOP 2]`, `[CHAINS]` and `[FINAL]`, marking gold
paragraphs with ★. It saves the trace to `runs/<qid>.json`, and `--replay` renders the same output without loading any
model. `--system` selects any of S0–S5, and `--compare S0` adds each gold paragraph's rank under the baseline.

Committed traces in `runs/` (all S5 with `--compare S0`):

| qid | question | shows |
|---|---|---|
| `5adcd5005542994d58a2f6f0` | Who voices the character that stars in Baseball Bugs? | bridge success |
| `5abd573c55429924427fcfb7` | Zach Parise's father played in which league? | bridge success, phrase filter leaves one doc |
| `5ae3fd2c5542995dadf2428f` | Are Ian Brown and Dee Snider both actors? | comparison: one hop 2 per entity |
| `5a73595055429901807dafd6` | When was the producer of the film Betrayal born? | S0 misses the second paragraph, S5 recovers it |
| `5abbf64055429965836003bc` | What county is Keene High School located in? | failure: wrong bridge chosen, gold bridge never extracted |

## Repository map

```
config.py                    all parameters (tuned values loaded from results/tuned_params.json)
demo.py                      inspectable trace for one question
data/prepare.py              download + corpora, EVAL / TUNE splits, LTR training questions
data/own_queries.jsonl       own questions with judged gold titles
index/preprocess.py          tokenise, stop words, Porter stemming, positions
index/inverted_index.py      positional zone (title / body) inverted index, CSR arrays
index/build_index.py         builds index/index_<corpus>[_nostem].pkl
index/embed.py               MiniLM document embeddings -> cache/doc_emb_<corpus>.npy
retrieval/bm25.py            zone BM25, term-at-a-time, heap top-k
retrieval/cosine.py          lnc.ltc cosine
retrieval/boolean.py         Boolean AND (df order) / OR, phrase queries
retrieval/prf.py             Rocchio pseudo-relevance feedback
agent/state.py               ResearchState trace (JSON round-trip)
agent/parser.py              question type, query terms, entities, hop 1
agent/entities.py            title-surface entity linker
agent/bridge.py              bridge candidates, hand score, residual query
agent/features.py            12 IR features for bridge ranking
agent/train_bridge_ltr.py    LambdaMART + logistic-regression training
agent/bridge_ranker.py       learned bridge ranking at query time
agent/hop2.py                hop 2: BM25 mode and hybrid phrase-filter -> dense mode
agent/chains.py              chain scoring and final ranking
agent/fusion.py              RRF, CombSUM
agent/resources.py           loads index, linker, embeddings, ranker
agent/pipeline.py            systems S0-S5
eval/metrics.py              recall@k, joint@k, P@k, MRR, hop-2 recall, bridge hit
eval/run_eval.py             evaluation harness
eval/tune.py                 grid search on TUNE
eval/plots.py                plots and summary tables
eval/run_own_queries.py      S0 vs S5 on own questions
tests/                       pytest suite
results/                     metrics, tables, diagnostics, plots
models/                      trained bridge ranker
runs/                        saved demo traces
```

Generated and gitignored: `data/raw/`, `data/processed/`, `index/*.pkl`, `cache/`.

## Data and model credit

- **HotpotQA** (Yang et al., 2018), distractor setting, from the Hugging Face `hotpotqa/hotpot_qa` dataset.
  Licensed CC BY-SA 4.0.
- **all-MiniLM-L6-v2** (`sentence-transformers/all-MiniLM-L6-v2`), Apache 2.0. See its model card on Hugging Face.

## AI use

TODO (humans)
