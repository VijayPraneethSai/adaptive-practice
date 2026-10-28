# Benchmarks (synthetic data run)

Numbers below are from a full run of the pipeline end to end on **synthetic
data** (`adaptive_practice.data.synthetic`, 6,000 users, 3,000 questions,
414,180 rows), not the real Riiid/EdNet dataset. The real-data numbers from
the Colab notebook (`notebooks/colab.ipynb`) have not been re-verified against
this run and may differ — re-run there before quoting final numbers.

All 28 tests pass (`pytest`).

## Retrieval (two-tower vs. popularity baseline)

| metric | two-tower | popularity |
| --- | --- | --- |
| recall@10 | 0.034 | 0.028 (**~1.2x**) |
| recall@50 | 0.148 | 0.123 |
| recall@100 | 0.260 | 0.220 |
| ndcg@10 | 0.015 | 0.013 |

- HNSW/flat index overlap @100: 0.997
- Latency: flat p50 0.037ms / p99 0.088ms, HNSW p50 0.077ms / p99 0.161ms (sub-millisecond either way)

Note: this is a **~1.2x** lift on synthetic data, not the 3x figure that was
on an earlier resume draft — that number has not been reproduced here and
should not be quoted until it is.

## Ranking (transformer vs. GBM baseline)

| model | AUC (correct) | AUC (continue) | MAE (elapsed, s) |
| --- | --- | --- | --- |
| transformer ranker | 0.694 | 0.575 | 4.87 |
| GBM baseline | 0.747 | 0.603 | 4.38 |

The GBM baseline currently beats the transformer ranker on this synthetic
run (small 5-epoch training run, ~330K rows). Worth investigating before
claiming the transformer "outperforms" anything on a resume.

## Off-policy evaluation (validated against simulator ground truth)

Doubly-robust (DR) estimates land within ~1-3 points of ground truth across
all three objectives (engagement/learning/balanced), e.g. balanced: DR
(model propensity) 0.878 vs ground truth 0.886. IPS alone is biased
(0.40-0.56), confirming DR/SNIPS are the estimators worth trusting.

## Long-term simulated rollout (skill gain per 100 questions)

| objective | sessions | accuracy | skill gain / 100 q |
| --- | --- | --- | --- |
| app (current) | 2.81 | 0.465 | 0.327 |
| engagement-weighted | 2.52 | 0.461 | 0.292 |
| learning-weighted | 2.52 | 0.484 | 0.291 |
| balanced | 2.84 | 0.474 | 0.310 |

## How to reproduce

```
pip install -r requirements.txt
python -m adaptive_practice.data.synthetic --users 6000 --questions 3000 --out data/synthetic
python -m adaptive_practice.features.spark_job --data data/synthetic --out data/features
python -m adaptive_practice.retrieval.two_tower --features data/features --out runs/latest
python -m adaptive_practice.ranking.model --features data/features --out runs/latest
python -m adaptive_practice.ranking.baseline --features data/features --out runs/latest
python -m adaptive_practice.evaluation.simulated --run runs/latest --features data/features
```
