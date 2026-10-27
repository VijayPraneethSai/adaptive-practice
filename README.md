# adaptive-practice

Picks the next question for a student practicing for an exam, instead of
handing them a fixed worksheet. Built on the Riiid/EdNet dataset (100M+ real
logs from an exam-prep app) and the recommendation stack behind feeds like
this: retrieve candidates, rank them on more than one signal, and trade off
what the slate optimizes for.

Practice apps lose students two ways: the next question is too easy or too
hard, or it's unrelated to what they're weak on. Picking well means
balancing how much a question teaches against whether the student sticks
around and whether practice covers the syllabus — the same trade-off a feed
makes between engagement, retention and diversity.

## Pipeline

1. **Features** (`adaptive_practice/features/spark_job.py`) — a PySpark job
   turns the raw interaction log into per-container features: prior
   accuracy overall, per part, per tag and within the current session,
   lecture counts, lag since the last session, and the next container's
   correctness/time/continue labels. Verified leak-free against an
   independent pandas reference (`tests/test_spark_job.py`).
2. **Retrieval** (`adaptive_practice/retrieval/two_tower.py`) — a two-tower
   model embeds a student's recent history (a GRU over past bundles) and
   every question bundle, trained with in-batch softmax. Candidates come
   back through a FAISS index (flat or HNSW) in under a tenth of a
   millisecond.
3. **Ranking** (`adaptive_practice/ranking/model.py`) — a causal
   transformer encodes the student's history, then a cross-attention
   decoder scores each candidate; three MMoE heads predict whether the
   student answers correctly, how long they'll take, and whether they
   keep going. A gradient-boosting baseline (`ranking/baseline.py`) is
   trained on the same features for comparison.
4. **Re-ranking** (`adaptive_practice/policy/rerank.py`) — a tunable
   blend of learning value (peaked near 70% predicted accuracy),
   engagement, tag coverage and time cost picks the next bundle from the
   candidate set.
5. **Evaluation** (`adaptive_practice/evaluation/`) — no live users, so the
   policy is checked two ways instead of an A/B test:
   - **Off-policy evaluation** (`ope.py`): IPS, SNIPS and doubly-robust
     estimates of how a new policy would have done on logged data,
     validated against a student simulator's *known* true answer
     (`simulated.py`, `tests/test_simulated_eval.py`) — the doubly-robust
     estimate lands within a couple of points of ground truth.
   - **Long-term simulated rollouts**: multi-session students (with
     learning, fatigue and return behavior) are simulated under each
     objective to see the effect on skill gained and sessions returned.

## Run it

```
pip install -r requirements.txt

# synthetic data (no download needed)
python -m adaptive_practice.data.synthetic --users 6000 --questions 3000 --out data/synthetic
python -m adaptive_practice.features.spark_job --data data/synthetic --out data/features
python -m adaptive_practice.retrieval.two_tower --features data/features --out runs/latest
python -m adaptive_practice.ranking.model --features data/features --out runs/latest
python -m adaptive_practice.ranking.baseline --features data/features --out runs/latest
python -m adaptive_practice.evaluation.simulated --run runs/latest --features data/features

# try it yourself
python -m scripts.demo --run runs/latest --features data/features --objective balanced
```

For the real Riiid/EdNet data (100M+ rows), see `notebooks/colab.ipynb` —
it downloads the data with the Kaggle API, runs the same pipeline on a
Colab T4 GPU, and reports retrieval recall/latency and ranker AUC on real
held-out students.

## Test

```
pytest
```
