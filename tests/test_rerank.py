import numpy as np
import pytest
import torch

from adaptive_practice.data.synthetic import write
from adaptive_practice.features import sequences
from adaptive_practice.features.spark_job import run
from adaptive_practice.policy.rerank import Engine, Objective, greedy_slate, learning_value, utility
from adaptive_practice.ranking.model import Ranker, RankerConfig, attach_question_parts, question_parts
from adaptive_practice.ranking.model import train as train_ranker
from adaptive_practice.retrieval.two_tower import TwoTower, TwoTowerConfig
from adaptive_practice.retrieval.two_tower import train as train_retriever


def test_learning_value_peaks_at_target():
    v = learning_value(np.array([0.2, 0.7, 0.99]), 0.7, 0.2)
    assert v[1] == 1.0 and v[0] < v[1] and v[2] < v[1]


def test_engagement_weight_prefers_sticky_candidates():
    scores = {"p_correct": np.array([0.7, 0.7]), "p_continue": np.array([0.5, 0.9]), "seconds": np.array([20.0, 20.0])}
    u = utility(scores, np.array([1, 2]), {}, Objective(learning=1, engagement=1, coverage=0, time=0))
    assert u[1] > u[0]


def test_slate_spreads_tags_when_coverage_matters():
    scores = {"p_correct": np.array([0.7, 0.69, 0.5]), "p_continue": np.full(3, 0.9), "seconds": np.full(3, 20.0)}
    tags = np.array([5, 5, 9])
    assert greedy_slate(scores, tags, {}, Objective(coverage=0.0), 2) == [0, 1]
    assert greedy_slate(scores, tags, {}, Objective(coverage=2.0), 2) == [0, 2]


@pytest.fixture(scope="module")
def engine_setup(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    out = tmp_path_factory.mktemp("features")
    write(raw, n_users=60, n_questions=300, seed=11)
    run(raw, out, memory="2g")
    torch.manual_seed(0)
    dev = torch.device("cpu")
    seq = sequences.load(out, "train")
    cont = sequences.containers(seq)
    bundles = sequences.load_bundles(out)
    rcfg = TwoTowerConfig(dim=16, history=10, batch_size=64, epochs=1, examples_per_epoch=2000)
    retriever = TwoTower(len(bundles), bundles["part"].to_numpy(), rcfg.dim)
    train_retriever(retriever, cont, rcfg, dev, log=lambda _: None)
    kcfg = RankerConfig(dim=32, heads=2, window=40, epochs=1, batch_size=32, windows_per_epoch=100)
    parts = question_parts(out)
    ranker = attach_question_parts(Ranker(len(parts) - 1, kcfg), parts)
    train_ranker(ranker, seq, kcfg, dev, log=lambda _: None)
    ranker.eval()
    engine = Engine(retriever, rcfg, ranker, sequences.question_table(out), sequences.bundle_questions(out, len(bundles)), k=20)
    return engine, seq, cont


def test_engine_policy_is_a_distribution_over_the_catalogue(engine_setup):
    engine, seq, cont = engine_setup
    i = 5
    cand = engine.candidates(cont, [i])[0]
    assert len(set(cand.tolist())) == len(cand) == 20
    d = engine.decide(seq, cont.user_index[i], cont.first_row[i], cand, cont.lag[i], cont.position[i], 0)
    total = sum(engine.action_prob(d, b) for b in range(engine.n_bundles))
    assert abs(total - 1.0) < 1e-9
    heavy = engine.reweight(d, seq, cont.first_row[i], Objective(learning=0, engagement=5, coverage=0, time=0))
    assert heavy.candidates[np.argmax(heavy.probs)] == d.candidates[np.argmax(d.scores["p_continue"])]
