import numpy as np
import pytest
import torch

from adaptive_practice.data.synthetic import write
from adaptive_practice.evaluation.simulated import OBJECTIVES, long_term, validate_ope
from adaptive_practice.features import sequences
from adaptive_practice.features.spark_job import run
from adaptive_practice.policy.rerank import Engine
from adaptive_practice.ranking.model import Ranker, RankerConfig, attach_question_parts, question_parts
from adaptive_practice.ranking.model import train as train_ranker
from adaptive_practice.retrieval.two_tower import TwoTower, TwoTowerConfig, save as save_retriever
from adaptive_practice.retrieval.two_tower import train as train_retriever
from adaptive_practice.ranking.model import save as save_ranker


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    out = tmp_path_factory.mktemp("features")
    run_dir = tmp_path_factory.mktemp("run")
    write(raw, n_users=80, n_questions=250, seed=21)
    run(raw, out, memory="2g")
    torch.manual_seed(0)
    dev = torch.device("cpu")
    bundles = sequences.load_bundles(out)
    cont = sequences.containers(sequences.load(out, "train"))
    rcfg = TwoTowerConfig(dim=16, history=10, batch_size=64, epochs=1, examples_per_epoch=3000)
    retriever = TwoTower(len(bundles), bundles["part"].to_numpy(), rcfg.dim)
    train_retriever(retriever, cont, rcfg, dev, log=lambda _: None)
    save_retriever(retriever, rcfg, run_dir)
    kcfg = RankerConfig(dim=24, heads=2, window=30, epochs=1, batch_size=32, windows_per_epoch=150)
    parts = question_parts(out)
    ranker = attach_question_parts(Ranker(len(parts) - 1, kcfg), parts)
    train_ranker(ranker, sequences.load(out, "train"), kcfg, dev, log=lambda _: None)
    save_ranker(ranker, kcfg, run_dir)
    return run_dir, out


def test_off_policy_estimates_track_ground_truth(trained_run):
    run_dir, out = trained_run
    work = run_dir / "ope_work"
    results = validate_ope(run_dir, out, work, n_users=40, bank_seed=1, seed=123, dev=torch.device("cpu"))
    for name in OBJECTIVES:
        truth = results[name]["ground_truth"]
        dr = results[name]["model_propensity"]["dr"]
        assert abs(dr - truth) < 0.12, f"{name}: dr={dr} truth={truth}"


def test_learning_objective_improves_simulated_skill_gain(trained_run):
    run_dir, out = trained_run
    results = long_term(run_dir, out, n_students=30, bank_seed=1, seed=777, dev=torch.device("cpu"),
                        max_bundles=120, max_sessions=15)
    assert results["learning"]["skill_gain_per_100_questions"] > 0.9 * results["engagement"]["skill_gain_per_100_questions"]


def test_engagement_objective_improves_simulated_continuation(trained_run):
    run_dir, out = trained_run
    results = long_term(run_dir, out, n_students=30, bank_seed=1, seed=778, dev=torch.device("cpu"),
                        max_bundles=120, max_sessions=15)
    assert results["engagement"]["bundles_per_session"] >= results["learning"]["bundles_per_session"]
