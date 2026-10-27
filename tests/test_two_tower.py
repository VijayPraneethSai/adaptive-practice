import numpy as np
import torch

from adaptive_practice.features import sequences
from adaptive_practice.features.sequences import containers
from adaptive_practice.retrieval.two_tower import (
    TwoTower, TwoTowerConfig, build_batch, decision_indices, ranking_metrics, to_device, train, user_vectors,
)


def tiny_sequences():
    rng = np.random.default_rng(0)
    rows = []
    for user in range(40):
        for c in range(int(rng.integers(3, 30))):
            bundle = int(rng.integers(0, 4)) if user % 2 else int(rng.integers(4, 8))
            rows.append((user, c, bundle))
    n = len(rows)
    user, container, bundle = map(np.array, zip(*rows))
    offsets = np.r_[0, np.cumsum(np.bincount(user))]
    return sequences.Sequences(
        offsets=offsets, user_ids=np.unique(user), question=bundle + 1, bundle=bundle, part=bundle % 7 + 1,
        container=container, position=np.zeros(n, dtype=np.int64), lag=np.ones(n, dtype=np.int64),
        explained=np.zeros(n, dtype=np.int64), correct=rng.integers(0, 2, n), elapsed=np.ones(n, dtype=np.int64),
        log_elapsed=np.zeros(n), cont=np.ones(n, dtype=np.int64), propensity=np.full(n, np.nan),
    )


def test_batches_only_use_earlier_containers_of_the_same_user():
    cont = containers(tiny_sequences())
    idx = decision_indices(cont)
    batch = build_batch(cont, idx, history=5)
    for row, i in enumerate(idx):
        start = cont.offsets[cont.user_index[i]]
        expected = cont.bundle[max(start, i - 5):i] + 1
        n = batch["lengths"][row]
        assert n == len(expected)
        assert np.array_equal(batch["hist_bundle"][row, :n], expected)
        assert (batch["hist_bundle"][row, n:] == 0).all()


def test_training_learns_user_groups():
    torch.manual_seed(0)
    cont = containers(tiny_sequences())
    cfg = TwoTowerConfig(dim=16, history=5, batch_size=64, epochs=30, lr=1e-2, examples_per_epoch=10_000)
    model = TwoTower(8, np.arange(8) % 7 + 1, cfg.dim)
    train(model, cont, cfg, torch.device("cpu"), log=lambda _: None)
    idx = decision_indices(cont)
    batch = to_device(build_batch(cont, idx, cfg.history), torch.device("cpu"))
    with torch.no_grad():
        top = model.logits(user_vectors(model, batch), model.item_vectors()).topk(4).indices.numpy()
    group = cont.bundle[idx] >= 4
    assert ((top >= 4).all(axis=1) == group).mean() > 0.95


def test_ranking_metrics():
    ranked = np.array([[3, 1, 2], [0, 1, 2], [5, 6, 7]])
    m = ranking_metrics(ranked, np.array([3, 2, 9]), ks=(1, 3))
    assert m["recall@1"] == 1 / 3
    assert m["recall@3"] == 2 / 3
    assert abs(m["ndcg@3"] - (1 + 1 / np.log2(4)) / 3) < 1e-12
