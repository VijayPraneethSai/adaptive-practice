import numpy as np
import pytest

from adaptive_practice.evaluation import ope


def test_ips_is_unbiased_in_expectation():
    rng = np.random.default_rng(0)
    mean_reward_a, mean_reward_b = 0.65, 0.3
    n = 50000
    chose_a = rng.random(n) < 0.4
    reward = np.where(chose_a, rng.random(n) < mean_reward_a, rng.random(n) < mean_reward_b).astype(np.float64)
    behavior = np.where(chose_a, 0.4, 0.6)
    target = np.where(chose_a, 1.0, 0.0)
    w = target / behavior
    assert abs(ope.ips(reward, w) - mean_reward_a) < 0.02


def test_snips_has_lower_variance_than_ips_under_weight_skew():
    rng = np.random.default_rng(1)
    n = 4000
    reward = rng.random(n)
    behavior = np.clip(rng.random(n), 0.05, 1.0)
    target = np.clip(rng.random(n), 0.05, 1.0)
    w = target / behavior
    ips_draws, snips_draws = [], []
    for seed in range(300):
        idx = np.random.default_rng(seed).integers(0, n, n)
        ips_draws.append(ope.ips(reward[idx], w[idx]))
        snips_draws.append(ope.snips(reward[idx], w[idx]))
    assert np.var(snips_draws) < np.var(ips_draws)


def test_doubly_robust_matches_dm_when_model_is_exact():
    rng = np.random.default_rng(2)
    n = 2000
    q_target = rng.random(n)
    reward = q_target
    w = np.ones(n)
    assert abs(ope.doubly_robust(reward, w, q_target, q_target) - np.mean(q_target)) < 1e-9


def test_doubly_robust_corrects_a_biased_model():
    rng = np.random.default_rng(3)
    n = 20000
    true_value = 0.7
    behavior = np.full(n, 0.5)
    target = np.full(n, 0.5)
    reward = (rng.random(n) < true_value).astype(np.float64)
    q_logged_wrong = np.full(n, 0.2)
    q_target_wrong = np.full(n, 0.2)
    w = target / behavior
    assert abs(ope.doubly_robust(reward, w, q_logged_wrong, q_target_wrong) - true_value) < 0.02


def test_effective_sample_size_bounds():
    assert ope.effective_sample_size(np.ones(100)) == pytest.approx(100.0)
    w = np.zeros(100)
    w[0] = 1.0
    assert ope.effective_sample_size(w) == pytest.approx(1.0)


def test_clipping_trades_bias_for_lower_variance():
    rng = np.random.default_rng(4)
    n = 3000
    reward = rng.random(n)
    behavior = np.clip(rng.exponential(0.1, n), 0.01, 1.0)
    target = np.full(n, 0.5)
    w = target / behavior
    unclipped = [ope.ips(reward[np.random.default_rng(s).integers(0, n, n)], w[np.random.default_rng(s).integers(0, n, n)])
                 for s in range(100)]
    clipped_w = np.minimum(w, 5.0)
    clipped = [ope.ips(reward[np.random.default_rng(s).integers(0, n, n)], clipped_w[np.random.default_rng(s).integers(0, n, n)])
               for s in range(100)]
    assert np.var(clipped) < np.var(unclipped)
