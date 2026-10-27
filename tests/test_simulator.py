import numpy as np

from adaptive_practice.data.synthetic import generate
from adaptive_practice.simulator import AppPolicy, Simulator, UserLog, make_bank


def test_app_policy_propensities_form_a_distribution():
    rng = np.random.default_rng(0)
    bank = make_bank(300, rng)
    policy = AppPolicy(bank)
    log = UserLog()
    log.attempts[:] = [5, 3, 9, 1, 0, 4, 2]
    log.hits[:] = [4, 1, 3, 1, 0, 4, 0]
    total = sum(policy.propensity(log, b) for b in range(bank.n_bundles))
    assert abs(total - 1.0) < 1e-9


def test_app_policy_sampling_matches_propensity():
    rng = np.random.default_rng(1)
    bank = make_bank(120, rng)
    policy = AppPolicy(bank)
    log = UserLog()
    counts = np.zeros(bank.n_bundles)
    n = 40000
    for _ in range(n):
        bundle, _ = policy.choose(log, rng)
        counts[bundle] += 1
    expected = np.array([policy.propensity(log, b) for b in range(bank.n_bundles)])
    assert np.abs(counts / n - expected).max() < 0.01


def test_learning_peaks_at_sweet_spot():
    sim = Simulator(make_bank(50, np.random.default_rng(2)))
    gains = sim.learning_gain(np.array([0.3, 0.7, 0.95]))
    assert gains[1] > gains[0] and gains[1] > gains[2]


def test_generation_is_reproducible_and_well_formed():
    a, questions, _, _ = generate(30, 200, seed=7)
    b, _, _, _ = generate(30, 200, seed=7)
    assert a.equals(b)
    q = a[a.content_type_id == 0]
    assert q.content_id.isin(questions.question_id).all()
    assert (a.groupby("user_id").timestamp.diff().dropna() >= 0).all()
    first = q.groupby("user_id").head(1)
    assert first.prior_question_elapsed_time.isna().all()
