import numpy as np

from adaptive_practice.features.sequences import elapsed_bucket, lag_bucket


def test_lag_buckets_separate_first_same_and_long_gaps():
    b = lag_bucket(np.array([-1, 0, 500, 60_000, 86_400_000 * 30]))
    assert b[0] == 0 and b[1] == 1
    assert b[2] < b[3] < b[4]
    assert b.max() < 32


def test_elapsed_buckets_handle_missing_values():
    b = elapsed_bucket(np.array([np.nan, 0.0, 15_000.0, 300_000.0, 10_000_000.0]))
    assert b[0] == 0
    assert b[1] < b[2] < b[3] <= b[4] < 64
