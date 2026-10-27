import numpy as np
import pandas as pd
import pytest

from adaptive_practice.data.schema import SESSION_GAP_MS
from adaptive_practice.data.synthetic import write
from adaptive_practice.features.spark_job import run


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    out = tmp_path_factory.mktemp("features")
    write(raw, n_users=60, n_questions=400, seed=3)
    run(raw, out, memory="2g")
    features = pd.read_parquet(out / "interactions").sort_values(["user_id", "timestamp", "row_id"]).reset_index(drop=True)
    train = pd.read_csv(raw / "train.csv")
    return train, features


def reference(train):
    t = train.sort_values(["user_id", "timestamp", "row_id"])
    q = t[t.content_type_id == 0].copy()
    lectures = t[t.content_type_id == 1]
    rows = []
    for user, g in q.groupby("user_id"):
        lec_ts = np.sort(lectures.loc[lectures.user_id == user, "timestamp"].to_numpy())
        stamps = g["timestamp"].drop_duplicates().to_numpy()
        session = np.cumsum(np.r_[1, np.diff(stamps) > SESSION_GAP_MS]) - 1
        for i, ts in enumerate(stamps):
            before = g[g.timestamp < ts]
            same_session = before[before.timestamp.isin(stamps[session == session[i]])]
            if i + 1 < len(stamps):
                cont = int(session[i + 1] == session[i])
                elapsed = g.loc[g.timestamp == stamps[i + 1], "prior_question_elapsed_time"].iloc[0]
            else:
                cont, elapsed = None, np.nan
            rows.append({
                "user_id": user,
                "timestamp": ts,
                "ref_prior_attempts": len(before),
                "ref_prior_accuracy": (before.answered_correctly.sum() + 1) / (len(before) + 2),
                "ref_session_index": session[i],
                "ref_session_prior_attempts": len(same_session),
                "ref_prior_lectures": int(np.searchsorted(lec_ts, ts)),
                "ref_label_continue": cont,
                "ref_label_elapsed": elapsed,
            })
    return pd.DataFrame(rows)


def test_row_count_matches_questions(built):
    train, features = built
    assert len(features) == int((train.content_type_id == 0).sum())


def test_history_features_are_leak_free(built):
    train, features = built
    merged = features.merge(reference(train), on=["user_id", "timestamp"])
    assert len(merged) == len(features)
    assert (merged.prior_attempts == merged.ref_prior_attempts).all()
    assert np.allclose(merged.prior_accuracy, merged.ref_prior_accuracy)
    assert (merged.session_index == merged.ref_session_index).all()
    assert (merged.session_prior_attempts == merged.ref_session_prior_attempts).all()
    assert (merged.prior_lectures == merged.ref_prior_lectures).all()


def test_labels_look_ahead_one_container(built):
    train, features = built
    merged = features.merge(reference(train), on=["user_id", "timestamp"])
    known = merged.ref_label_continue.notna()
    assert (merged.label_continue[known] == merged.ref_label_continue[known]).all()
    assert merged.label_continue[~known].isna().all()
    both = merged.ref_label_elapsed.notna()
    assert np.allclose(merged.label_elapsed[both], merged.ref_label_elapsed[both])


def test_users_stay_in_one_split(built):
    _, features = built
    assert (features.groupby("user_id").split.nunique() == 1).all()
    assert set(features.split) <= {"train", "valid", "test"}
