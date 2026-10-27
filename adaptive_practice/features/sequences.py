from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "user_id", "timestamp", "row_id", "content_id", "bundle_idx", "part", "container_seq", "session_position",
    "lag_ms", "prior_explained", "tag", "session_index", "q_accuracy", "q_attempts", "q_mean_elapsed",
    "prior_attempts", "prior_accuracy", "prior_part_attempts", "prior_part_accuracy", "prior_tag_attempts",
    "prior_tag_accuracy", "session_prior_attempts", "session_prior_accuracy", "label_correct", "label_elapsed", "label_continue", "propensity",
]

LAG_BUCKETS = 32
ELAPSED_BUCKETS = 64
POSITION_BUCKETS = 64
DENSE_FEATURES = 11


def dense_features(q_accuracy, q_attempts, q_mean_elapsed, prior, part, tag, session):
    cols = [np.asarray(q_accuracy, dtype=np.float64), np.log1p(q_attempts), np.log1p(np.asarray(q_mean_elapsed) / 1000.0)]
    for attempts, accuracy in (prior, part, tag, session):
        cols += [np.asarray(accuracy, dtype=np.float64), np.log1p(attempts)]
    return np.stack(cols, axis=-1).astype(np.float32)


def lag_bucket(lag_ms):
    lag = np.asarray(lag_ms, dtype=np.float64)
    seconds = np.maximum(lag, 0) / 1000.0
    bucket = np.clip(np.floor(np.log2(seconds + 1.0)).astype(np.int64) + 2, 2, LAG_BUCKETS - 1)
    bucket[lag == 0] = 1
    bucket[lag < 0] = 0
    return bucket


def elapsed_bucket(elapsed_ms):
    e = np.asarray(elapsed_ms, dtype=np.float64)
    seconds = np.nan_to_num(e, nan=-1000.0) / 1000.0
    bucket = np.clip(np.floor(np.sqrt(np.maximum(seconds, 0)) * 4).astype(np.int64) + 1, 1, ELAPSED_BUCKETS - 1)
    bucket[seconds < 0] = 0
    return bucket


@dataclass
class Sequences:
    offsets: np.ndarray
    user_ids: np.ndarray
    question: np.ndarray
    bundle: np.ndarray
    part: np.ndarray
    container: np.ndarray
    position: np.ndarray
    lag: np.ndarray
    explained: np.ndarray
    correct: np.ndarray
    elapsed: np.ndarray
    log_elapsed: np.ndarray
    cont: np.ndarray
    propensity: np.ndarray
    tag: np.ndarray = None
    session: np.ndarray = None
    dense: np.ndarray = None

    @property
    def n_users(self):
        return len(self.user_ids)

    def user_slice(self, i):
        return slice(self.offsets[i], self.offsets[i + 1])


def from_frame(df):
    df = df.sort_values(["user_id", "timestamp", "row_id"], kind="stable")
    users, starts = np.unique(df["user_id"].to_numpy(), return_index=True)
    offsets = np.r_[starts, len(df)].astype(np.int64)
    elapsed = df["label_elapsed"].to_numpy(dtype=np.float64)
    cont = df["label_continue"].to_numpy(dtype=np.float64)
    propensity = df["propensity"].to_numpy(dtype=np.float64) if "propensity" in df else np.full(len(df), np.nan)
    return Sequences(
        offsets=offsets,
        user_ids=users,
        question=df["content_id"].to_numpy(dtype=np.int64) + 1,
        bundle=df["bundle_idx"].to_numpy(dtype=np.int64),
        part=df["part"].to_numpy(dtype=np.int64),
        container=df["container_seq"].to_numpy(dtype=np.int64),
        position=np.minimum(df["session_position"].to_numpy(dtype=np.int64), POSITION_BUCKETS - 1),
        lag=lag_bucket(df["lag_ms"].to_numpy()),
        explained=df["prior_explained"].to_numpy(dtype=np.int64),
        correct=df["label_correct"].to_numpy(dtype=np.int64),
        elapsed=elapsed_bucket(elapsed),
        log_elapsed=np.log1p(np.nan_to_num(elapsed, nan=0.0) / 1000.0),
        cont=np.nan_to_num(cont, nan=-1.0).astype(np.int64),
        propensity=propensity,
        tag=df["tag"].to_numpy(dtype=np.int64) if "tag" in df else None,
        session=df["session_index"].to_numpy(dtype=np.int64) if "session_index" in df else None,
        dense=dense_features(
            df["q_accuracy"], df["q_attempts"], df["q_mean_elapsed"],
            (df["prior_attempts"], df["prior_accuracy"]),
            (df["prior_part_attempts"], df["prior_part_accuracy"]),
            (df["prior_tag_attempts"], df["prior_tag_accuracy"]),
            (df["session_prior_attempts"], df["session_prior_accuracy"]),
        ) if "q_accuracy" in df else None,
    )


def load(features_dir, split):
    path = Path(features_dir) / "interactions" / f"split={split}"
    df = pd.read_parquet(path, columns=COLUMNS)
    return from_frame(df)


def load_all(features_dir):
    path = Path(features_dir) / "interactions"
    return from_frame(pd.read_parquet(path, columns=COLUMNS))


def load_bundles(features_dir):
    return pd.read_parquet(Path(features_dir) / "bundles").sort_values("bundle_idx").reset_index(drop=True)


def load_questions(features_dir):
    return pd.read_parquet(Path(features_dir) / "questions").sort_values("content_id").reset_index(drop=True)


@dataclass
class QuestionTable:
    part: np.ndarray
    tag: np.ndarray
    accuracy: np.ndarray
    attempts: np.ndarray
    mean_elapsed: np.ndarray


def question_table(features_dir):
    q = load_questions(features_dir)
    size = int(q["content_id"].max()) + 2
    idx = q["content_id"].to_numpy() + 1
    table = QuestionTable(*(np.zeros(size) for _ in range(5)))
    table.part[idx] = q["part"].to_numpy()
    table.tag[idx] = q["tag"].to_numpy()
    table.accuracy[idx] = q["q_accuracy"].to_numpy()
    table.attempts[idx] = q["q_attempts"].to_numpy()
    table.mean_elapsed[idx] = q["q_mean_elapsed"].to_numpy()
    table.part = table.part.astype(np.int64)
    table.tag = table.tag.astype(np.int64)
    return table


def online_dense(seq, table, user, upto, questions):
    lo = seq.offsets[user]
    past_correct = seq.correct[lo:upto]
    past_part = seq.part[lo:upto]
    past_tag = seq.tag[lo:upto]
    session = seq.session[upto] if upto < seq.offsets[user + 1] else None
    in_session = (seq.session[lo:upto] == session) if session is not None else np.zeros(upto - lo, dtype=bool)
    q = np.asarray(questions)
    part = table.part[q]
    tag = table.tag[q]
    part_n = np.array([np.sum(past_part == p) for p in part])
    part_h = np.array([np.sum(past_correct[past_part == p]) for p in part])
    tag_n = np.array([np.sum(past_tag == t) for t in tag])
    tag_h = np.array([np.sum(past_correct[past_tag == t]) for t in tag])
    n, h = len(past_correct), past_correct.sum()
    s_n, s_h = in_session.sum(), past_correct[in_session].sum()
    ones = np.ones(len(q))
    acc = lambda hits, attempts: (hits + 1.0) / (attempts + 2.0)
    return dense_features(
        table.accuracy[q], table.attempts[q], table.mean_elapsed[q],
        (n * ones, acc(h, n) * ones),
        (part_n, acc(part_h, part_n)),
        (tag_n, acc(tag_h, tag_n)),
        (s_n * ones, acc(s_h, s_n) * ones),
    )


def bundle_questions(features_dir, n_bundles):
    questions = load_questions(features_dir)
    members = [[] for _ in range(n_bundles)]
    for q, b in zip(questions["content_id"].to_numpy(), questions["bundle_idx"].to_numpy()):
        members[int(b)].append(int(q) + 1)
    return [np.array(sorted(m), dtype=np.int64) for m in members]


@dataclass
class Containers:
    offsets: np.ndarray
    user_index: np.ndarray
    bundle: np.ndarray
    part: np.ndarray
    accuracy: np.ndarray
    log_elapsed: np.ndarray
    lag: np.ndarray
    position: np.ndarray
    cont: np.ndarray
    propensity: np.ndarray
    first_row: np.ndarray

    def __len__(self):
        return len(self.bundle)

    def user_start(self, idx):
        return self.offsets[self.user_index[idx]]


def containers(seq):
    user_of_row = np.repeat(np.arange(seq.n_users), np.diff(seq.offsets))
    key = user_of_row * (seq.container.max() + 1) + seq.container
    boundary = np.r_[True, key[1:] != key[:-1]]
    first = np.flatnonzero(boundary)
    group = np.cumsum(boundary) - 1
    size = np.bincount(group)
    accuracy = np.bincount(group, weights=seq.correct) / size
    user_index = user_of_row[first]
    counts = np.bincount(user_index, minlength=seq.n_users)
    return Containers(
        offsets=np.r_[0, np.cumsum(counts)].astype(np.int64),
        user_index=user_index,
        bundle=seq.bundle[first],
        part=seq.part[first],
        accuracy=accuracy,
        log_elapsed=seq.log_elapsed[first],
        lag=seq.lag[first],
        position=seq.position[first],
        cont=seq.cont[first],
        propensity=seq.propensity[first],
        first_row=first,
    )
