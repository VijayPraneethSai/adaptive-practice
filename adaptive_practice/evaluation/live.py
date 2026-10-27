import numpy as np

from adaptive_practice.features import sequences


class LiveHistory:
    def __init__(self, table, bundle_questions):
        self.table = table
        self.bundle_questions = bundle_questions
        self.rows = {k: [] for k in ("question", "bundle", "part", "container", "position", "lag", "explained", "correct",
                                     "elapsed", "log_elapsed", "tag", "session")}
        self.containers = 0
        self.last_time = None

    def __len__(self):
        return len(self.rows["question"])

    def add(self, bundle, correct, elapsed_ms, timestamp, session, position, explained=1):
        lag_ms = -1 if self.last_time is None else timestamp - self.last_time
        mean_elapsed = float(np.mean(elapsed_ms))
        for q, c in zip(self.bundle_questions[bundle], correct):
            self._append(q, bundle, self.containers, position, lag_ms, explained, int(c), mean_elapsed, session)
        self.containers += 1
        self.last_time = timestamp

    def _append(self, q, bundle, container, position, lag_ms, explained, correct, elapsed_ms, session):
        r = self.rows
        r["question"].append(int(q))
        r["bundle"].append(int(bundle))
        r["part"].append(int(self.table.part[q]))
        r["container"].append(container)
        r["position"].append(min(position, sequences.POSITION_BUCKETS - 1))
        r["lag"].append(int(sequences.lag_bucket(np.array([lag_ms]))[0]))
        r["explained"].append(int(explained))
        r["correct"].append(correct)
        r["elapsed"].append(int(sequences.elapsed_bucket(np.array([elapsed_ms]))[0]) if elapsed_ms is not None else 0)
        r["log_elapsed"].append(float(np.log1p(elapsed_ms / 1000.0)) if elapsed_ms is not None else 0.0)
        r["tag"].append(int(self.table.tag[q]))
        r["session"].append(int(session))

    def snapshot(self, now, session, position, explained=1):
        lag_ms = -1 if self.last_time is None else now - self.last_time
        n = len(self)
        arrays = {k: np.array(v, dtype=np.float64 if k == "log_elapsed" else np.int64) for k, v in self.rows.items()}
        pending = {
            "question": 1, "bundle": 0, "part": int(self.table.part[1]), "container": self.containers,
            "position": min(position, sequences.POSITION_BUCKETS - 1),
            "lag": int(sequences.lag_bucket(np.array([lag_ms]))[0]), "explained": int(explained), "correct": 0,
            "elapsed": 0, "log_elapsed": 0.0, "tag": int(self.table.tag[1]), "session": int(session),
        }
        for k, v in pending.items():
            arrays[k] = np.append(arrays[k], v).astype(arrays[k].dtype if n else (np.float64 if k == "log_elapsed" else np.int64))
        total = n + 1
        seq = sequences.Sequences(
            offsets=np.array([0, total]), user_ids=np.array([0]), question=arrays["question"], bundle=arrays["bundle"],
            part=arrays["part"], container=arrays["container"], position=arrays["position"], lag=arrays["lag"],
            explained=arrays["explained"], correct=arrays["correct"], elapsed=arrays["elapsed"],
            log_elapsed=arrays["log_elapsed"], cont=np.full(total, -1), propensity=np.full(total, np.nan),
            tag=arrays["tag"], session=arrays["session"],
            dense=np.zeros((total, sequences.DENSE_FEATURES), dtype=np.float32),
        )
        cont = sequences.containers(seq)
        return seq, cont, n, len(cont) - 1


class EnginePolicy:
    def __init__(self, engine, objective=None, greedy=False):
        self.engine = engine
        self.objective = objective
        self.greedy = greedy
        self.reset()

    def reset(self):
        self.history = LiveHistory(self.engine.table, self.engine.bundle_questions)

    def observe(self, log, bundle, propensity, correct, elapsed, p, clock):
        self.history.add(bundle, correct, elapsed, clock, log.session_index, log.session_position)

    def decision(self, log, objective=None):
        seq, cont, upto, idx = self.history.snapshot(log.now, log.session_index, log.session_position)
        candidates = self.engine.candidates(cont, [idx])[0]
        d = self.engine.decide(seq, 0, upto, candidates, cont.lag[idx], cont.position[idx], 1, objective or self.objective)
        return d, seq, upto

    def choose(self, log, rng):
        d, _, _ = self.decision(log)
        if self.greedy:
            bundle = int(d.candidates[int(np.argmax(d.utility))])
            return bundle, 1.0
        bundle = self.engine.sample(d, rng)
        return bundle, self.engine.action_prob(d, bundle)
