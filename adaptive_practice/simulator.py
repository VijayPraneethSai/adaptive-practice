from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from adaptive_practice.data.schema import N_PARTS, N_TAGS

PART_SHARE = np.array([0.08, 0.12, 0.18, 0.15, 0.27, 0.1, 0.1])
BUNDLED_PARTS = {3, 4, 6, 7}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Bank:
    questions: pd.DataFrame
    lectures: pd.DataFrame
    difficulty: np.ndarray
    bundle_ids: np.ndarray
    bundle_questions: list
    bundle_part: np.ndarray
    bundle_difficulty: np.ndarray

    @property
    def n_bundles(self):
        return len(self.bundle_ids)


def make_bank(n_questions, rng, n_lectures=60):
    rows = []
    difficulty = []
    qid = 0
    part_tags = {p: rng.choice(N_TAGS, size=12, replace=False) + 1 for p in range(1, N_PARTS + 1)}
    while qid < n_questions:
        part = int(rng.choice(N_PARTS, p=PART_SHARE)) + 1
        size = int(rng.integers(3, 6)) if part in BUNDLED_PARTS else 1
        size = min(size, n_questions - qid)
        base = rng.normal(0.0, 1.0) + 0.15 * (part - 4)
        bundle_id = qid
        for _ in range(size):
            main_tag = int(rng.choice(part_tags[part]))
            extra = rng.choice(N_TAGS, size=int(rng.integers(0, 3)), replace=False) + 1
            tags = " ".join(str(t) for t in [main_tag, *extra.tolist()])
            rows.append((qid, bundle_id, int(rng.integers(0, 4)), part, tags))
            difficulty.append(base + rng.normal(0.0, 0.35))
            qid += 1
    questions = pd.DataFrame(rows, columns=["question_id", "bundle_id", "correct_answer", "part", "tags"])
    lectures = pd.DataFrame({
        "lecture_id": np.arange(n_lectures) + 30000,
        "tag": rng.integers(1, N_TAGS + 1, size=n_lectures),
        "part": rng.integers(1, N_PARTS + 1, size=n_lectures),
        "type_of": rng.choice(["concept", "solving question", "intention"], size=n_lectures),
    })
    difficulty = np.array(difficulty)
    grouped = questions.groupby("bundle_id", sort=True)
    bundle_ids = np.array(list(grouped.groups.keys()))
    bundle_questions = [g["question_id"].to_numpy() for _, g in grouped]
    bundle_part = grouped["part"].first().to_numpy()
    bundle_difficulty = np.array([difficulty[q].mean() for q in bundle_questions])
    return Bank(questions, lectures, difficulty, bundle_ids, bundle_questions, bundle_part, bundle_difficulty)


@dataclass
class SimParams:
    learning_rate: float = 0.06
    sweet_spot: float = 0.7
    sweet_width: float = 0.2
    forgetting_per_day: float = 0.01
    continue_base: float = 2.6
    fatigue: float = 0.05
    frustration: float = 4.0
    boredom: float = 6.0
    return_base: float = 0.6
    return_balance: float = 2.5
    lecture_rate: float = 0.02


@dataclass
class Student:
    skill: np.ndarray
    baseline: np.ndarray
    patience: float
    speed: float


@dataclass
class UserLog:
    bundles: list = field(default_factory=list)
    parts: list = field(default_factory=list)
    correct: list = field(default_factory=list)
    timestamps: list = field(default_factory=list)
    session_position: int = 0
    session_index: int = 0
    attempts: np.ndarray = field(default_factory=lambda: np.zeros(N_PARTS))
    hits: np.ndarray = field(default_factory=lambda: np.zeros(N_PARTS))


class Simulator:
    def __init__(self, bank, params=None):
        self.bank = bank
        self.params = params or SimParams()

    def new_student(self, rng):
        overall = rng.normal(0.0, 1.0)
        skill = overall + rng.normal(0.0, 0.5, size=N_PARTS)
        return Student(skill=skill, baseline=skill.copy(), patience=rng.normal(0.0, 0.7), speed=rng.normal(0.0, 0.25))

    def p_correct(self, student, bundle):
        qids = self.bank.bundle_questions[bundle]
        part = self.bank.bundle_part[bundle] - 1
        return sigmoid(student.skill[part] - self.bank.difficulty[qids])

    def learning_gain(self, p):
        prm = self.params
        return prm.learning_rate * np.exp(-(((p - prm.sweet_spot) / prm.sweet_width) ** 2))

    def attempt(self, student, bundle, rng):
        p = self.p_correct(student, bundle)
        correct = (rng.random(len(p)) < p).astype(np.int8)
        qids = self.bank.bundle_questions[bundle]
        mean_log = 9.6 + 0.25 * self.bank.difficulty[qids] - 0.15 * correct + student.speed
        elapsed = np.exp(rng.normal(mean_log, 0.45))
        part = self.bank.bundle_part[bundle] - 1
        student.skill[part] += float(np.sum(self.learning_gain(p) * np.where(correct == 1, 1.0, 0.6)))
        return correct, elapsed, p

    def continue_prob(self, student, position, recent_accuracy, p_mean):
        prm = self.params
        logit = (
            prm.continue_base
            + student.patience
            - prm.fatigue * position
            - prm.frustration * max(0.0, 0.5 - recent_accuracy)
            - prm.boredom * max(0.0, p_mean - 0.85)
        )
        return float(sigmoid(logit))

    def return_gap_ms(self, student, session_accuracy, rng):
        prm = self.params
        balance = 1.0 - abs(session_accuracy - prm.sweet_spot) / 0.5
        p_return = sigmoid(prm.return_base + prm.return_balance * (balance - 0.5) + 0.5 * student.patience)
        if rng.random() >= p_return:
            return None
        days = rng.exponential(1.5) + 0.05
        decay = prm.forgetting_per_day * days
        student.skill -= decay * (student.skill - student.baseline)
        return days * 86_400_000


class AppPolicy:
    def __init__(self, bank, epsilon=0.1, part_temperature=3.0, bucket_temperature=1.5, n_buckets=5):
        self.bank = bank
        self.epsilon = epsilon
        self.part_temperature = part_temperature
        self.bucket_temperature = bucket_temperature
        self.n_buckets = n_buckets
        self.buckets = []
        for part in range(1, N_PARTS + 1):
            idx = np.flatnonzero(bank.bundle_part == part)
            order = idx[np.argsort(bank.bundle_difficulty[idx])]
            self.buckets.append([b for b in np.array_split(order, n_buckets) if len(b)])
        self.lookup = {}
        for p, buckets in enumerate(self.buckets):
            for k, members in enumerate(buckets):
                for b in members:
                    self.lookup[int(b)] = (p, k)

    def _distributions(self, log):
        accuracy = (log.hits + 1.0) / (log.attempts + 2.0)
        part_logits = -self.part_temperature * accuracy
        part_probs = np.exp(part_logits - part_logits.max())
        part_probs /= part_probs.sum()
        bucket_probs = []
        for p, buckets in enumerate(self.buckets):
            n = len(buckets)
            target = accuracy[p] * (n - 1)
            logits = -self.bucket_temperature * np.abs(np.arange(n) - target)
            probs = np.exp(logits - logits.max())
            bucket_probs.append(probs / probs.sum())
        return part_probs, bucket_probs

    def propensity(self, log, bundle):
        part_probs, bucket_probs = self._distributions(log)
        p, k = self.lookup[int(bundle)]
        structured = part_probs[p] * bucket_probs[p][k] / len(self.buckets[p][k])
        return (1.0 - self.epsilon) * structured + self.epsilon / self.bank.n_bundles

    def choose(self, log, rng):
        if rng.random() < self.epsilon:
            bundle = int(rng.integers(self.bank.n_bundles))
        else:
            part_probs, bucket_probs = self._distributions(log)
            p = int(rng.choice(N_PARTS, p=part_probs))
            k = int(rng.choice(len(bucket_probs[p]), p=bucket_probs[p]))
            bundle = int(rng.choice(self.buckets[p][k]))
        return bundle, self.propensity(log, bundle)


def simulate_user(sim, policy, student, rng, max_bundles=300, max_sessions=30, on_event=None):
    log = UserLog()
    clock = 0.0
    total_correct = 0
    total_questions = 0
    while log.session_index < max_sessions and len(log.bundles) < max_bundles:
        log.session_position = 0
        session_hits = 0
        session_questions = 0
        recent = []
        while len(log.bundles) < max_bundles:
            bundle, propensity = policy.choose(log, rng)
            correct, elapsed, p = sim.attempt(student, bundle, rng)
            part = sim.bank.bundle_part[bundle] - 1
            if on_event is not None:
                on_event(log, bundle, propensity, correct, elapsed, p, clock)
            log.bundles.append(bundle)
            log.parts.append(part)
            log.correct.append(float(correct.mean()))
            log.timestamps.append(clock)
            log.attempts[part] += len(correct)
            log.hits[part] += correct.sum()
            session_hits += int(correct.sum())
            session_questions += len(correct)
            recent = (recent + correct.tolist())[-6:]
            clock += float(elapsed.sum()) + rng.uniform(3_000, 20_000)
            log.session_position += 1
            keep = rng.random() < sim.continue_prob(student, log.session_position, float(np.mean(recent)), float(p.mean()))
            if not keep:
                break
        total_correct += session_hits
        total_questions += session_questions
        gap = sim.return_gap_ms(student, session_hits / max(session_questions, 1), rng)
        log.session_index += 1
        if gap is None:
            break
        clock += gap
    return log, total_correct, total_questions
