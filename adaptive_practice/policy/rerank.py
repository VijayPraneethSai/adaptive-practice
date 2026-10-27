from dataclasses import dataclass

import numpy as np
import torch

from adaptive_practice.features import sequences
from adaptive_practice.ranking import model as ranking
from adaptive_practice.ranking.model import score_candidates
from adaptive_practice.retrieval import two_tower
from adaptive_practice.retrieval.two_tower import encode, item_matrix


@dataclass
class Objective:
    learning: float = 1.0
    engagement: float = 1.0
    coverage: float = 0.3
    time: float = 0.1
    target: float = 0.7
    width: float = 0.2


def learning_value(p_correct, target, width):
    return np.exp(-(((np.asarray(p_correct) - target) / width) ** 2))


def novelty(tags, seen):
    return 1.0 / (1.0 + np.array([seen.get(int(t), 0) for t in tags], dtype=np.float64))


def utility(scores, tags, seen, objective):
    return (
        objective.learning * learning_value(scores["p_correct"], objective.target, objective.width)
        + objective.engagement * scores["p_continue"]
        + objective.coverage * novelty(tags, seen)
        - objective.time * np.log1p(scores["seconds"] / 60.0)
    )


def greedy_slate(scores, tags, seen, objective, size):
    seen = dict(seen)
    remaining = list(range(len(tags)))
    order = []
    while remaining and len(order) < size:
        sub = {k: np.asarray(v)[remaining] for k, v in scores.items()}
        u = utility(sub, np.asarray(tags)[remaining], seen, objective)
        pick = remaining[int(np.argmax(u))]
        order.append(pick)
        seen[int(tags[pick])] = seen.get(int(tags[pick]), 0) + 1
        remaining.remove(pick)
    return order


def softmax(x, temperature):
    z = (x - x.max()) / temperature
    e = np.exp(z)
    return e / e.sum()


@dataclass
class Decision:
    candidates: np.ndarray
    scores: dict
    utility: np.ndarray
    probs: np.ndarray


class Engine:
    def __init__(self, retriever, retriever_cfg, ranker, table, bundle_questions, objective=None, k=50,
                 temperature=0.05, epsilon=0.05, dev=None):
        self.retriever = retriever
        self.retriever_cfg = retriever_cfg
        self.ranker = ranker
        self.table = table
        self.bundle_questions = bundle_questions
        self.bundle_tag = np.array([table.tag[q[0]] if len(q) else 0 for q in bundle_questions])
        self.objective = objective or Objective()
        self.k = k
        self.temperature = temperature
        self.epsilon = epsilon
        self.dev = dev or torch.device("cpu")
        self.items = item_matrix(retriever)

    @property
    def n_bundles(self):
        return len(self.bundle_questions)

    def user_vectors(self, cont, decisions):
        return encode(self.retriever, cont, np.asarray(decisions), self.retriever_cfg.history, self.dev)

    def candidates(self, cont, decisions):
        users = self.user_vectors(cont, decisions)
        scores = users @ self.items.T
        k = min(self.k, scores.shape[1])
        top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        return np.take_along_axis(top, np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1), axis=1)

    def session_tags(self, seq, upto):
        user = np.searchsorted(seq.offsets, upto, side="right") - 1
        lo = seq.offsets[user]
        session = seq.session[upto] if upto < seq.offsets[user + 1] else seq.session[upto - 1]
        mask = seq.session[lo:upto] == session
        tags, counts = np.unique(seq.tag[lo:upto][mask], return_counts=True)
        return dict(zip(tags.tolist(), counts.tolist()))

    def score(self, seq, user, upto, candidates, lag, position, explained):
        return score_candidates(self.ranker, seq, self.table, user, upto, [self.bundle_questions[b] for b in candidates],
                                lag, position, explained, self.dev)

    def decide(self, seq, user, upto, candidates, lag, position, explained, objective=None):
        objective = objective or self.objective
        scores = self.score(seq, user, upto, candidates, lag, position, explained)
        u = utility(scores, self.bundle_tag[candidates], self.session_tags(seq, upto), objective)
        return Decision(np.asarray(candidates), scores, u, softmax(u, self.temperature))

    def reweight(self, decision, seq, upto, objective):
        u = utility(decision.scores, self.bundle_tag[decision.candidates], self.session_tags(seq, upto), objective)
        return Decision(decision.candidates, decision.scores, u, softmax(u, self.temperature))

    def action_prob(self, decision, action):
        hit = np.flatnonzero(decision.candidates == action)
        inside = decision.probs[hit[0]] if len(hit) else 0.0
        return (1.0 - self.epsilon) * inside + self.epsilon / self.n_bundles

    def sample(self, decision, rng):
        if rng.random() < self.epsilon:
            return int(rng.integers(self.n_bundles))
        return int(decision.candidates[rng.choice(len(decision.candidates), p=decision.probs)])


def load_engine(run_dir, features_dir, dev, **kwargs):
    bundles = sequences.load_bundles(features_dir)
    retriever, rcfg = two_tower.load_model(run_dir, len(bundles), bundles["part"].to_numpy(), dev)
    ranker, _ = ranking.load_model(run_dir, features_dir, dev)
    table = sequences.question_table(features_dir)
    return Engine(retriever, rcfg, ranker, table, sequences.bundle_questions(features_dir, len(bundles)), dev=dev, **kwargs)
