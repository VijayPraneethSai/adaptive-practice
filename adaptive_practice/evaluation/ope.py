from dataclasses import dataclass

import numpy as np

from adaptive_practice.policy.rerank import learning_value, softmax


@dataclass
class Logged:
    reward: np.ndarray
    target_prob: np.ndarray
    behavior_true: np.ndarray
    behavior_model: np.ndarray
    q_logged: np.ndarray
    q_target: np.ndarray
    learning_target: np.ndarray
    learning_logged: np.ndarray


def ips(r, w):
    return float(np.mean(w * r))


def snips(r, w):
    return float(np.sum(w * r) / np.sum(w))


def doubly_robust(r, w, q_logged, q_target):
    return float(np.mean(q_target + w * (r - q_logged)))


def effective_sample_size(w):
    return float(np.sum(w) ** 2 / np.sum(w ** 2))


def estimates(r, w, q_logged, q_target, clip=None):
    if clip is not None:
        w = np.minimum(w, clip)
    return {
        "ips": ips(r, w),
        "snips": snips(r, w),
        "dr": doubly_robust(r, w, q_logged, q_target),
        "dm": float(np.mean(q_target)),
        "ess": effective_sample_size(w),
        "max_weight": float(np.max(w)),
    }


def bootstrap(r, w, q_logged, q_target, clip=None, n=200, seed=0):
    rng = np.random.default_rng(seed)
    draws = {"snips": [], "dr": []}
    for _ in range(n):
        i = rng.integers(0, len(r), len(r))
        e = estimates(r[i], w[i], q_logged[i], q_target[i], clip)
        draws["snips"].append(e["snips"])
        draws["dr"].append(e["dr"])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()}


def behavior_model_probs(engine, cont, decisions):
    users = engine.user_vectors(cont, decisions)
    scale = float(engine.retriever.log_scale.exp().item())
    logits = scale * users @ engine.items.T
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs[np.arange(len(decisions)), cont.bundle[decisions]]


def eligible_decisions(cont, max_decisions=None, seed=0):
    idx = np.arange(len(cont))
    has_history = idx - cont.offsets[cont.user_index] >= 1
    known = cont.cont >= 0
    idx = idx[has_history & known]
    if max_decisions and len(idx) > max_decisions:
        idx = np.sort(np.random.default_rng(seed).choice(idx, size=max_decisions, replace=False))
    return idx


def collect(engine, seq, cont, decisions, objectives):
    candidates = engine.candidates(cont, decisions)
    behavior_model = behavior_model_probs(engine, cont, decisions)
    out = {name: {"target_prob": [], "q_target": [], "learning_target": []} for name in objectives}
    q_logged, learning_logged = [], []
    for row, i in enumerate(decisions):
        logged = cont.bundle[i]
        cand = candidates[row]
        with_logged = cand if logged in cand else np.r_[cand, logged]
        upto = cont.first_row[i]
        base = engine.decide(seq, cont.user_index[i], upto, with_logged, cont.lag[i], cont.position[i], seq.explained[upto])
        j = int(np.flatnonzero(with_logged == logged)[0])
        q_logged.append(base.scores["p_continue"][j])
        learning_logged.append(learning_value(base.scores["p_correct"][j], engine.objective.target, engine.objective.width))
        keep = np.isin(with_logged, cand)
        for name, objective in objectives.items():
            d = engine.reweight(base, seq, upto, objective)
            probs = softmax(d.utility[keep], engine.temperature)
            inside = probs[int(np.flatnonzero(cand == logged)[0])] if logged in cand else 0.0
            eps = engine.epsilon
            out[name]["target_prob"].append((1 - eps) * inside + eps / engine.n_bundles)
            q_cand = d.scores["p_continue"][keep]
            learn_cand = learning_value(d.scores["p_correct"][keep], objective.target, objective.width)
            out[name]["q_target"].append((1 - eps) * float(probs @ q_cand) + eps * float(q_cand.mean()))
            out[name]["learning_target"].append((1 - eps) * float(probs @ learn_cand) + eps * float(learn_cand.mean()))
    reward = cont.cont[decisions].astype(np.float64)
    behavior_true = cont.propensity[decisions]
    return {
        name: Logged(
            reward=reward,
            target_prob=np.array(v["target_prob"]),
            behavior_true=behavior_true,
            behavior_model=behavior_model,
            q_logged=np.array(q_logged),
            q_target=np.array(v["q_target"]),
            learning_target=np.array(v["learning_target"]),
            learning_logged=np.array(learning_logged),
        )
        for name, v in out.items()
    }


def report(logged, clip=50.0):
    result = {"logging_policy_value": float(np.mean(logged.reward)),
              "logging_learning_value": float(np.mean(logged.learning_logged)),
              "target_learning_value": float(np.mean(logged.learning_target))}
    for label, behavior in (("model_propensity", logged.behavior_model), ("true_propensity", logged.behavior_true)):
        if np.isnan(behavior).any():
            continue
        w = logged.target_prob / behavior
        result[label] = estimates(logged.reward, w, logged.q_logged, logged.q_target, clip)
        result[label]["ci95"] = bootstrap(logged.reward, w, logged.q_logged, logged.q_target, clip)
    return result
