import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_practice.common import device, save_json
from adaptive_practice.data.synthetic import generate
from adaptive_practice.evaluation import ope
from adaptive_practice.evaluation.live import EnginePolicy
from adaptive_practice.features import sequences
from adaptive_practice.features.spark_job import run as build_features
from adaptive_practice.policy.rerank import Objective, load_engine
from adaptive_practice.simulator import AppPolicy, Simulator, bank_from_tables, simulate_user

OBJECTIVES = {
    "engagement": Objective(learning=0.0, engagement=1.0, coverage=0.0, time=0.0),
    "learning": Objective(learning=1.0, engagement=0.0, coverage=0.0, time=0.0),
    "balanced": Objective(),
}


def expected_keep(sim, student, log, bundle):
    p = sim.p_correct(student, bundle)
    total = 0.0
    for outcome in itertools.product((0, 1), repeat=len(p)):
        o = np.array(outcome)
        weight = float(np.prod(np.where(o == 1, p, 1 - p)))
        recent = (log.recent + o.tolist())[-6:]
        total += weight * sim.continue_prob(student, log.session_position + 1, float(np.mean(recent)), float(p.mean()))
    return total


class GroundTruth:
    def __init__(self, engine, objectives, uniform_samples=32, seed=0):
        self.engine = engine
        self.objectives = objectives
        self.policy = EnginePolicy(engine)
        self.uniform_samples = uniform_samples
        self.rng = np.random.default_rng(seed)
        self.records = []

    def start_user(self, user_id, sim, student):
        self.user = user_id
        self.sim = sim
        self.container = 0
        self.policy.reset()

    def on_decision(self, log, student):
        d, seq, upto = self.policy.decision(log)
        keep = np.array([expected_keep(self.sim, student, log, b) for b in d.candidates])
        uniform = self.rng.integers(0, self.engine.n_bundles, self.uniform_samples)
        keep_uniform = float(np.mean([expected_keep(self.sim, student, log, b) for b in uniform]))
        eps = self.engine.epsilon
        values = {}
        for name, objective in self.objectives.items():
            dd = self.engine.reweight(d, seq, upto, objective)
            values[name] = (1 - eps) * float(dd.probs @ keep) + eps * keep_uniform
        self.pending = values

    def on_event(self, log, bundle, propensity, correct, elapsed, p, clock):
        self.policy.observe(log, bundle, propensity, correct, elapsed, p, clock)
        self.records.append({"user_id": self.user, "container": self.container, **self.pending})
        self.container += 1


def bank_for(features_dir, seed):
    return bank_from_tables(sequences.load_bundles(features_dir), sequences.load_questions(features_dir),
                            np.random.default_rng(seed))


def validate_ope(run_dir, features_dir, work_dir, n_users, bank_seed, seed, dev, k=50):
    engine = load_engine(run_dir, features_dir, dev, k=k)
    truth = GroundTruth(engine, OBJECTIVES, seed=seed)
    bank = bank_for(features_dir, bank_seed)
    raw, _, lectures, bank = generate(n_users, bank=bank, seed=seed, observers=[truth])
    work = Path(work_dir)
    raw_dir = work / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(raw_dir / "train.csv", index=False)
    bank.questions.to_csv(raw_dir / "questions.csv", index=False)
    lectures.to_csv(raw_dir / "lectures.csv", index=False)
    build_features(raw_dir, work / "features", memory="2g")
    seq = sequences.load_all(work / "features")
    cont = sequences.containers(seq)
    decisions = ope.eligible_decisions(cont)
    logged = ope.collect(engine, seq, cont, decisions, OBJECTIVES)
    gt = pd.DataFrame(truth.records)
    key = pd.DataFrame({"user_id": seq.user_ids[cont.user_index[decisions]],
                        "container": decisions - cont.offsets[cont.user_index[decisions]]})
    gt = key.merge(gt, on=["user_id", "container"], how="left")
    results = {"decisions": int(len(decisions))}
    for name in OBJECTIVES:
        r = ope.report(logged[name])
        r["ground_truth"] = float(gt[name].mean())
        results[name] = r
    return results


def rollout(sim, policy_factory, n_students, seed, max_bundles, max_sessions):
    rows = []
    for s in range(n_students):
        rng = np.random.default_rng(seed + s)
        student = sim.new_student(rng)
        start = student.skill.copy()
        policy = policy_factory()
        observe = getattr(policy, "observe", None)
        log, hits, questions = simulate_user(sim, policy, student, rng, max_bundles=max_bundles, max_sessions=max_sessions,
                                             on_event=observe)
        rows.append({
            "sessions": log.session_index,
            "returned": int(log.session_index > 1),
            "bundles": len(log.bundles),
            "questions": questions,
            "accuracy": hits / max(questions, 1),
            "skill_gain": float(np.mean(student.skill - start)),
        })
    df = pd.DataFrame(rows)
    summary = df.mean().to_dict()
    summary["skill_gain_per_100_questions"] = float(100 * df["skill_gain"].sum() / max(df["questions"].sum(), 1))
    summary["bundles_per_session"] = float(df["bundles"].sum() / max(df["sessions"].sum(), 1))
    return {k: float(v) for k, v in summary.items()}


def long_term(run_dir, features_dir, n_students, bank_seed, seed, dev, max_bundles=200, max_sessions=20, k=50):
    engine = load_engine(run_dir, features_dir, dev, k=k)
    sim = Simulator(bank_for(features_dir, bank_seed))
    policies = {"app": lambda: AppPolicy(sim.bank)}
    for name, objective in OBJECTIVES.items():
        policies[name] = lambda objective=objective: EnginePolicy(engine, objective)
    return {name: rollout(sim, factory, n_students, seed, max_bundles, max_sessions) for name, factory in policies.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/latest")
    parser.add_argument("--features", default="data/features")
    parser.add_argument("--work", default="data/ope_validation")
    parser.add_argument("--bank-seed", type=int, default=0)
    parser.add_argument("--users", type=int, default=200)
    parser.add_argument("--students", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--skip-ope", action="store_true")
    parser.add_argument("--skip-long-term", action="store_true")
    args = parser.parse_args()
    dev = device()
    out = Path(args.run)
    if not args.skip_ope:
        results = validate_ope(args.run, args.features, args.work, args.users, args.bank_seed, args.seed, dev)
        save_json(out / "ope_validation.json", results)
        print(results)
    if not args.skip_long_term:
        results = long_term(args.run, args.features, args.students, args.bank_seed, args.seed + 1, dev)
        save_json(out / "long_term.json", results)
        print(results)


if __name__ == "__main__":
    main()
