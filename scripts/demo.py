import argparse

import numpy as np

from adaptive_practice.common import device
from adaptive_practice.evaluation.live import EnginePolicy
from adaptive_practice.policy.rerank import Objective, load_engine


def ask(bank_questions, bank, bundle, rng):
    print(f"\n  part {bank.bundle_part[bundle]}  ({len(bank.bundle_questions[bundle])} question(s))")
    answers = []
    for q in bank.bundle_questions[bundle]:
        raw = input("  your answer (0-3, enter=skip/random): ").strip()
        answers.append(int(raw) if raw else int(rng.integers(0, 4)))
    correct = [int(a == bank.questions.loc[bank.questions.question_id == q, "correct_answer"].iloc[0])
               for a, q in zip(answers, bank.bundle_questions[bundle])]
    for q, c in zip(bank.bundle_questions[bundle], correct):
        print(f"    question {q}: {'correct' if c else 'incorrect'}")
    return np.array(correct), np.full(len(correct), 15000.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/syn")
    parser.add_argument("--features", default="data/features")
    parser.add_argument("--objective", choices=["learning", "engagement", "balanced"], default="balanced")
    parser.add_argument("--rounds", type=int, default=8)
    args = parser.parse_args()

    from adaptive_practice.features import sequences
    from adaptive_practice.simulator import bank_from_tables

    dev = device()
    engine = load_engine(args.run, args.features, dev, k=50)
    bank = bank_from_tables(sequences.load_bundles(args.features), sequences.load_questions(args.features),
                            np.random.default_rng(1))
    objectives = {
        "learning": Objective(learning=1, engagement=0, coverage=0, time=0),
        "engagement": Objective(learning=0, engagement=1, coverage=0, time=0),
        "balanced": Objective(),
    }
    policy = EnginePolicy(engine, objectives[args.objective])
    rng = np.random.default_rng(0)

    class Log:
        session_index = 0
        session_position = 0
        now = 0.0

    log = Log()
    print(f"adaptive practice demo — objective: {args.objective}")
    for i in range(args.rounds):
        d, _, _ = policy.decision(log)
        import time
        t0 = time.perf_counter()
        bundle = int(d.candidates[int(np.argmax(d.utility))])
        latency_ms = (time.perf_counter() - t0) * 1000
        print(f"\nround {i + 1}  (decision latency {latency_ms:.1f} ms,"
              f" predicted p(correct)={d.scores['p_correct'][np.argmax(d.utility)]:.2f},"
              f" p(continue)={d.scores['p_continue'][np.argmax(d.utility)]:.2f})")
        correct, elapsed = ask(bank.questions, bank, bundle, rng)
        policy.observe(log, bundle, 0.0, correct, elapsed, correct.astype(float), log.now)
        log.now += float(elapsed.sum()) + 5000
        log.session_position += 1
    print("\nsession accuracy:", np.mean(policy.history.rows["correct"]))


if __name__ == "__main__":
    main()
