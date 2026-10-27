import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_practice.data.schema import TRAIN_DTYPES
from adaptive_practice.simulator import AppPolicy, Simulator, make_bank, simulate_user


def generate(n_users, n_questions=2000, seed=0, max_bundles=300, bank=None, observers=(), lecture_rate=None):
    rng = np.random.default_rng(seed)
    bank = bank if bank is not None else make_bank(n_questions, rng)
    sim = Simulator(bank)
    if lecture_rate is not None:
        sim.params.lecture_rate = lecture_rate
    policy = AppPolicy(bank)
    rows = []
    lecture_ids = bank.lectures["lecture_id"].to_numpy()
    correct_answers = bank.questions["correct_answer"].to_numpy()

    for user_id in range(n_users):
        student = sim.new_student(rng)
        for obs in observers:
            obs.start_user(user_id, sim, student)
        state = {"container": 0, "prev_elapsed": np.nan, "prev_explained": None}

        def on_event(log, bundle, propensity, correct, elapsed, p, clock):
            for obs in observers:
                obs.on_event(log, bundle, propensity, correct, elapsed, p, clock)
            ts = int(clock)
            if rng.random() < sim.params.lecture_rate:
                rows.append((ts, user_id, int(rng.choice(lecture_ids)), 1, state["container"], -1, -1, np.nan, None, np.nan))
                state["container"] += 1
                ts += 1
            explained = state["prev_explained"]
            for q, c in zip(bank.bundle_questions[bundle], correct):
                answer = correct_answers[q] if c else (correct_answers[q] + int(rng.integers(1, 4))) % 4
                rows.append((ts, user_id, int(q), 0, state["container"], int(answer), int(c), state["prev_elapsed"], explained, propensity))
            state["container"] += 1
            state["prev_elapsed"] = float(np.mean(elapsed))
            state["prev_explained"] = bool(rng.random() < 0.9)

        def on_decision(log, student):
            for obs in observers:
                obs.on_decision(log, student)

        simulate_user(sim, policy, student, rng, max_bundles=max_bundles, on_event=on_event, on_decision=on_decision)

    train = pd.DataFrame(rows, columns=[
        "timestamp", "user_id", "content_id", "content_type_id", "task_container_id", "user_answer",
        "answered_correctly", "prior_question_elapsed_time", "prior_question_had_explanation", "propensity",
    ])
    train.insert(0, "row_id", np.arange(len(train)))
    train = train.astype({**TRAIN_DTYPES, "propensity": "float64"})
    return train, bank.questions, bank.lectures, bank


def write(out_dir, n_users, n_questions, seed):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train, questions, lectures, _ = generate(n_users, n_questions, seed)
    train.to_csv(out / "train.csv", index=False)
    questions.to_csv(out / "questions.csv", index=False)
    lectures.to_csv(out / "lectures.csv", index=False)
    return len(train)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/synthetic")
    parser.add_argument("--users", type=int, default=2000)
    parser.add_argument("--questions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rows = write(args.out, args.users, args.questions, args.seed)
    print(f"wrote {rows} rows to {args.out}")


if __name__ == "__main__":
    main()
