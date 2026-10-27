import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

from adaptive_practice.common import save_json

FEATURES = [
    "part", "lag_ms", "session_position", "prior_lectures", "prior_attempts", "prior_accuracy", "prior_part_attempts",
    "prior_part_accuracy", "prior_tag_attempts", "prior_tag_accuracy", "session_prior_attempts", "session_prior_accuracy",
    "prior_elapsed", "prior_explained", "q_accuracy", "q_mean_elapsed", "q_attempts",
]


def load(features_dir, split, max_rows=None, seed=0):
    df = pd.read_parquet(Path(features_dir) / "interactions" / f"split={split}",
                         columns=["user_id", "timestamp", "row_id", *FEATURES, "label_correct", "label_elapsed", "label_continue"])
    if max_rows and len(df) > max_rows:
        users = df["user_id"].drop_duplicates().sample(frac=1.0, random_state=seed)
        sizes = df.groupby("user_id").size().reindex(users)
        keep = users[sizes.cumsum().to_numpy() <= max_rows]
        df = df[df["user_id"].isin(keep)]
    return df.sort_values(["user_id", "timestamp", "row_id"]).reset_index(drop=True)


def first_of_container(df):
    return df.drop_duplicates(["user_id", "timestamp"])


def fit(train, seed=0):
    models = {}
    params = dict(max_iter=300, learning_rate=0.1, max_leaf_nodes=63, early_stopping=True, random_state=seed)
    models["correct"] = HistGradientBoostingClassifier(**params).fit(train[FEATURES], train["label_correct"])
    cont = first_of_container(train).dropna(subset=["label_continue"])
    models["continue"] = HistGradientBoostingClassifier(**params).fit(cont[FEATURES], cont["label_continue"].astype(int))
    timed = first_of_container(train).dropna(subset=["label_elapsed"])
    models["elapsed"] = HistGradientBoostingRegressor(**params).fit(timed[FEATURES], np.log1p(timed["label_elapsed"] / 1000.0))
    return models


def evaluate(models, df):
    cont = first_of_container(df).dropna(subset=["label_continue"])
    timed = df.dropna(subset=["label_elapsed"])
    seconds = np.expm1(models["elapsed"].predict(timed[FEATURES]))
    return {
        "rows": int(len(df)),
        "auc_correct": float(roc_auc_score(df["label_correct"], models["correct"].predict_proba(df[FEATURES])[:, 1])),
        "auc_continue": float(roc_auc_score(cont["label_continue"].astype(int), models["continue"].predict_proba(cont[FEATURES])[:, 1])),
        "mae_elapsed_seconds": float(np.mean(np.abs(timed["label_elapsed"].to_numpy() / 1000.0 - seconds))),
        "auc_correct_question_only": float(roc_auc_score(df["label_correct"], df["q_accuracy"])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/features")
    parser.add_argument("--out", default="runs/latest")
    parser.add_argument("--max-train-rows", type=int, default=5_000_000)
    parser.add_argument("--eval-split", default="valid")
    args = parser.parse_args()
    models = fit(load(args.features, "train", args.max_train_rows))
    results = evaluate(models, load(args.features, args.eval_split))
    save_json(Path(args.out) / f"baseline_{args.eval_split}.json", results)
    print(results)


if __name__ == "__main__":
    main()
