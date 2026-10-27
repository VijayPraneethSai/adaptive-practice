import numpy as np
import pytest
import torch

from adaptive_practice.features import sequences
from adaptive_practice.features.spark_job import run
from adaptive_practice.data.synthetic import write
from adaptive_practice.ranking.model import (
    Ranker, RankerConfig, attach_question_parts, build_batch, predict, question_parts, score_candidates, to_device, train,
)


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    out = tmp_path_factory.mktemp("features")
    write(raw, n_users=50, n_questions=300, seed=5)
    run(raw, out, memory="2g")
    torch.manual_seed(0)
    cfg = RankerConfig(dim=32, heads=2, window=40, epochs=1, batch_size=32, windows_per_epoch=200)
    parts = question_parts(out)
    model = attach_question_parts(Ranker(len(parts) - 1, cfg), parts)
    seq = sequences.load(out, "train")
    train(model, seq, cfg, torch.device("cpu"), log=lambda _: None)
    model.eval()
    return model, cfg, seq, sequences.question_table(out)


def test_predictions_ignore_answers_from_the_same_and_later_containers(fitted):
    model, cfg, seq, _ = fitted
    user = int(np.argmax(np.diff(seq.offsets)))
    lo = seq.offsets[user]
    b = build_batch(seq, np.array([lo]), np.array([lo]), cfg.window)
    with torch.no_grad():
        base = [o.numpy() for o in model(to_device(b, torch.device("cpu")))]
    t = 20
    same = b["container"][0] >= b["container"][0, t]
    changed = {k: v.copy() for k, v in b.items()}
    changed["correct"][0, same] = 1 - changed["correct"][0, same]
    changed["elapsed"][0, same] = (changed["elapsed"][0, same] + 7) % 60
    with torch.no_grad():
        after = [o.numpy() for o in model(to_device(changed, torch.device("cpu")))]
    for a, c in zip(base, after):
        assert np.allclose(a[0, : t + 1], c[0, : t + 1], atol=1e-5)
        assert not np.allclose(a[0, t + 1 :], c[0, t + 1 :])


def test_candidate_scoring_matches_batch_prediction(fitted):
    model, cfg, seq, table = fitted
    p_correct, elapsed, p_continue = predict(model, seq, cfg, torch.device("cpu"))
    cont = sequences.containers(seq)
    checked = 0
    for i in range(len(cont)):
        u = cont.user_index[i]
        r = cont.first_row[i]
        end = cont.first_row[i + 1] if i + 1 < len(cont) and cont.user_index[i + 1] == u else seq.offsets[u + 1]
        if end - seq.offsets[u] > cfg.window:
            continue
        questions = seq.question[r:end]
        scores = score_candidates(model, seq, table, u, r, [questions], seq.lag[r], seq.position[r], seq.explained[r],
                                  torch.device("cpu"))
        assert np.isclose(scores["p_correct"][0], p_correct[r:end].mean(), atol=1e-5)
        assert np.isclose(scores["p_continue"][0], p_continue[r:end].mean(), atol=1e-5)
        checked += 1
        if checked >= 25:
            break
    assert checked >= 10
