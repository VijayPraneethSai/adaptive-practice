import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from adaptive_practice.common import device, load_json, save_json, seed_everything
from adaptive_practice.data.schema import N_PARTS
from adaptive_practice.features import sequences

ACCURACY_BUCKETS = 6


@dataclass
class TwoTowerConfig:
    dim: int = 64
    history: int = 50
    batch_size: int = 1024
    epochs: int = 5
    lr: float = 2e-3
    weight_decay: float = 1e-5
    examples_per_epoch: int = 2_000_000
    seed: int = 0


class TwoTower(nn.Module):
    def __init__(self, n_bundles, bundle_part, dim=64):
        super().__init__()
        self.register_buffer("bundle_part", torch.as_tensor(bundle_part, dtype=torch.long))
        self.item_emb = nn.Embedding(n_bundles + 1, dim, padding_idx=0)
        self.part_emb = nn.Embedding(N_PARTS + 1, dim, padding_idx=0)
        self.accuracy_emb = nn.Embedding(ACCURACY_BUCKETS, dim, padding_idx=0)
        self.lag_emb = nn.Embedding(sequences.LAG_BUCKETS, dim)
        self.position_emb = nn.Embedding(sequences.POSITION_BUCKETS, dim)
        self.item_mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.user_mlp = nn.Sequential(nn.Linear(dim * 3, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.log_scale = nn.Parameter(torch.tensor(np.log(20.0), dtype=torch.float32))

    def item_vectors(self):
        ids = torch.arange(1, self.item_emb.num_embeddings, device=self.item_emb.weight.device)
        x = self.item_emb(ids) + self.part_emb(self.bundle_part)
        return F.normalize(self.item_mlp(x), dim=-1)

    def user_vectors(self, hist_bundle, hist_part, hist_accuracy, lengths, lag, position):
        x = self.item_emb(hist_bundle) + self.part_emb(hist_part) + self.accuracy_emb(hist_accuracy)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        h = h[-1] * (lengths > 0).unsqueeze(-1).float()
        z = torch.cat([h, self.lag_emb(lag), self.position_emb(position)], dim=-1)
        return F.normalize(self.user_mlp(z), dim=-1)

    def logits(self, users, items):
        return self.log_scale.exp() * users @ items.T


def accuracy_bucket(accuracy):
    return np.clip(np.floor(accuracy * 4.999).astype(np.int64) + 1, 1, ACCURACY_BUCKETS - 1)


def decision_indices(cont, min_history=1):
    idx = np.arange(len(cont))
    starts = cont.offsets[cont.user_index]
    return idx[idx - starts >= min_history]


def build_batch(cont, idx, history):
    starts = cont.offsets[cont.user_index[idx]]
    window = idx[:, None] - history + np.arange(history)[None, :]
    valid = window >= starts[:, None]
    lengths = valid.sum(axis=1)
    shift = history - lengths
    order = (np.arange(history)[None, :] + shift[:, None]) % history
    window = np.take_along_axis(window, order, axis=1)
    valid = np.take_along_axis(valid, order, axis=1)
    safe = np.where(valid, window, 0)
    return {
        "hist_bundle": np.where(valid, cont.bundle[safe] + 1, 0),
        "hist_part": np.where(valid, cont.part[safe], 0),
        "hist_accuracy": np.where(valid, accuracy_bucket(cont.accuracy[safe]), 0),
        "lengths": lengths,
        "lag": cont.lag[idx],
        "position": cont.position[idx],
        "target": cont.bundle[idx],
    }


def to_device(batch, dev):
    return {k: torch.as_tensor(v, dtype=torch.long, device=dev) for k, v in batch.items()}


def user_vectors(model, batch):
    return model.user_vectors(batch["hist_bundle"], batch["hist_part"], batch["hist_accuracy"], batch["lengths"],
                              batch["lag"], batch["position"])


def train(model, cont, cfg, dev, log=print):
    rng = np.random.default_rng(cfg.seed)
    decisions = decision_indices(cont)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    model.to(dev)
    for epoch in range(cfg.epochs):
        model.train()
        sample = rng.permutation(decisions)[: cfg.examples_per_epoch]
        total, n = 0.0, 0
        for start in range(0, len(sample), cfg.batch_size):
            batch = to_device(build_batch(cont, sample[start:start + cfg.batch_size], cfg.history), dev)
            logits = model.logits(user_vectors(model, batch), model.item_vectors())
            loss = F.cross_entropy(logits, batch["target"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(batch["target"])
            n += len(batch["target"])
        log(f"epoch {epoch + 1} loss {total / max(n, 1):.4f}")
    return model


@torch.no_grad()
def encode(model, cont, idx, history, dev, batch_size=4096):
    model.eval()
    out = []
    for start in range(0, len(idx), batch_size):
        batch = to_device(build_batch(cont, idx[start:start + batch_size], history), dev)
        out.append(user_vectors(model, batch).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


@torch.no_grad()
def item_matrix(model):
    model.eval()
    return model.item_vectors().cpu().numpy().astype(np.float32)


def build_index(items, kind="flat"):
    dim = items.shape[1]
    if kind == "hnsw":
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = 128
    else:
        index = faiss.IndexFlatIP(dim)
    index.add(items)
    return index


def ranking_metrics(ranked, targets, ks=(10, 50, 100)):
    hits = ranked == targets[:, None]
    pos = np.where(hits.any(axis=1), hits.argmax(axis=1), -1)
    metrics = {}
    for k in ks:
        found = (pos >= 0) & (pos < k)
        metrics[f"recall@{k}"] = float(found.mean())
        metrics[f"ndcg@{k}"] = float(np.where(found, 1.0 / np.log2(np.maximum(pos, 0) + 2.0), 0.0).mean())
    return metrics


def popularity_ranking(train_cont, n_bundles, n_queries, k):
    counts = np.bincount(train_cont.bundle, minlength=n_bundles)
    top = np.argsort(-counts, kind="stable")[:k]
    return np.tile(top, (n_queries, 1))


def latency_ms(index, queries, k, repeats=200):
    timings = []
    for i in range(min(repeats, len(queries))):
        t0 = time.perf_counter()
        index.search(queries[i:i + 1], k)
        timings.append((time.perf_counter() - t0) * 1000)
    timings = np.array(timings)
    return {"p50_ms": float(np.percentile(timings, 50)), "p99_ms": float(np.percentile(timings, 99))}


def evaluate(model, train_cont, eval_cont, cfg, dev, max_queries=50_000, seed=0):
    rng = np.random.default_rng(seed)
    idx = decision_indices(eval_cont)
    if len(idx) > max_queries:
        idx = np.sort(rng.choice(idx, size=max_queries, replace=False))
    queries = encode(model, eval_cont, idx, cfg.history, dev)
    items = item_matrix(model)
    k = 100
    targets = eval_cont.bundle[idx]
    results = {"queries": int(len(idx)), "n_bundles": int(items.shape[0])}
    flat = build_index(items, "flat")
    _, ranked = flat.search(queries, k)
    results["two_tower"] = ranking_metrics(ranked, targets)
    results["popularity"] = ranking_metrics(popularity_ranking(train_cont, items.shape[0], len(idx), k), targets)
    hnsw = build_index(items, "hnsw")
    _, approx = hnsw.search(queries, k)
    results["hnsw_overlap@100"] = float(np.mean([len(np.intersect1d(a, b)) / k for a, b in zip(ranked, approx)]))
    results["latency_flat"] = latency_ms(flat, queries, k)
    results["latency_hnsw"] = latency_ms(hnsw, queries, k)
    return results


def save(model, cfg, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "two_tower.pt")
    save_json(out / "two_tower_config.json", asdict(cfg))


def load_model(out_dir, n_bundles, bundle_part, dev):
    cfg = TwoTowerConfig(**load_json(Path(out_dir) / "two_tower_config.json"))
    model = TwoTower(n_bundles, bundle_part, cfg.dim)
    model.load_state_dict(torch.load(Path(out_dir) / "two_tower.pt", map_location=dev))
    return model.to(dev).eval(), cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/features")
    parser.add_argument("--out", default="runs/latest")
    parser.add_argument("--epochs", type=int, default=TwoTowerConfig.epochs)
    parser.add_argument("--examples-per-epoch", type=int, default=TwoTowerConfig.examples_per_epoch)
    parser.add_argument("--eval-split", default="valid")
    args = parser.parse_args()
    cfg = TwoTowerConfig(epochs=args.epochs, examples_per_epoch=args.examples_per_epoch)
    seed_everything(cfg.seed)
    dev = device()
    bundles = sequences.load_bundles(args.features)
    train_cont = sequences.containers(sequences.load(args.features, "train"))
    eval_cont = sequences.containers(sequences.load(args.features, args.eval_split))
    model = TwoTower(len(bundles), bundles["part"].to_numpy(), cfg.dim)
    train(model, train_cont, cfg, dev)
    save(model, cfg, args.out)
    results = evaluate(model, train_cont, eval_cont, cfg, dev)
    save_json(Path(args.out) / f"retrieval_{args.eval_split}.json", results)
    print(results)


if __name__ == "__main__":
    main()
