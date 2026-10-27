import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn

from adaptive_practice.common import device, load_json, save_json, seed_everything
from adaptive_practice.data.schema import N_PARTS
from adaptive_practice.features import sequences

TASKS = ("correct", "elapsed", "continue")


@dataclass
class RankerConfig:
    dim: int = 128
    heads: int = 4
    encoder_layers: int = 2
    decoder_layers: int = 2
    experts: int = 4
    window: int = 100
    dropout: float = 0.1
    batch_size: int = 256
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-5
    windows_per_epoch: int = 400_000
    loss_weights: tuple = (1.0, 0.5, 1.0)
    seed: int = 0


class CrossLayer(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_ff = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory, mask):
        attended, _ = self.attn(self.norm_q(x), memory, memory, attn_mask=mask, need_weights=False)
        x = x + self.drop(attended)
        return x + self.drop(self.ff(self.norm_ff(x)))


class MMoE(nn.Module):
    def __init__(self, dim, experts, tasks):
        super().__init__()
        self.experts = nn.ModuleList(nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)) for _ in range(experts))
        self.gates = nn.ModuleList(nn.Linear(dim, experts) for _ in range(tasks))
        self.towers = nn.ModuleList(nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1)) for _ in range(tasks))

    def forward(self, x):
        expert_out = torch.stack([e(x) for e in self.experts], dim=-2)
        outputs = []
        for gate, tower in zip(self.gates, self.towers):
            w = torch.softmax(gate(x), dim=-1).unsqueeze(-1)
            outputs.append(tower((w * expert_out).sum(dim=-2)).squeeze(-1))
        return outputs


class Ranker(nn.Module):
    def __init__(self, n_questions, cfg):
        super().__init__()
        d = cfg.dim
        self.cfg = cfg
        self.question = nn.Embedding(n_questions + 1, d, padding_idx=0)
        self.part = nn.Embedding(N_PARTS + 1, d, padding_idx=0)
        self.correct = nn.Embedding(2, d)
        self.elapsed = nn.Embedding(sequences.ELAPSED_BUCKETS, d)
        self.lag = nn.Embedding(sequences.LAG_BUCKETS, d)
        self.session_position = nn.Embedding(sequences.POSITION_BUCKETS, d)
        self.explained = nn.Embedding(2, d)
        self.position = nn.Embedding(cfg.window + 1, d)
        self.dense = nn.Sequential(nn.Linear(sequences.DENSE_FEATURES, d), nn.GELU(), nn.Linear(d, d))
        self.bos = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, cfg.heads, d * 4, cfg.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.encoder_layers, enable_nested_tensor=False)
        self.decoder = nn.ModuleList(CrossLayer(d, cfg.heads, cfg.dropout) for _ in range(cfg.decoder_layers))
        self.norm = nn.LayerNorm(d)
        self.heads = MMoE(d, cfg.experts, len(TASKS))

    def encode(self, b):
        n = b["question"].shape[1]
        pos = torch.arange(n, device=b["question"].device)
        x = (self.question(b["question"]) + self.part(b["part"]) + self.correct(b["correct"]) + self.elapsed(b["elapsed"])
             + self.lag(b["lag"]) + self.position(pos))
        causal = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), diagonal=1)
        h = self.encoder(x, mask=causal, src_key_padding_mask=~b["valid"], is_causal=False)
        bos = self.bos.expand(h.shape[0], 1, -1)
        return torch.cat([bos, h], dim=1)

    def query(self, q):
        return (self.question(q["question"]) + self.part(q["part"]) + self.lag(q["lag"])
                + self.session_position(q["session_position"]) + self.explained(q["explained"]) + self.position(q["slot"])
                + self.dense(q["dense"]))

    def decode(self, memory, queries, visible):
        mask = torch.zeros(visible.shape, dtype=memory.dtype, device=memory.device)
        mask = mask.masked_fill(~visible, float("-inf"))
        mask = mask.repeat_interleave(self.cfg.heads, dim=0)
        x = queries
        for layer in self.decoder:
            x = layer(x, memory, mask)
        return self.heads(self.norm(x))

    def forward(self, b):
        memory = self.encode(b)
        queries = self.query({"question": b["question"], "part": b["part"], "lag": b["lag"],
                              "session_position": b["session_position"], "explained": b["explained"], "dense": b["dense"],
                              "slot": torch.arange(b["question"].shape[1], device=memory.device).expand_as(b["question"])})
        c = b["container"]
        earlier = (c.unsqueeze(2) > c.unsqueeze(1)) & b["valid"].unsqueeze(1)
        visible = torch.cat([torch.ones_like(earlier[:, :, :1]), earlier], dim=2)
        return self.decode(memory, queries, visible)


def windows(seq, window):
    starts, loss_from = [], []
    half = window // 2
    for u in range(seq.n_users):
        lo, hi = seq.offsets[u], seq.offsets[u + 1]
        starts.append(lo)
        loss_from.append(lo)
        s = lo + half
        while s + half < hi:
            starts.append(s)
            loss_from.append(s + half)
            s += half
    return np.array(starts, dtype=np.int64), np.array(loss_from, dtype=np.int64)


def user_end(seq, starts):
    user = np.searchsorted(seq.offsets, starts, side="right") - 1
    return seq.offsets[user + 1]


def build_batch(seq, starts, loss_from, window):
    ends = np.minimum(starts + window, user_end(seq, starts))
    idx = starts[:, None] + np.arange(window)[None, :]
    valid = idx < ends[:, None]
    safe = np.where(valid, idx, 0)
    loss = valid & (idx >= loss_from[:, None])
    take = lambda a, fill=0: np.where(valid, a[safe], fill)
    return {
        "question": take(seq.question),
        "part": take(seq.part),
        "correct": take(seq.correct),
        "elapsed": take(seq.elapsed),
        "lag": take(seq.lag),
        "session_position": take(seq.position),
        "explained": take(seq.explained),
        "container": take(seq.container, -1),
        "dense": np.where(valid[..., None], seq.dense[safe], 0.0).astype(np.float32),
        "valid": valid,
        "loss": loss,
        "y_correct": take(seq.correct).astype(np.float32),
        "y_elapsed": take(seq.log_elapsed, 0.0).astype(np.float32),
        "m_elapsed": loss & (take(seq.elapsed) > 0),
        "y_continue": np.maximum(take(seq.cont, -1), 0).astype(np.float32),
        "m_continue": loss & (take(seq.cont, -1) >= 0),
        "rows": np.where(valid, idx, -1),
    }


def to_device(batch, dev):
    out = {}
    for k, v in batch.items():
        if v.dtype == np.bool_:
            out[k] = torch.as_tensor(v, device=dev)
        elif v.dtype == np.float32:
            out[k] = torch.as_tensor(v, device=dev)
        else:
            out[k] = torch.as_tensor(v, dtype=torch.long, device=dev)
    return out


def losses(outputs, b, weights):
    logit_c, pred_e, logit_k = outputs
    m = b["loss"]
    l_c = F.binary_cross_entropy_with_logits(logit_c[m], b["y_correct"][m])
    me = b["m_elapsed"]
    l_e = F.huber_loss(pred_e[me], b["y_elapsed"][me]) if me.any() else logit_c.sum() * 0
    mk = b["m_continue"]
    l_k = F.binary_cross_entropy_with_logits(logit_k[mk], b["y_continue"][mk]) if mk.any() else logit_c.sum() * 0
    total = weights[0] * l_c + weights[1] * l_e + weights[2] * l_k
    return total, (l_c.item(), l_e.item(), l_k.item())


def train(model, seq, cfg, dev, log=print):
    rng = np.random.default_rng(cfg.seed)
    starts, loss_from = windows(seq, cfg.window)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * int(np.ceil(min(len(starts), cfg.windows_per_epoch) / cfg.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=max(steps, 1), pct_start=0.1)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.to(dev)
    for epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(len(starts))[: cfg.windows_per_epoch]
        sums, n = np.zeros(3), 0
        for i in range(0, len(order), cfg.batch_size):
            pick = order[i:i + cfg.batch_size]
            b = to_device(build_batch(seq, starts[pick], loss_from[pick], cfg.window), dev)
            with torch.autocast(dev.type, enabled=use_amp):
                outputs = model(b)
            total, parts = losses([o.float() for o in outputs], b, cfg.loss_weights)
            opt.zero_grad()
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            sums += parts
            n += 1
        log(f"epoch {epoch + 1} correct {sums[0] / n:.4f} elapsed {sums[1] / n:.4f} continue {sums[2] / n:.4f}")
    return model


@torch.no_grad()
def predict(model, seq, cfg, dev, batch_size=512):
    model.eval()
    starts, loss_from = windows(seq, cfg.window)
    p_correct = np.full(len(seq.question), np.nan)
    elapsed = np.full(len(seq.question), np.nan)
    p_continue = np.full(len(seq.question), np.nan)
    for i in range(0, len(starts), batch_size):
        b = build_batch(seq, starts[i:i + batch_size], loss_from[i:i + batch_size], cfg.window)
        out = model(to_device(b, dev))
        rows = b["rows"][b["loss"]]
        m = torch.as_tensor(b["loss"], device=dev)
        p_correct[rows] = torch.sigmoid(out[0][m]).float().cpu().numpy()
        elapsed[rows] = out[1][m].float().cpu().numpy()
        p_continue[rows] = torch.sigmoid(out[2][m]).float().cpu().numpy()
    return p_correct, elapsed, p_continue


def task_metrics(seq, p_correct, elapsed, p_continue):
    first = sequences.containers(seq).first_row
    known_cont = seq.cont[first] >= 0
    has_elapsed = seq.elapsed > 0
    true_seconds = np.expm1(seq.log_elapsed[has_elapsed])
    pred_seconds = np.expm1(elapsed[has_elapsed])
    return {
        "rows": int(len(seq.question)),
        "auc_correct": float(roc_auc_score(seq.correct, p_correct)),
        "auc_continue": float(roc_auc_score(seq.cont[first][known_cont], p_continue[first][known_cont])),
        "mae_elapsed_seconds": float(np.mean(np.abs(true_seconds - pred_seconds))),
    }


def query_tensors(q, dev):
    return {k: torch.as_tensor(v, dtype=torch.long, device=dev) for k, v in q.items()}


@torch.no_grad()
def score_candidates(model, seq, table, user, upto, candidates, lag, session_position, explained, dev):
    cfg = model.cfg
    lo = max(seq.offsets[user], upto - (cfg.window - 1))
    n = upto - lo
    b = build_batch(seq, np.array([lo]), np.array([lo]), cfg.window)
    if n < cfg.window:
        b["valid"][0, n:] = False
    if n == 0:
        b["valid"][0, 0] = True
    tb = to_device(b, dev)
    memory = model.encode(tb)
    questions = np.concatenate(candidates)
    owner = np.repeat(np.arange(len(candidates)), [len(c) for c in candidates])
    k = len(questions)
    q = query_tensors({
        "question": questions[None, :],
        "part": np.zeros((1, k), dtype=np.int64),
        "lag": np.full((1, k), lag),
        "session_position": np.full((1, k), session_position),
        "explained": np.full((1, k), explained),
        "slot": np.minimum(n + np.concatenate([np.arange(len(c)) for c in candidates]), cfg.window)[None, :],
    }, dev)
    q["part"] = model_part(model, q["question"])
    q["dense"] = torch.as_tensor(sequences.online_dense(seq, table, user, upto, questions)[None], device=dev)
    visible = torch.zeros((1, k, cfg.window + 1), dtype=torch.bool, device=dev)
    visible[0, :, : n + 1] = True
    out = model.decode(memory, model.query(q), visible)
    pc = torch.sigmoid(out[0][0]).float().cpu().numpy()
    el = np.expm1(out[1][0].float().cpu().numpy())
    pk = torch.sigmoid(out[2][0]).float().cpu().numpy()
    count = np.bincount(owner)
    return {
        "p_correct": np.bincount(owner, weights=pc) / count,
        "seconds": np.bincount(owner, weights=el),
        "p_continue": np.bincount(owner, weights=pk) / count,
    }


def model_part(model, questions):
    return model.question_part[questions]


def attach_question_parts(model, question_part):
    model.register_buffer("question_part", torch.tensor(np.array(question_part), dtype=torch.long))
    return model


def save(model, cfg, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "ranker.pt")
    payload = asdict(cfg)
    payload["loss_weights"] = list(cfg.loss_weights)
    save_json(out / "ranker_config.json", payload)


def question_parts(features_dir):
    questions = sequences.load_questions(features_dir)
    parts = np.zeros(int(questions["content_id"].max()) + 2, dtype=np.int64)
    parts[questions["content_id"].to_numpy() + 1] = questions["part"].to_numpy()
    return parts


def load_model(out_dir, features_dir, dev):
    raw = load_json(Path(out_dir) / "ranker_config.json")
    raw["loss_weights"] = tuple(raw["loss_weights"])
    cfg = RankerConfig(**raw)
    parts = question_parts(features_dir)
    model = attach_question_parts(Ranker(len(parts) - 1, cfg), parts)
    model.load_state_dict(torch.load(Path(out_dir) / "ranker.pt", map_location=dev))
    return model.to(dev).eval(), cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/features")
    parser.add_argument("--out", default="runs/latest")
    parser.add_argument("--epochs", type=int, default=RankerConfig.epochs)
    parser.add_argument("--windows-per-epoch", type=int, default=RankerConfig.windows_per_epoch)
    parser.add_argument("--batch-size", type=int, default=RankerConfig.batch_size)
    parser.add_argument("--eval-split", default="valid")
    args = parser.parse_args()
    cfg = RankerConfig(epochs=args.epochs, windows_per_epoch=args.windows_per_epoch, batch_size=args.batch_size)
    seed_everything(cfg.seed)
    dev = device()
    parts = question_parts(args.features)
    model = attach_question_parts(Ranker(len(parts) - 1, cfg), parts)
    train(model, sequences.load(args.features, "train"), cfg, dev)
    save(model, cfg, args.out)
    eval_seq = sequences.load(args.features, args.eval_split)
    results = task_metrics(eval_seq, *predict(model, eval_seq, cfg, dev))
    save_json(Path(args.out) / f"ranker_{args.eval_split}.json", results)
    print(results)


if __name__ == "__main__":
    main()
