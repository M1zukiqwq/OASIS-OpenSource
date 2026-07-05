#!/usr/bin/env python3
"""Local simulation: optimizing the OASIS learned SINGLE-COLUMN prior.

Mirrors OasisTorchV3 in essence (a learned 1D histogram refiner that takes a stale
prior + K feedback range-observations and predicts the current distribution, with a
residual-over-stale mode). The documented failure is a synthetic->real generalization
gap: the learned prior wins in-distribution but barely beats the analytic ISOMER
projection on real data. This harness tests, HONESTLY:

  TRAIN on synthetic drifted columns; TEST on REAL held-out columns (true OOD).
  Baselines: stale, ISOMER (1D IPF projection of stale onto feedback).
  Learned framings: direct | residual | residual+gate.
  Training distribution: single drift family  vs  OOD-augmented (all families).

A learned variant only "wins" if, after the SAME feedback projection everyone gets,
it beats ISOMER on the REAL test columns (held-out range-predicate Q-error).

Run with the torch venv:
  .venv_torch/bin/python oasis_prior_sim.py --epochs 40
"""
from __future__ import annotations

import argparse
import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn

import crosscol_feedback_experiment as base

G = 32                      # histogram bins over [0,1]
torch.manual_seed(0)


# ------------------------------- distributions -------------------------------

def _norm_hist(v, g=G):
    lo, hi = np.percentile(v, [0.5, 99.5]); hi = max(hi, lo + 1e-9)
    u = np.clip((v - lo) / (hi - lo), 0, 1)
    h, _ = np.histogram(u, bins=g, range=(0, 1))
    h = h.astype(float) + 1e-6
    return h / h.sum()


def real_columns():
    """Pull single columns from the real pair datasets -> list of (name, hist)."""
    cols = []
    seen = set()
    for (name, path, sep, skip, ca, cb) in base.PAIRS:
        try:
            x, y = base.read_pair(path, sep, skip, ca, cb)
        except FileNotFoundError:
            continue
        for tag, v in ((name.split(":")[0] + ":" + name.split("~")[0].split(":")[-1], x),
                       (name.split(":")[0] + ":" + name.split("~")[-1], y)):
            if tag in seen or len(v) < 500:
                continue
            seen.add(tag)
            cols.append((tag, _norm_hist(np.asarray(v))))
    return cols


# ------------------------------- drift families ------------------------------
# stale = base distribution; true = drift(base). model sees stale + feedback(true).

def drift(base_h, family, rng, g=G):
    h = base_h.copy()
    x = np.arange(g)
    if family == "shift":
        k = rng.integers(-6, 7)
        h = np.roll(h, k)
    elif family == "skew":
        p = rng.uniform(0.5, 2.0)
        idx = (np.linspace(0, 1, g) ** p * (g - 1)).astype(int)
        h = np.bincount(idx, weights=h, minlength=g)[:g]
    elif family == "reweight":
        c = rng.uniform(0, 1); s = rng.uniform(2, 8)
        w = np.exp(-((np.linspace(0, 1, g) - c) ** 2) * s)
        h = h * (0.3 + w)
    elif family == "spike":
        c = rng.integers(0, g); h = h.copy(); h[c] += h.sum() * rng.uniform(0.2, 0.6)
    elif family == "contrast":
        gamma = rng.uniform(0.5, 2.0); h = h ** gamma
    elif family == "tailfat":
        h = h + h.max() * rng.uniform(0.02, 0.1) * (x / g)
    h = np.clip(h, 1e-6, None)
    return h / h.sum()


def synth_base(rng, g=G):
    """random smooth-ish base distribution."""
    k = rng.integers(1, 4)
    h = np.zeros(g)
    xs = np.linspace(0, 1, g)
    for _ in range(k):
        c = rng.uniform(0, 1); s = rng.uniform(4, 30); a = rng.uniform(0.3, 1.0)
        h += a * np.exp(-((xs - c) ** 2) * s)
    h += rng.uniform(0.02, 0.1)
    return h / h.sum()


# ------------------------------- analytic ISOMER -----------------------------

def ipf_1d(prior, intervals, targets, n_iter=60):
    m = prior.copy()
    for _ in range(n_iter):
        for (a, b), t in zip(intervals, targets):
            cur = m[a:b].sum()
            if cur > 1e-12:
                m[a:b] *= t / cur
        s = m.sum()
        if s > 1e-12:
            m /= s
    return m


def feedback(true_h, k, rng, g=G):
    iv, tg = [], []
    while len(iv) < k:
        a, b = sorted(rng.sample(range(g + 1), 2))
        iv.append((a, b)); tg.append(float(true_h[a:b].sum()))
    return iv, tg


def coverage_vec(intervals, g=G):
    c = np.zeros(g)
    for (a, b) in intervals:
        c[a:b] += 1.0
    return c / max(c.max(), 1.0)


def hist_qerr(est, true, rng, n=200, g=G, min_true=1e-3):
    qs = []
    for _ in range(n):
        a, b = sorted(rng.sample(range(g + 1), 2))
        t = float(true[a:b].sum())
        if t >= min_true:
            qs.append(base.qerr(float(est[a:b].sum()), t))
    return float(np.exp(np.mean(np.log(qs)))) if qs else float("nan")


# --------------------------------- model -------------------------------------

class PriorNet(nn.Module):
    def __init__(self, g=G, hidden=128, mode="residual"):
        super().__init__()
        self.mode = mode
        self.net = nn.Sequential(
            nn.Linear(3 * g, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, g),
        )
        if mode == "gate":
            self.gate = nn.Sequential(nn.Linear(3 * g, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, stale, isomer, cov):
        """all inputs (Bsz, g) in probability space. returns predicted hist (Bsz, g)."""
        feat = torch.cat([stale, isomer, cov], dim=-1)
        out = self.net(feat)
        if self.mode == "direct":
            return torch.softmax(out, dim=-1)
        delta = torch.tanh(out)                       # bounded correction
        if self.mode == "gate":
            g_ = torch.sigmoid(self.gate(feat))       # (Bsz,1) per-instance trust
            delta = g_ * delta
        pred = stale * torch.exp(delta)
        return pred / pred.sum(dim=-1, keepdim=True)


# ----------------------------- instance assembly -----------------------------

def build_instance(base_h, family, k, rng_np, rng_py):
    true_h = drift(base_h, family, rng_np)
    stale = base_h
    iv, tg = feedback(true_h, k, rng_py)
    iso = ipf_1d(stale, iv, tg)
    cov = coverage_vec(iv)
    return stale, iso, cov, true_h, iv, tg


def to_t(*arrs):
    return [torch.tensor(np.asarray(a), dtype=torch.float32) for a in arrs]


# --------------------------------- training ----------------------------------

def make_batch(families, n, rng_np, rng_py, k_choices=(2, 4, 8, 16)):
    S, I, C, T = [], [], [], []
    for _ in range(n):
        b = synth_base(rng_np)
        fam = families[rng_np.integers(0, len(families))]
        k = int(rng_py.choice(k_choices))
        stale, iso, cov, true_h, iv, tg = build_instance(b, fam, k, rng_np, rng_py)
        S.append(stale); I.append(iso); C.append(cov); T.append(true_h)
    return to_t(S, I, C, T)


def build_pool(families, n, seed, k_choices=(2, 4, 8, 16)):
    """Precompute a fixed pool of training instances once (avoids per-batch IPF)."""
    rng_np = np.random.default_rng(seed)
    rng_py = random.Random(seed)
    S, I, C, T = make_batch(families, n, rng_np, rng_py, k_choices)  # already (n,g) tensors
    return S, I, C, T


def downstream_loss(pred, true, n_pred=24):
    """log-Q-error on random range predicates (matches train.py future objective)."""
    g = pred.shape[-1]
    cs_p = torch.cumsum(pred, dim=-1)
    cs_t = torch.cumsum(true, dim=-1)
    cs_p = torch.cat([torch.zeros_like(cs_p[:, :1]), cs_p], dim=-1)
    cs_t = torch.cat([torch.zeros_like(cs_t[:, :1]), cs_t], dim=-1)
    loss = 0.0
    for _ in range(n_pred):
        a = torch.randint(0, g, (1,)).item()
        b = torch.randint(a + 1, g + 1, (1,)).item()
        sp = (cs_p[:, b] - cs_p[:, a]).clamp_min(1e-6)
        st = (cs_t[:, b] - cs_t[:, a]).clamp_min(1e-6)
        loss = loss + (torch.log(sp) - torch.log(st)).abs().mean()
    return loss / n_pred


def train(mode, families, epochs, steps, bs, lr, seed=0, pool=None, pool_n=6000):
    torch.manual_seed(seed)
    if pool is None:
        pool = build_pool(families, pool_n, seed)
    Sp, Ip, Cp, Tp = pool
    N = Sp.shape[0]
    g = torch.Generator().manual_seed(seed)
    model = PriorNet(mode=mode)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * steps)
    model.train()
    for ep in range(epochs):
        for _ in range(steps):
            idx = torch.randint(0, N, (bs,), generator=g)
            pred = model(Sp[idx], Ip[idx], Cp[idx])
            loss = downstream_loss(pred, Tp[idx])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
    model.eval()
    return model


# ------------------------------ real-data eval -------------------------------

def evaluate_real(model, cols, families, k_list=(2, 6, 16), trials=20, seed=123):
    """test on REAL columns with drift; compare learned vs stale/ISOMER after projection."""
    rng_np = np.random.default_rng(seed)
    rng_py = random.Random(seed)
    out = {}
    for k in k_list:
        qs = {"stale": [], "isomer": [], "learned": [], "learned_proj": []}
        for (name, base_h) in cols:
            for _ in range(trials):
                fam = families[rng_np.integers(0, len(families))]
                stale, iso, cov, true_h, iv, tg = build_instance(base_h, fam, k, rng_np, rng_py)
                with torch.no_grad():
                    St, It, Ct = to_t([stale], [iso], [cov])
                    pred = model(St, It, Ct)[0].numpy()
                pred_proj = ipf_1d(pred, iv, tg)       # same projection everyone gets
                erng = random.Random(rng_py.randint(0, 1 << 30))
                qs["stale"].append(hist_qerr(stale, true_h, erng))
                erng = random.Random(erng.randint(0, 1 << 30))
                qs["isomer"].append(hist_qerr(iso, true_h, erng))
                erng = random.Random(erng.randint(0, 1 << 30))
                qs["learned"].append(hist_qerr(pred, true_h, erng))
                erng = random.Random(erng.randint(0, 1 << 30))
                qs["learned_proj"].append(hist_qerr(pred_proj, true_h, erng))
        out[k] = {m: float(np.exp(np.mean(np.log(v)))) for m, v in qs.items()}
    return out


ALL_FAMILIES = ["shift", "skew", "reweight", "spike", "contrast", "tailfat"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--modes", nargs="+", default=["direct", "residual", "gate"])
    ap.add_argument("--train-family", default="all",
                    help="'all' = OOD-augmented, or a single family name (in-distribution)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--pool-n", type=int, default=6000)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results/oasis_prior_sim"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    cols = real_columns()
    print(f"real test columns: {len(cols)} -> {[c[0] for c in cols]}")
    if a.smoke:
        rng_np = np.random.default_rng(1); rng_py = random.Random(1)
        S, I, C, T = make_batch(ALL_FAMILIES, 4, rng_np, rng_py)
        m = PriorNet(mode="residual")
        print("forward ok:", m(S, I, C).shape, "loss:", float(downstream_loss(m(S, I, C), T)))
        return

    train_fams = ALL_FAMILIES if a.train_family == "all" else [a.train_family]
    os.makedirs(a.out, exist_ok=True)

    # analytic-only reference (no learning) on real
    ref = evaluate_real(PriorNet(mode="direct"), cols, ALL_FAMILIES)  # learned cols ignored below
    print("\n=== REAL test (held-out range-predicate geom Q-err; lower better) ===")
    print(f"train family: {a.train_family}   (test always = all real columns x all drift families)")
    hdr = "  ".join(f"K={k}" for k in (2, 6, 16))
    print(f"{'method':>16} | {hdr}")
    print("-" * 46)
    print(f"{'stale':>16} | " + "  ".join(f"{ref[k]['stale']:.3f}" for k in (2, 6, 16)))
    print(f"{'ISOMER':>16} | " + "  ".join(f"{ref[k]['isomer']:.3f}" for k in (2, 6, 16)))

    # build the training pool ONCE (the expensive step) and reuse across modes/seeds
    pool = build_pool(train_fams, a.pool_n, seed=1234)

    rows = [{"method": "ISOMER", **{f"k{k}": ref[k]["isomer"] for k in (2, 6, 16)}}]
    for mode in a.modes:
        # per-seed deltas vs ISOMER (mean +/- range across model seeds)
        per_seed = {k: [] for k in (2, 6, 16)}
        last = None
        for sd in a.seeds:
            model = train(mode, train_fams, a.epochs, a.steps, a.bs, a.lr, seed=sd, pool=pool)
            res = evaluate_real(model, cols, ALL_FAMILIES)
            last = res
            for k in (2, 6, 16):
                per_seed[k].append((ref[k]["isomer"] - res[k]["learned_proj"]) / ref[k]["isomer"] * 100)
        def fmt(k):
            d = per_seed[k]
            return f"K{k}:{np.mean(d):+.1f}%[{min(d):+.1f},{max(d):+.1f}]"
        print(f"{('learned:'+mode):>16} | " +
              "  ".join(f"{last[k]['learned_proj']:.3f}" for k in (2, 6, 16)) +
              "   vs ISOMER " + " ".join(fmt(k) for k in (2, 6, 16)))
        rows.append({"method": f"learned_proj:{mode}",
                     **{f"k{k}": last[k]["learned_proj"] for k in (2, 6, 16)},
                     **{f"k{k}_delta_mean": float(np.mean(per_seed[k])) for k in (2, 6, 16)}})

    with open(os.path.join(a.out, f"summary_{a.train_family}.csv"), "w", newline="") as h:
        fns = ["method", "k2", "k6", "k16", "k2_delta_mean", "k6_delta_mean", "k16_delta_mean"]
        w = csv.DictWriter(h, fieldnames=fns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\nwrote {a.out}/summary_{a.train_family}.csv")


if __name__ == "__main__":
    main()
