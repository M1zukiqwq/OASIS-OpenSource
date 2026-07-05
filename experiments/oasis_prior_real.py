#!/usr/bin/env python3
"""Validate the sim finding on the REAL OasisTorchV3 transformer (remote GPU).

Mirrors oasis_prior_sim.py EXACTLY (synthetic-OOD drift training -> 16 real held-out
columns; direct/residual framings; K=2/6/16 vs ISOMER; held-out range-predicate geom
Q-error) but swaps the simplified MLP for the real OasisTorchV3 (transformer, equi-depth
boundary output, residual-around-stale) and its differentiable ops. The ONLY changed
variable is the model architecture -> tests whether the sim's +0.9%@K2 edge survives.

Self-contained except for model.py (OasisTorchV3 + ops) and real_columns.npz.
Run on remote:  python3 oasis_prior_real.py --device cuda --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn

from model import (OasisTorchV3, OBS_FEAT_DIM, PREDICATE_ORDER,
                   boundaries_from_logits, cell_masses, sel_from_masses,
                   interval_coverage, ipf_project)

G = 32
B_BUCKETS = 10
BETWEEN = PREDICATE_ORDER.index("BETWEEN")


def qerr(est, true):
    est = max(float(est), 1e-9); true = max(float(true), 1e-9)
    return max(est / true, true / est)


def gm(xs):
    return float(np.exp(np.mean(np.log(xs))))


# ------------------------------ distributions (mirror sim) -------------------

def synth_base(rng, g=G):
    k = rng.integers(1, 4); h = np.zeros(g); xs = np.linspace(0, 1, g)
    for _ in range(k):
        c = rng.uniform(0, 1); s = rng.uniform(4, 30); a = rng.uniform(0.3, 1.0)
        h += a * np.exp(-((xs - c) ** 2) * s)
    h += rng.uniform(0.02, 0.1)
    return h / h.sum()


def drift(base_h, family, rng, g=G):
    h = base_h.copy(); x = np.arange(g)
    if family == "shift":
        h = np.roll(h, int(rng.integers(-6, 7)))
    elif family == "skew":
        p = rng.uniform(0.5, 2.0); idx = (np.linspace(0, 1, g) ** p * (g - 1)).astype(int)
        h = np.bincount(idx, weights=h, minlength=g)[:g]
    elif family == "reweight":
        c = rng.uniform(0, 1); s = rng.uniform(2, 8)
        h = h * (0.3 + np.exp(-((np.linspace(0, 1, g) - c) ** 2) * s))
    elif family == "spike":
        c = int(rng.integers(0, g)); h = h.copy(); h[c] += h.sum() * rng.uniform(0.2, 0.6)
    elif family == "contrast":
        h = h ** rng.uniform(0.5, 2.0)
    elif family == "tailfat":
        h = h + h.max() * rng.uniform(0.02, 0.1) * (x / g)
    h = np.clip(h, 1e-6, None)
    return h / h.sum()


ALL_FAMILIES = ["shift", "skew", "reweight", "spike", "contrast", "tailfat"]


# ------------------------------ representation -------------------------------

GRID = torch.linspace(0, 1, G + 1)
LEVELS = torch.linspace(0, 1, B_BUCKETS + 1)


def hist_to_boundaries(h, B=None, g=G):
    B = B_BUCKETS if B is None else B
    edges = np.linspace(0, 1, g + 1)
    cdf = np.concatenate([[0.0], np.cumsum(h)]); cdf = cdf / cdf[-1]
    bnd = [0.0] + [float(np.interp(i / B, cdf, edges)) for i in range(1, B)] + [1.0]
    bnd = np.maximum.accumulate(bnd)                    # ensure non-decreasing
    for i in range(1, len(bnd)):                        # ensure strictly increasing
        if bnd[i] <= bnd[i - 1]:
            bnd[i] = min(bnd[i - 1] + 1e-4, 1.0)
    return np.array(bnd)


def hist_to_masses_t(h, device):
    return torch.tensor(h, dtype=torch.float32, device=device).unsqueeze(0)  # (1,G)


def feedback(true_h, k, rng, g=G):
    iv, tg = [], []
    while len(iv) < k:
        a, b = sorted(rng.sample(range(g + 1), 2))
        iv.append((a, b)); tg.append(float(true_h[a:b].sum()))
    return iv, tg


def obs_tensor(stale_h, iv, tg, kmax=16, g=G):
    feats = np.zeros((kmax, OBS_FEAT_DIM), dtype=np.float32)
    mask = np.zeros(kmax, dtype=np.float32)
    for j, ((a, b), t) in enumerate(zip(iv, tg)):
        if j >= kmax:
            break
        lo, hi = a / g, b / g
        est = float(stale_h[a:b].sum())
        feats[j, BETWEEN] = 1.0
        feats[j, len(PREDICATE_ORDER) + 0] = lo
        feats[j, len(PREDICATE_ORDER) + 1] = hi
        feats[j, len(PREDICATE_ORDER) + 2] = est
        feats[j, len(PREDICATE_ORDER) + 3] = t
        feats[j, len(PREDICATE_ORDER) + 4] = est - t
        feats[j, len(PREDICATE_ORDER) + 5] = 1.0
        mask[j] = 1.0
    return feats, mask


# --------------------------------- pool --------------------------------------

def build_pool(families, n, seed, device, k_choices=(2, 4, 8, 16)):
    rng_np = np.random.default_rng(seed); rng_py = random.Random(seed)
    OF, OM, SB, TM, COV, TGT, FMASK = [], [], [], [], [], [], []
    for _ in range(n):
        base = synth_base(rng_np); fam = families[rng_np.integers(0, len(families))]
        true_h = drift(base, fam, rng_np); stale_h = base
        k = int(rng_py.choice(k_choices))
        iv, tg = feedback(true_h, k, rng_py)
        of, om = obs_tensor(stale_h, iv, tg)
        sb = hist_to_boundaries(stale_h)
        # interval coverage + targets for differentiable projection (padded to 16)
        cov = np.zeros((16, G), dtype=np.float32); tgt = np.zeros(16, dtype=np.float32)
        fmask = np.zeros(16, dtype=np.float32)
        for j, ((a, b), t) in enumerate(zip(iv, tg)):
            cov[j, a:b] = 1.0; tgt[j] = t; fmask[j] = 1.0
        OF.append(of); OM.append(om); SB.append(sb); TM.append(true_h)
        COV.append(cov); TGT.append(tgt); FMASK.append(fmask)
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=device)
    return t(OF), t(OM), t(SB), t(TM), t(COV), t(TGT), t(FMASK)


# --------------------------- loss / eval predicates --------------------------

def rand_pred_t(bsz, n, device, g=G):
    a = torch.randint(0, g, (bsz, n), device=device)
    b = torch.randint(0, g, (bsz, n), device=device)
    lo = torch.minimum(a, b).float() / g
    hi = (torch.maximum(a, b).float() + 1) / g
    pt = torch.full((bsz, n), float(BETWEEN), device=device)
    return pt, lo, hi


def downstream_loss(m_pred, true_m, device, n_pred=24):
    bsz = m_pred.shape[0]
    pt, lo, hi = rand_pred_t(bsz, n_pred, device)
    sp = sel_from_masses(m_pred, GRID.to(device), pt, lo, hi)
    st = sel_from_masses(true_m, GRID.to(device), pt, lo, hi)
    return (torch.log(sp.clamp_min(1e-6)) - torch.log(st.clamp_min(1e-6))).abs().mean()


def masses_from_model(model, of, om, sb, device):
    logits = model(of, om, sb)
    bnd = boundaries_from_logits(logits)
    return cell_masses(bnd, LEVELS.to(device), GRID.to(device))


# --------------------------------- train -------------------------------------

def train(pool, residual, epochs, steps, bs, lr, seed, device):
    torch.manual_seed(seed)
    OF, OM, SB, TM, COV, TGT, FMASK = pool
    N = OF.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    model = OasisTorchV3(num_buckets=B_BUCKETS, residual_prior=residual).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * steps)
    model.train()
    for _ in range(epochs):
        for _ in range(steps):
            idx = torch.randint(0, N, (bs,), generator=g).to(device)
            m = masses_from_model(model, OF[idx], OM[idx], SB[idx], device)
            loss = downstream_loss(m, TM[idx], device)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    model.eval()
    return model


# ------------------------------ real-data eval -------------------------------

def hist_qerr(est, true, rng, n=200, g=G, min_true=1e-3):
    qs = []
    for _ in range(n):
        a, b = sorted(rng.sample(range(g + 1), 2))
        t = float(true[a:b].sum())
        if t >= min_true:
            qs.append(qerr(float(est[a:b].sum()), t))
    return gm(qs) if qs else float("nan")


def evaluate_real(model, cols, device, k_list=(2, 6, 16), trials=20, seed=123):
    """Batched eval: build all (col x trial) instances per K, run model + projection
    once over the whole batch (avoids per-instance GPU kernel-launch overhead)."""
    rng_np = np.random.default_rng(seed); rng_py = random.Random(seed)
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=device)
    out = {}
    for k in k_list:
        OF, OM, SB, COV, TGT, FMASK, STALE, TRUE, ERS = [], [], [], [], [], [], [], [], []
        for (name, base_h) in cols:
            for _ in range(trials):
                fam = ALL_FAMILIES[rng_np.integers(0, len(ALL_FAMILIES))]
                true_h = drift(base_h, fam, rng_np); stale_h = base_h
                iv, tg = feedback(true_h, k, rng_py)
                of, om = obs_tensor(stale_h, iv, tg)
                cov = np.zeros((16, G), dtype=np.float32); tgt = np.zeros(16, dtype=np.float32)
                fmask = np.zeros(16, dtype=np.float32)
                for j, ((a, b), tt) in enumerate(zip(iv, tg)):
                    cov[j, a:b] = 1.0; tgt[j] = tt; fmask[j] = 1.0
                OF.append(of); OM.append(om); SB.append(hist_to_boundaries(stale_h))
                COV.append(cov); TGT.append(tgt); FMASK.append(fmask)
                STALE.append(stale_h); TRUE.append(true_h)
                ERS.append(rng_py.randint(0, 1 << 30))
        with torch.no_grad():
            m_pred = masses_from_model(model, t(OF), t(OM), t(SB), device)       # (Ninst,G)
            m_proj = ipf_project(m_pred, t(COV), t(TGT), t(FMASK), n_iter=20).cpu().numpy()
            iso = ipf_project(t(np.asarray(STALE)), t(COV), t(TGT), t(FMASK), n_iter=20).cpu().numpy()
        qs = {"stale": [], "isomer": [], "learned_proj": []}
        for i in range(len(TRUE)):
            er = random.Random(ERS[i])
            qs["stale"].append(hist_qerr(STALE[i], TRUE[i], er))
            er = random.Random(er.randint(0, 1 << 30))
            qs["isomer"].append(hist_qerr(iso[i], TRUE[i], er))
            er = random.Random(er.randint(0, 1 << 30))
            qs["learned_proj"].append(hist_qerr(m_proj[i], TRUE[i], er))
        out[k] = {m: gm(v) for m, v in qs.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pool-n", type=int, default=6000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--train-family", default="all")
    ap.add_argument("--buckets", type=int, default=10, help="OasisTorchV3 num_buckets (output resolution)")
    ap.add_argument("--framings", nargs="+", default=["direct", "residual"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--npz", default="real_columns.npz")
    ap.add_argument("--out", default="results_oasis_prior_real")
    a = ap.parse_args()
    dev = a.device
    global B_BUCKETS, LEVELS
    B_BUCKETS = a.buckets
    LEVELS = torch.linspace(0, 1, B_BUCKETS + 1)
    print(f"device={dev} cuda_avail={torch.cuda.is_available()} buckets={B_BUCKETS}")

    d = np.load(a.npz, allow_pickle=True)
    cols = list(zip([str(x) for x in d["names"]], [h for h in d["hists"]]))
    print(f"real test columns: {len(cols)}")
    fams = ALL_FAMILIES if a.train_family == "all" else [a.train_family]
    os.makedirs(a.out, exist_ok=True)

    pool = build_pool(fams, a.pool_n, seed=1234, device=dev)
    ref = evaluate_real(OasisTorchV3(num_buckets=B_BUCKETS).to(dev), cols, dev)  # analytic ISOMER ref
    print("\n=== REAL test (held-out range-predicate geom Q-err; lower better) ===")
    print(f"train family: {a.train_family}")
    print(f"{'method':>18} | " + "  ".join(f"K={k}" for k in (2, 6, 16)))
    print("-" * 48)
    print(f"{'stale':>18} | " + "  ".join(f"{ref[k]['stale']:.3f}" for k in (2, 6, 16)))
    print(f"{'ISOMER':>18} | " + "  ".join(f"{ref[k]['isomer']:.3f}" for k in (2, 6, 16)))

    rows = [{"method": "ISOMER", **{f"k{k}": ref[k]["isomer"] for k in (2, 6, 16)}}]
    for tag in a.framings:
        residual = (tag == "residual")
        per = {k: [] for k in (2, 6, 16)}; last = None
        for sd in a.seeds:
            model = train(pool, residual, a.epochs, a.steps, a.bs, a.lr, sd, dev)
            res = evaluate_real(model, cols, dev); last = res
            for k in (2, 6, 16):
                per[k].append((ref[k]["isomer"] - res[k]["learned_proj"]) / ref[k]["isomer"] * 100)
        fmt = lambda k: f"K{k}:{np.mean(per[k]):+.1f}%[{min(per[k]):+.1f},{max(per[k]):+.1f}]"
        print(f"{('learned:'+tag):>18} | " +
              "  ".join(f"{last[k]['learned_proj']:.3f}" for k in (2, 6, 16)) +
              "   vs ISOMER " + " ".join(fmt(k) for k in (2, 6, 16)))
        rows.append({"method": f"learned:{tag}", **{f"k{k}": last[k]["learned_proj"] for k in (2, 6, 16)},
                     **{f"k{k}_delta": float(np.mean(per[k])) for k in (2, 6, 16)}})

    with open(os.path.join(a.out, f"summary_{a.train_family}_b{B_BUCKETS}.csv"), "w", newline="") as h:
        fns = ["method", "k2", "k6", "k16", "k2_delta", "k6_delta", "k16_delta"]
        w = csv.DictWriter(h, fieldnames=fns, extrasaction="ignore"); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\nwrote {a.out}/summary_{a.train_family}.csv")


if __name__ == "__main__":
    main()
