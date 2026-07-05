#!/usr/bin/env python3
"""Downstream propagation: does correcting the single-column histogram help the
estimators that CONSUME it? Compares three marginal sources fed into two standard
histogram-consuming estimators, on real data:

  修正前   (stale)  : the drifted, uncorrected marginal
  baseline (ISOMER) : max-entropy 1D projection of stale onto K feedback obs
  ours     (OASIS)  : learned B=32 prior (residual+OOD) then the SAME projection
  ref      (AVI*)   : true marginals (independence ceiling)

Downstream estimators (both use ONLY single-column histograms):
  (1) AVI 2-column joint  -> geom conjunctive-rectangle Q-error vs the TRUE joint
  (2) self-join size Sum h_i^2 -> Q-error vs true

Run on remote (imports the trained real transformer from oasis_prior_real):
  python3 oasis_downstream.py --device cuda --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import csv
import os
import random

import numpy as np
import torch

import oasis_prior_real as opr

G = opr.G


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


def avi(mx, my):
    J = np.outer(mx, my)
    return J / J.sum()


def conj_qerr(estJ, trueJ, rng, n=200, g=G, min_true=1e-3):
    qs = []
    for _ in range(n):
        i0, i1 = sorted(rng.sample(range(g + 1), 2))
        j0, j1 = sorted(rng.sample(range(g + 1), 2))
        t = float(trueJ[i0:i1, j0:j1].sum())
        if t >= min_true:
            qs.append(opr.qerr(float(estJ[i0:i1, j0:j1].sum()), t))
    return opr.gm(qs) if qs else float("nan")


def correct_marginal(model, true_m, fam, k, rng_np, rng_py, dev):
    """returns (stale, iso, oasis) marginals for one column."""
    stale = opr.drift(true_m, fam, rng_np)
    iv, tg = opr.feedback(true_m, k, rng_py)              # feedback observes TRUE marginal
    of, om = opr.obs_tensor(stale, iv, tg)
    sb = opr.hist_to_boundaries(stale)
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)
    with torch.no_grad():
        m_learned = opr.masses_from_model(model, t([of]), t([om]), t([sb]), dev)[0].cpu().numpy()
    iso = ipf_1d(stale, iv, tg)
    oasis = ipf_1d(m_learned, iv, tg)                     # SAME projection as ISOMER
    return stale, iso, oasis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pool-n", type=int, default=6000)
    ap.add_argument("--buckets", type=int, default=32)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pairs", default="real_pairs.npz")
    ap.add_argument("--out", default="results_downstream")
    a = ap.parse_args()
    dev = a.device
    opr.B_BUCKETS = a.buckets
    opr.LEVELS = torch.linspace(0, 1, a.buckets + 1)
    os.makedirs(a.out, exist_ok=True)

    d = np.load(a.pairs, allow_pickle=True)
    names = [str(x) for x in d["names"]]; joints = [J for J in d["joints"]]
    print(f"device={dev} buckets={a.buckets} pairs={len(names)}")

    pool = opr.build_pool(opr.ALL_FAMILIES, a.pool_n, seed=1234, device=dev)

    SRC = ["stale", "isomer", "oasis"]
    # accumulate per K: joint conj-qerr and self-join qerr, over seeds/pairs/trials
    agg = {k: {"avi": {s: [] for s in SRC + ["avistar"]},
               "selfjoin": {s: [] for s in SRC}} for k in (2, 6, 16)}

    for sd in a.seeds:
        model = opr.train(pool, residual=True, epochs=a.epochs, steps=a.steps,
                          bs=a.bs, lr=a.lr, seed=sd, device=dev)
        rng_np = np.random.default_rng(1000 + sd); rng_py = random.Random(1000 + sd)
        for name, P in zip(names, joints):
            true_x = P.sum(1); true_y = P.sum(0)
            for k in (2, 6, 16):
                for _ in range(a.trials):
                    fam = opr.ALL_FAMILIES[rng_np.integers(0, len(opr.ALL_FAMILIES))]
                    sx, ix, ox = correct_marginal(model, true_x, fam, k, rng_np, rng_py, dev)
                    sy, iy, oy = correct_marginal(model, true_y, fam, k, rng_np, rng_py, dev)
                    mx = {"stale": sx, "isomer": ix, "oasis": ox, "avistar": true_x}
                    my = {"stale": sy, "isomer": iy, "oasis": oy, "avistar": true_y}
                    er = random.Random(rng_py.randint(0, 1 << 30))
                    for s in SRC + ["avistar"]:
                        agg[k]["avi"][s].append(conj_qerr(avi(mx[s], my[s]), P, er))
                    # self-join size on each column (avg the two)
                    for s in SRC:
                        sj = 0.0
                        for m, tm in ((mx[s], true_x), (my[s], true_y)):
                            sj += opr.qerr(float((m ** 2).sum()), float((tm ** 2).sum()))
                        agg[k]["selfjoin"][s].append(sj / 2)

    def G_(vals):
        return opr.gm([v for v in vals if v == v])

    print("\n=== Downstream (1): AVI 2-column joint — geom conjunctive Q-err vs TRUE joint (lower better) ===")
    print(f"{'K':>3} | {'stale(修正前)':>14} {'ISOMER(baseline)':>16} {'OASIS(ours)':>12} | {'AVI*(true-marg)':>15} | ours vs ISOMER")
    rows = []
    for k in (2, 6, 16):
        gs = {s: G_(agg[k]["avi"][s]) for s in SRC + ["avistar"]}
        dvi = (gs["isomer"] - gs["oasis"]) / gs["isomer"] * 100
        print(f"{k:>3} | {gs['stale']:>14.4f} {gs['isomer']:>16.4f} {gs['oasis']:>12.4f} | {gs['avistar']:>15.4f} | {dvi:+.2f}%")
        rows.append({"metric": "avi_joint", "K": k, **gs, "oasis_vs_isomer_pct": dvi})

    print("\n=== Downstream (2): self-join size Sum h_i^2 — geom Q-err vs true (lower better) ===")
    print(f"{'K':>3} | {'stale':>10} {'ISOMER':>10} {'OASIS':>10} | ours vs ISOMER")
    for k in (2, 6, 16):
        gs = {s: G_(agg[k]["selfjoin"][s]) for s in SRC}
        dvi = (gs["isomer"] - gs["oasis"]) / gs["isomer"] * 100
        print(f"{k:>3} | {gs['stale']:>10.4f} {gs['isomer']:>10.4f} {gs['oasis']:>10.4f} | {dvi:+.2f}%")
        rows.append({"metric": "selfjoin", "K": k, **gs, "oasis_vs_isomer_pct": dvi})

    with open(os.path.join(a.out, "summary.csv"), "w", newline="") as h:
        fns = ["metric", "K", "stale", "isomer", "oasis", "avistar", "oasis_vs_isomer_pct"]
        w = csv.DictWriter(h, fieldnames=fns, extrasaction="ignore"); w.writeheader()
        for r in rows:
            w.writerow({kk: (f"{v:.4f}" if isinstance(v, float) else v) for kk, v in r.items()})
    print(f"\nwrote {a.out}/summary.csv")


if __name__ == "__main__":
    main()
