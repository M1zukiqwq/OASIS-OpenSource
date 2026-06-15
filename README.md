# OASIS — Repairing Stale and Correlation-Blind Query-Optimizer Statistics from Query Feedback

Code and trained checkpoints accompanying the paper. OASIS is a **training-free
feedback-projection** system that maintains a query optimizer's statistics from
the feedback the optimizer already produces — no table re-scans.

The paper rests on a **dichotomy**, validated on real data:

1. **Single-column repair is easy.** Maximum-entropy projection of the stale
   histogram onto the feedback (the ISOMER projection) is near-optimal; a learned
   prior gives no further gain. This artifact reproduces that result and ships the
   learned prior **only as a baseline / negative control**, not as a method.
2. **The independence (AVI) assumption is the bottleneck — and feedback fixes it.**
   A 2-D max-entropy (IPF) repair of the joint, seeded from the independent
   product and projected onto observed conjunctive feedback masses, beats AVI by
   **+13.6% aggregate (up to +31.2%)** on real correlated column pairs, with the
   gain growing in the column correlation. This is the positive core.

An *identifiability* result (when feedback pins the statistic up to a free part)
explains both halves.

## Layout

```
oasis-artifact/
├── requirements.txt                 # numpy / scipy / torch  (Python 3.9)
├── REPRODUCE.md                     # every paper result → exact command
├── cdf_kll_ml_pipeline/            # shared core library (histogram math, baselines, codecs)
│   ├── histogram_math.py  histogram_types.py  kll_codec.py  cdf_teacher.py
│   ├── baselines.py  modern_baselines.py        # ISOMER / STHoles / QuickSel-H / ...
│   ├── mlp_histogram_model_v2.py  tensorizer.py
│   └── simulate_memory_kll_dataset.py  json_histogram_parser.py  extended_drift_generators.py
└── experiments/
    ├── crosscol_feedback_experiment.py          # cross-column AVI repair (positive core; self-contained)
    ├── stage1_estimator_swap_experiment.py      # single-column repair (via run_v3 stage1swap)
    ├── composition_family_experiment.py         # downstream: composition (via run_v3 comp)
    ├── factorjoin_oasis_experiment.py           # downstream: FactorJoin   (via run_v3 fj)
    ├── postgres_planner_stats_injection_experiment.py   # PostgreSQL planner table (run_v3 pg)
    ├── postgres_runtime_tpch_experiment.py      # TPC-H runtime table (run_v3 tpch)
    ├── copula_oasis_experiment.py  optimizer_decision_proxy_experiment.py  oasis_accuracy_smoke.py
    ├── make_real_dataset_cases.py               # real-data sliding-window drift case generator
    ├── make_real_singlecol_table.py  make_tpch_runtime_table.py    # LaTeX table writers
    ├── gen_train_pool.sh  run_real_local.sh  tpch_setup_drift.sh   # drivers
    ├── data/                                     # download_data.sh + dataset docs (raw data NOT committed)
    └── oasis_torch/                              # learned-prior TRAINING + inference
        ├── model.py  train.py  data.py          # architecture / training / tensorization
        ├── v3_infer.py  run_v3.py               # inference adapter + the prior-injection runner
        ├── gate_eval.py  mechanism_analysis.py  gen_ood_samples.py  DESIGN.md
        └── artifacts/  ckpt_real_v1.pt  ckpt_v3_it3.pt   # trained checkpoints (tracked)
```

## Quickstart

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
( cd experiments/data && ./download_data.sh )        # 5 UCI datasets

# the positive core — no GPU, no PostgreSQL, ~1 min:
cd experiments && python crosscol_feedback_experiment.py
```

## Results → commands (summary; full detail in REPRODUCE.md)

| Paper result | Command (from `experiments/`) | Needs |
|---|---|---|
| Single-column real-data table | `./run_real_local.sh` → `python make_real_singlecol_table.py` | CPU |
| **Cross-column AVI repair table** | `python crosscol_feedback_experiment.py` | CPU |
| Composition / FactorJoin (downstream) | `… run_v3.py comp` / `… run_v3.py fj` | CPU |
| PostgreSQL planner table | `… run_v3.py pg --batch …` | PostgreSQL 16 |
| TPC-H runtime table | `./tpch_setup_drift.sh` → `… run_v3.py tpch …` | PostgreSQL 16 + tpch-kit |
| Retrain the prior (baseline) | `./gen_train_pool.sh` → `data.py` → `train.py` | CPU |

The trained Torch prior is injected into the numpy experiments at run time via
`oasis_torch/run_v3.py <target>` (it monkeypatches the model loader; see
REPRODUCE.md). The cross-column experiment is pure numpy and needs no prior.

## Notes

- **Environment:** Python 3.9.6, numpy 2.0.2 / scipy 1.13.1 / torch 2.8.0. The
  PostgreSQL experiments shell out to `psql` (no Python DB driver); the TPC-H one
  also needs a `tpch-kit` checkout. Everything else is CPU-only.
- **Scope:** this artifact contains only the code and checkpoints behind the
  paper's claims. Cut/archived experiments are not included (see REPRODUCE.md).
