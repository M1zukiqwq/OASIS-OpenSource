# OASIS — Repairing Stale and Correlation-Blind Query-Optimizer Statistics from Query Feedback

Code, trained priors, and reproduction artifact for the paper:

> **OASIS: Repairing Stale and Correlation-Blind Query-Optimizer Statistics from Query Feedback**
> Qichu Tian and Heng Chen, Xi'an Jiaotong University.

The compiled manuscript is in [`paper/OASIS.pdf`](paper/OASIS.pdf).

## What this is

Cost-based optimizers plan with column statistics that drift stale between `ANALYZE`
refreshes and that usually omit cross-column dependence. Ordinary query execution already
exposes both errors: a completed predicate reveals the selectivity the optimizer got wrong.
**OASIS** is a statistics-layer middleware that maintains the statistics an optimizer consumes
*from that feedback alone* — no table rescan — and hands the corrected object back for the
existing planner to use unchanged.

The paper is organized as a **regime map**, not a new histogram algorithm: it says *when*
feedback repair helps and *how much*, along two axes — feedback density and dimensionality.

1. **Single column, dense feedback → the classical projection is already near-optimal.**
   The maximum-entropy feedback projection (the ISOMER/IPF step) pins the marginal where
   feedback falls; an *identifiability* result (a rank condition, `Proposition 1`) says when
   feedback pins it completely. On held-out **real** data (Power / Forest / Census) a learned
   prior and the self-tuning histograms STHoles and QuickSel-H give **no stable edge** over the
   projection. Here OASIS is intentionally ISOMER-like — the equality is the point, and
   single-column repair needs no machine learning in the deployment regime.

2. **Single column, *sparse* feedback + matched resolution → a small, located learned gain.**
   A controlled study (`§ When a Learned Prior Helps`) finds the one corner where a learned
   prior still adds value: very sparse feedback (K=2), and only once its output resolution
   reaches the evaluation grid (B=32). There it beats the projection by **+1.4%** on real
   held-out columns, shrinking to zero as feedback densifies — and the gain **propagates
   downstream** (+1–3% into AVI joins and self-joins). Reported honestly as a boundary on real
   column shapes with controlled drift, *not* an end-to-end speedup.

3. **Cross-column → feedback repairs the independence assumption. This is the positive core.**
   Per-column statistics structurally cannot encode correlation, so optimizers fall back on
   attribute-value independence (AVI) and misestimate conjunctions. Conjunctive feedback
   observes exactly the missing joint mass. A 2-D max-entropy (IPF) repair of the joint, seeded
   from the independent product and projected onto observed conjunctive masses, beats AVI by
   **+13.6% aggregate (up to ~31%)** on real correlated column pairs — the gain concentrated
   where the columns are actually correlated — and beats a faithful 2-D STHoles on every pair.
   On the Join Order Benchmark (live PostgreSQL 14) the same repair cuts conjunctive Q-error by
   **56–66%** on the genuinely correlated `title` / `aka_title` pairs.

Packaged as middleware and injected into a real PostgreSQL planner, feedback-maintained
single-column statistics cut row Q-error from **27.8 to 2.4** and lift fresh-plan match from
**56% to 96%** without a table scan.

## Layout

```
oasis-artifact/
├── README.md  REPRODUCE.md  LICENSE  requirements.txt  .gitignore
├── paper/OASIS.pdf                        # compiled manuscript
├── cdf_kll_ml_pipeline/                   # shared core library
│   ├── histogram_math.py  histogram_types.py  kll_codec.py  cdf_teacher.py
│   ├── baselines.py  modern_baselines.py  #   ISOMER / STHoles / QuickSel-H / ...
│   └── mlp_histogram_model_v2.py  tensorizer.py  ...
└── experiments/
    ├── crosscol_feedback_experiment.py         # cross-column AVI repair (positive core; numpy-only)
    ├── stage1_estimator_swap_experiment.py     # single-column repair (real-data table)
    ├── composition_family_experiment.py        # downstream: 6 composition estimators
    ├── factorjoin_oasis_experiment.py          # downstream: FactorJoin join kernel
    ├── postgres_planner_stats_injection_experiment.py   # PostgreSQL planner injection
    ├── postgres_runtime_tpch_experiment.py     # TPC-H runtime sanity check
    ├── oasis_prior_sim.py                       # §regime: learned-prior study (CPU MLP)
    ├── oasis_prior_real.py                       # §regime: on the real OasisTorchV3 transformer
    ├── oasis_downstream.py                        # §regime: downstream propagation (AVI join / self-join)
    ├── make_regime_tables.py                       # §regime: emit tab:regime + tab:downstream
    ├── make_real_singlecol_table.py  make_tpch_runtime_table.py   # LaTeX table writers
    ├── regime_data/  real_columns.npz  real_pairs.npz            # real-column snapshots (§regime inputs)
    ├── data/         download_data.sh  README.md                 # 5 UCI datasets (raw data NOT committed)
    ├── job_imdb/     crosscol_pg_inject_experiment.py  load_imdb.sh  ...   # JOB / IMDB cross-column
    ├── oasis_torch/  model.py  train.py  run_v3.py  artifacts/*.pt         # learned prior + checkpoints
    └── results/oasis_prior_real/                                  # curated §regime CSVs + generated tables
```

## Quickstart

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
( cd experiments/data && ./download_data.sh )          # 5 UCI datasets

# the positive core — no GPU, no PostgreSQL, ~1 min:
cd experiments && python crosscol_feedback_experiment.py

# the §regime tables — regenerated from the shipped curated CSVs, instantly:
python make_regime_tables.py
```

## Results → commands (summary; full detail in [REPRODUCE.md](REPRODUCE.md))

| Paper result | Command (from `experiments/`) | Needs |
|---|---|---|
| Single-column real-data table (`tab:real_singlecol`) | `./run_real_local.sh` → `python make_real_singlecol_table.py` | CPU |
| **Cross-column AVI repair (`fig:crosscol_heatmap`)** | `python crosscol_feedback_experiment.py` | CPU |
| Cross-column on JOB (`tab:crosscol_job`) | `job_imdb/` — see REPRODUCE.md | PostgreSQL + IMDB |
| §regime bucket sweep (`tab:regime`) | `python make_regime_tables.py` (curated CSVs) / `oasis_prior_real.py` to recompute | CPU / GPU |
| §regime downstream (`tab:downstream`) | `python make_regime_tables.py` / `oasis_downstream.py` to recompute | CPU / GPU |
| Composition / FactorJoin (downstream) | `oasis_torch/run_v3.py comp` / `… fj` | CPU |
| PostgreSQL planner table | `oasis_torch/run_v3.py pg --batch …` | PostgreSQL 16 |
| TPC-H runtime table | `./tpch_setup_drift.sh` → `oasis_torch/run_v3.py tpch …` | PostgreSQL 16 + tpch-kit |

## Notes

- **Environment:** Python 3.9, `numpy` / `scipy` / `torch` (see `requirements.txt`). The
  PostgreSQL experiments shell out to `psql` (no Python DB driver); the TPC-H one also needs a
  `tpch-kit` checkout. Everything else is CPU-only.
- **The learned prior** ships as (i) a baseline column in the single-column real-data table and
  (ii) the subject of the §regime study. In the **deployment regime it is dispensable** — the
  training-free projection is what OASIS deploys for single columns. Its one real, located win
  (sparse feedback + grid resolution) is documented honestly and confined to that corner.
- **Data availability:** the raw datasets (Adult/Census, Covertype/Forest, Power, Wine Quality,
  Bike Sharing; IMDB for JOB; TPC-H) are public and fetched by the scripts — not committed.
  `regime_data/*.npz` are small real-column histogram snapshots derived from those datasets.
- **Scope:** this artifact contains only the code, checkpoints, and curated data behind the
  paper's claims; archived/superseded runs are omitted. See `REPRODUCE.md`.
