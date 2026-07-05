# Reproducing the OASIS results

Every result in the paper maps to one command below. Unless noted, run from the
`experiments/` directory with the project venv active.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
( cd experiments/data && ./download_data.sh )      # fetch the 5 UCI datasets
```

By default the scripts call `python3`; to use a specific interpreter, export
`PY=/path/to/python` before running a `.sh` driver.

## The v3 prior-injection mechanism

Several numpy experiment scripts were written against an earlier JSON prior. The
trained Torch prior is injected at run time by
`oasis_torch/run_v3.py <target> [args...]`, which loads `$V3_CKPT` and
monkeypatches the model loader everywhere. Targets used by the paper:

| target | module | paper result |
|---|---|---|
| `stage1swap` | `stage1_estimator_swap_experiment.py` | single-column repair (`tab:real_singlecol`) |
| `comp`       | `composition_family_experiment.py`    | composition family (§ Downstream consumption) |
| `fj`         | `factorjoin_oasis_experiment.py`      | FactorJoin kernel (§ Downstream consumption) |
| `pg`         | `postgres_planner_stats_injection_experiment.py` | PostgreSQL planner (`tab:postgres_planner_stats_injection_batch`) |
| `tpch`       | `postgres_runtime_tpch_experiment.py` | TPC-H runtime (`tab:tpch_runtime`) |

Two checkpoints ship in `oasis_torch/artifacts/`:
- `ckpt_real_v1.pt` — prior **retrained on real columns** (Wine+Bike), used for
  the single-column real-data table (held-out Power/Forest/Census).
- `ckpt_v3_it3.pt` — the synthetic-pretrained prior, used as the injected prior
  for the downstream-consumption experiments.

## Result → command

### `tab:real_singlecol` — single-column repair on real data
```bash
cd experiments
./run_real_local.sh            # gen drift cases + stage1swap on Power/Forest/Census, dense K=16 & sparse K=2
python make_real_singlecol_table.py
# -> results/real_singlecol_v1/table_real_singlecol.tex
```
The message is a **near-tie**: once feedback is projected, STHoles / QuickSel-H / the
learned prior give no stable edge over the ISOMER projection. In the deployment regime
single-column repair needs no ML.

### `fig:crosscol_heatmap` — cross-column independence repair (the positive core)
```bash
cd experiments
python crosscol_feedback_experiment.py     # self-contained (numpy only)
# -> results/crosscol_feedback_v1/{summary.csv, ...}
```
AVI (true marginals) vs a faithful 2-D STHoles vs the OASIS feedback joint (2-D
max-entropy/IPF seeded from the marginals). OASIS is lowest on every pair; **+13.6%
aggregate, up to ~31%**, concentrated where the columns are correlated.

### `tab:crosscol_job` — cross-column repair on the Join Order Benchmark (real PostgreSQL)
Requires PostgreSQL (>=10) + the JOB/IMDB dataset; full setup in `experiments/job_imdb/README.md`.
```bash
cd experiments/job_imdb
PSQL_BIN=/path/to/psql PGHOST=/tmp PGPORT=5432 IMDB_DAT_DIR=/path/to/imdb_dat ./load_imdb.sh
PSQL_BIN=/path/to/psql PGHOST=/tmp PGPORT=5432 python3 crosscol_pg_inject_experiment.py
# -> results/pg_inject_summary.json   (the shipped curated result)
```
The honest cross-column effect is the **AVI→OASIS** reduction (same true marginals, only the
joint differs); no trained prior is needed (the repair is the training-free IPF projection).

### `tab:regime` and `tab:downstream` — the §regime study (When a Learned Prior Helps)

This is a **controlled study on real column shapes with synthetic drift**, not the
end-to-end PostgreSQL pipeline. The learned prior is trained only on synthetic drifted
columns and tested on the shapes of real held-out columns; it is corrected through the
*same* feedback projection as ISOMER, so any difference is the prior, not the projection.

The two tables regenerate **instantly from the shipped curated CSVs** — no GPU needed:
```bash
cd experiments
python make_regime_tables.py
# reads results/oasis_prior_real/{summary_all_b*.csv, downstream_summary.csv}
# -> results/oasis_prior_real/{table_regime.tex, table_downstream.tex}
```

To **recompute the CSVs from scratch** (needs the real OasisTorchV3 transformer; GPU
recommended, CPU works but is slow). The regime scripts import `model.py` from
`oasis_torch/`, so put it on `PYTHONPATH`:

```bash
cd experiments

# tab:regime — output-resolution sweep (residual prior, 3 seeds) on real held-out columns:
for B in 10 16 24 32; do
  PYTHONPATH=oasis_torch python3 oasis_prior_real.py --device cuda --seeds 0 1 2 \
      --buckets $B --framings residual \
      --npz regime_data/real_columns.npz --out results/oasis_prior_real
  # -> results/oasis_prior_real/summary_all_b$B.csv
done

# tab:downstream — feed stale/ISOMER/OASIS(B=32) marginals into AVI join + self-join:
PYTHONPATH=oasis_torch python3 oasis_downstream.py --device cuda --seeds 0 1 2 \
    --buckets 32 --pairs regime_data/real_pairs.npz --out results/oasis_prior_real
# writes results/oasis_prior_real/summary.csv; the curated copy read by the table
# generator is downstream_summary.csv:
cp results/oasis_prior_real/summary.csv results/oasis_prior_real/downstream_summary.csv

python3 make_regime_tables.py
```

`regime_data/real_columns.npz` and `real_pairs.npz` are real-column histogram snapshots
derived from the same public datasets; the CPU MLP that first isolated the finding is
`oasis_prior_sim.py` (`python3 oasis_prior_sim.py`, needs `download_data.sh` first).

### `tab:postgres_planner_stats_injection_batch` — PostgreSQL planner stats injection
Requires a local PostgreSQL 16 cluster (set the `--pg-*` / socket args for your install).
```bash
cd experiments
V3_CKPT=oasis_torch/artifacts/ckpt_v3_it3.pt python oasis_torch/run_v3.py pg \
  --batch --batch-drift-families left_shift right_shift --batch-rows 100000 200000 \
  --batch-seeds 20260529 20260530 20260531 --dim-rows-ratio 0.06 --min-dim-rows 5000 \
  --output-dir results/postgres_batch_v3
# -> results/postgres_batch_v3/table_postgres_planner_stats_injection_batch.tex
```
Row Q-error 27.8 → 2.4, fresh-plan match 56% → 96%. OASIS ≈ ISOMER here (dense feedback pins
the marginal), and the Router selects ISOMER on all 12 configs — the point is plan-shape
safety, not a single-column win.

### `tab:tpch_runtime` — TPC-H runtime sanity check
Requires PostgreSQL 16 + a `tpch-kit` checkout. Edit the env vars at the top of
`tpch_setup_drift.sh` (or export `PG_BIN`, `TPCH_KIT`, `PGHOST`, `PGPORT`, ...).
```bash
cd experiments
TPCH_KIT=~/tpch-kit ./tpch_setup_drift.sh
V3_CKPT=oasis_torch/artifacts/ckpt_v3_it3.pt python oasis_torch/run_v3.py tpch ...   # see script --help
python make_tpch_runtime_table.py
```
Calibrated statistics reproduce **fresh**-statistics behavior (accuracy, plan shape, runtime);
this is a single-schema check, not a runtime-superiority claim.

## Training the learned prior

The learned prior is **dispensable in the deployment regime** (single-column max-entropy
projection is near-optimal); it ships as a baseline column and as the subject of the §regime
study, where — at sparse feedback and grid-matched resolution — it earns a small, located
gain. To retrain it leakage-free on real data:

```bash
cd experiments
./gen_train_pool.sh                                                   # Wine+Bike columns only (disjoint from test sets)
python oasis_torch/data.py  --data-root data/real_cases/train_pool \
                            --out       data/real_cases/train_pool_tensor.pt
python oasis_torch/train.py --data data/real_cases/train_pool_tensor.pt \
                            --out  oasis_torch/artifacts/ckpt_real_v1.pt
# CPU-trainable in minutes; see train.py --help for the composite-objective weights.
```

## Not included

Archived/superseded runs and experiments cut from the final paper are intentionally omitted
to keep this artifact scoped to the paper's claims. Generated `results/` (except the curated
`results/oasis_prior_real/`) are git-ignored and regenerated by the commands above.
