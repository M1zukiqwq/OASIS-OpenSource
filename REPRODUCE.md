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

The numpy experiment scripts were written against an earlier JSON prior. The
trained Torch prior is injected at run time by
`oasis_torch/run_v3.py <target> [args...]`, which loads `$V3_CKPT` and
monkeypatches the model loader everywhere (`--model-path` arguments are then
ignored — no JSON file is read). Targets used by the paper:

| target | module | paper result |
|---|---|---|
| `stage1swap` | `stage1_estimator_swap_experiment.py` | single-column repair (Table `tab:real_singlecol`) |
| `comp`       | `composition_family_experiment.py`    | composition family (§ Downstream consumption) |
| `fj`         | `factorjoin_oasis_experiment.py`      | FactorJoin kernel (§ Downstream consumption) |
| `pg`         | `postgres_planner_stats_injection_experiment.py` | PostgreSQL planner (Table `table_postgres_planner_stats_injection_batch`) |
| `tpch`       | `postgres_runtime_tpch_experiment.py` | TPC-H runtime (Table `table_tpch_runtime`) |

Two checkpoints ship in `oasis_torch/artifacts/`:
- `ckpt_real_v1.pt` — prior **retrained on real columns** (Wine+Bike), used for
  the single-column real-data table (held-out Power/Forest/Census).
- `ckpt_v3_it3.pt` — the original synthetic-pretrained prior, used as the
  injected prior for the downstream-consumption experiments.

## Result → command

### Table: single-column repair on real data (`tab:real_singlecol`)
```bash
cd experiments
./run_real_local.sh            # gen drift cases + stage1swap on Power/Forest/Census, dense K=16 & sparse K=2
python make_real_singlecol_table.py
# -> results/real_singlecol_v1/table_real_singlecol.tex
```

### Table: cross-column independence repair (`tab:crosscol_feedback`) — the positive core
```bash
cd experiments
python crosscol_feedback_experiment.py     # self-contained (numpy only)
# -> results/crosscol_feedback_v1/{summary.csv, table_crosscol_feedback.tex}
```

### Table: cross-column repair on the Join Order Benchmark (`tab:crosscol_job`) — real PostgreSQL
Requires PostgreSQL (>=10) + the JOB/IMDB dataset; full setup notes in
`experiments/job_imdb/README.md`.
```bash
cd experiments/job_imdb
# load IMDB into an 'imdb' database (point at your psql + the <table>.dat files)
PSQL_BIN=/path/to/psql PGHOST=/tmp PGPORT=5432 IMDB_DAT_DIR=/path/to/imdb_dat ./load_imdb.sh
# run: PG-default vs PG-extended-stats vs AVI (true marginals) vs OASIS feedback joint
PSQL_BIN=/path/to/psql PGHOST=/tmp PGPORT=5432 python3 crosscol_pg_inject_experiment.py
# -> results_pg_inject/summary.json
```
The honest cross-column effect is the **AVI→OASIS** reduction (same true marginals, only the
joint differs); no trained prior is needed (the repair is the training-free IPF projection).

### Downstream: composition family and FactorJoin (inline numbers, § Downstream consumption)
```bash
cd experiments
V3_CKPT=oasis_torch/artifacts/ckpt_v3_it3.pt python oasis_torch/run_v3.py comp
V3_CKPT=oasis_torch/artifacts/ckpt_v3_it3.pt python oasis_torch/run_v3.py fj
```
`OASIS-noProj` in these outputs is the negative control (an unprojected
correction) — it shows the *projection*, not the prior, is what matters.

### Table: PostgreSQL planner stats injection (`table_postgres_planner_stats_injection_batch`)
Requires a local PostgreSQL 16 cluster (the script starts/uses it; set the
`--pg-*` / socket args for your install).
```bash
cd experiments
V3_CKPT=oasis_torch/artifacts/ckpt_v3_it3.pt python oasis_torch/run_v3.py pg \
  --batch --batch-drift-families left_shift right_shift --batch-rows 100000 200000 \
  --batch-seeds 20260529 20260530 20260531 --dim-rows-ratio 0.06 --min-dim-rows 5000 \
  --output-dir results/postgres_batch_v3
# -> results/postgres_batch_v3/table_postgres_planner_stats_injection_batch.tex
```

### Table: TPC-H runtime (`table_tpch_runtime`)
Requires PostgreSQL 16 + a `tpch-kit` checkout. Edit the env vars at the top of
`tpch_setup_drift.sh` (or export `PG_BIN`, `TPCH_KIT`, `PGHOST`, `PGPORT`, ...).
```bash
cd experiments
TPCH_KIT=~/tpch-kit ./tpch_setup_drift.sh                       # load SF10 + RF-style drift
V3_CKPT=oasis_torch/artifacts/ckpt_v3_it3.pt python oasis_torch/run_v3.py tpch ...   # see script --help
python make_tpch_runtime_table.py
# -> table_tpch_runtime.tex
```

## Training the learned prior (baseline control)

The learned prior is **not** a contribution of the paper (single-column
max-entropy projection is near-optimal; the prior only ties it). It survives as
a baseline column / negative control. To retrain it leakage-free on real data:

```bash
cd experiments
./gen_train_pool.sh                                                   # Wine+Bike columns only (disjoint from test sets)
python oasis_torch/data.py  --data-root data/real_cases/train_pool \
                            --out       data/real_cases/train_pool_tensor.pt
python oasis_torch/train.py --data data/real_cases/train_pool_tensor.pt \
                            --out  oasis_torch/artifacts/ckpt_real_v1.pt
# CPU-trainable in minutes; see train.py --help for the composite-objective weights.
```

`ckpt_v3_it3.pt` (the synthetic-pretrained prior) is provided as-is; its
synthetic training generators are not part of this artifact.

## Not included

Experiments cut from the final paper (synthetic single-column diagnostics,
OOD/trace studies, the sparse-PostgreSQL block, the optimizer-decision proxy)
and all archived/superseded runs are intentionally omitted to keep this artifact
scoped to the paper's claims.
