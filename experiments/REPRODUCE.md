# Reproducing the OASIS (Information Systems) results

This maps every `experiments/results/` directory the paper depends on to the
code that regenerates it. Archived/superseded runs live under
`experiments/_archive/` and are not part of the submission (see its README).

## Paper table/figure -> result directory -> command

Each paper artifact (identified by its caption and LaTeX label) is regenerated as below.
The committed `results/<dir>/table_*.tex` files are exactly the numbers the paper
`\input`s, so you can also just open them to verify **without running anything**.

| Paper item (caption) | LaTeX label | `results/<dir>` | Command | Needs |
|---|---|---|---|---|
| Degrees of freedom / rank | `tab:mr_rank` | `mechanism_rank_v3` | `python oasis_torch/mechanism_analysis.py` | Python |
| Stage-1 objective ablation | `tab:mr_ablation` | `ablation_objective_v3` | Stage-1 ablation (oasis_torch train/eval) | Python+GPU |
| Projection-init vs ISOMER | `tab:projection_initialization` | `proj_v3` | `oasis_torch/run_v3.py proj` | Python |
| Projection-init vs classical priors | `tab:projection_init_classical` | `proj_v3` | `oasis_torch/run_v3.py proj` | Python |
| Feedback locality (fig) | `fig:feedback_locality` | `proj_v3` | `oasis_torch/run_v3.py proj` | Python |
| Stage-1 prior-source swap | `tab:stage1_estimator_swap` | `stage1_estimator_swap_v3` | `run_v3_supporting.sh` (`stage1swap`) | Python |
| OOD drift realism | `tab:ood_drift_realism` | `ood_drift_realism_v3` | `run_v3_supporting.sh` (`ood`) | Python |
| Public NASA trace | `tab:public_trace_workload` | `public_trace_workload_v3` | `run_v3_supporting.sh` (`public`) | Python + NASA trace |
| DML-trace heatmap (fig) | `fig:trace_grounded_drift` | `trace_grounded_drift_v3` | `run_v3_supporting.sh` (`trace`) | Python |
| Composition family (fig) | `fig:composition_family` | `comp_v3` | `oasis_torch/run_v3.py comp` | Python |
| FactorJoin kernel (fig) | `fig:factorjoin` | `fj_v3` | `oasis_torch/run_v3.py fj` | Python |
| Feedback-budget sweep (fig) | `fig:feedback_budget` | `feedback_budget_sensitivity_v3` | `run_v3_supporting2.sh` (`budget`) | Python |
| Optimizer-decision proxy | `tab:optimizer_decision_proxy` | `optimizer_decision_proxy_v3` | `run_v3_supporting.sh` (`odp`) | Python |
| PostgreSQL planner injection | `tab:postgres_planner_stats_injection_batch` | `postgres_batch_v3` | `oasis_torch/run_v3.py pg --batch ...` | **PostgreSQL** |
| PostgreSQL sparse sweep | `tab:postgres_sparse_multiconfig` | `exp2_sparse_multiconfig_v3_20260602` | `./run_sparse_multiconfig_v3.sh` | **PostgreSQL** |
| TPC-H runtime check | `tab:tpch_runtime` | `postgres_runtime_tpch_multiseed_20260601` | `tpch_setup_drift.sh` then `oasis_torch/run_v3.py tpch` | **PostgreSQL + TPC-H** |

The architecture figure (`fig:architecture`) is a hand-drawn vector graphic
(`paper/figures/oasis_architecture.svg`) and the §2.3 schematic is TikZ in the manuscript;
neither is script-generated. The exact `run_v3.py pg` / `run_sparse_multiconfig` arguments
appear in the "Result directory -> generator" table further below.

## Environment

- Python venv: `../.venv_v3` (**Python 3.9**; `numpy==2.0.2`, `scipy==1.13.1`,
  `torch==2.8.0` — see `../requirements.txt`). All v3 runs use it.
- Trained v3 prior checkpoint: `oasis_torch/artifacts/ckpt_v3_it3.pt`
  (exported via `V3_CKPT=...`).
- Local PostgreSQL 16.9 for the planner/runtime injection experiments:
  binaries `/Volumes/QUQ/pg/pgsql/bin`, data `/Volumes/QUQ/pg/data`,
  socket `/Volumes/QUQ/pg/run`, port `55432` (the experiment scripts'
  defaults; the script starts/uses the cluster automatically).
- Shared inputs (do not delete):
  - `results/synthetic_paper_suite_rerun_20260529/compound_data` — cached
    held-out drift cases used by the proxy / projection / swap experiments.
  - `results/synthetic_paper_suite_rerun_20260529/models/oasis_k16.json` and
    `results/copula_model/oasis_k16.json` — model paths the scripts default to
    (under the v3 runner the trained torch prior is patched in regardless).

## What needs PostgreSQL, and verifying without re-running

**Pure-Python (no database).** Everything except the three rows marked *PostgreSQL* in the
table above runs from the venv and the bundled checkpoint alone — the projection /
locality / budget / noise / composition / FactorJoin / OOD / trace / estimator-swap /
proxy experiments. These read the cached drift cases under
`results/synthetic_paper_suite_rerun_20260529/compound_data` (committed).

**Requires PostgreSQL 16.x.** Only the planner-injection (`postgres_batch_v3`,
`exp2_sparse_multiconfig_v3_20260602`) and TPC-H runtime (`postgres_runtime_tpch_*`)
experiments need a local PostgreSQL server; the TPC-H one additionally needs a TPC-H
dataset built via `tpch_setup_drift.sh`. The scripts expect the cluster paths/port in the
Environment section above; point them at your own cluster if different.

**Verifying without running anything.** Every result directory the paper draws from is
committed, including the generated LaTeX tables (`results/<dir>/table_*.tex`) and the
per-experiment summary CSV/JSON. To check a paper number, open the corresponding
`table_*.tex` (or `summary*.json`) listed in the mapping above — it is byte-for-byte what
the manuscript `\input`s. This lets a reviewer confirm every table/figure **without a GPU
or a database**. The only artifacts *not* committed are the bulky per-row intermediates
(`predicate_rows.*`, `decision_rows.*`, etc., ~0.2 GB and, with archived runs, several GB),
which exceed GitHub's file-size limits; the scripts regenerate them, and they are
available from the authors on request.

## The v3 injection mechanism

The numpy experiment scripts were written against an earlier JSON prior. The
**v3 composite-objective** prior is injected at run time by
`oasis_torch/run_v3.py <target> [args...]`, which loads `$V3_CKPT` and
monkeypatches `oasis_boundaries` / `correct_marginal_with_oasis` everywhere.
Targets (`MAP` in `run_v3.py`): `proj comp fj ood stage1swap trace public odp
budget noise pg tpch routerdiag suite smoke`.

> Running a script **without** `run_v3.py` reproduces the *old* JSON-prior
> numbers (this is what produced the superseded, contradictory
> `exp2_sparse_sweep_20260601` — now archived). Always go through `run_v3.py`.

## Result directory → generator

| Result dir | How to regenerate |
|---|---|
| `proj_v3` | `run_v3.py proj` (projection-init / locality table + figure) |
| `comp_v3` | `run_v3.py comp` (composition family) |
| `fj_v3` | `run_v3.py fj` (FactorJoin kernel) |
| `ood_drift_realism_v3` | `run_v3_supporting.sh` (target `ood`) |
| `trace_grounded_drift_v3` | `run_v3_supporting.sh` (target `trace`) |
| `public_trace_workload_v3` | `run_v3_supporting.sh` (target `public`) |
| `stage1_estimator_swap_v3` | `run_v3_supporting.sh` (target `stage1swap`) |
| `optimizer_decision_proxy_v3` | `run_v3_supporting.sh` (target `odp`) |
| `feedback_budget_sensitivity_v3` | `run_v3_supporting2.sh` (target `budget`) |
| `feedback_noise_robustness_v3` | `run_v3_supporting2.sh` (target `noise`) |
| `postgres_batch_v3` | `run_v3.py pg --batch --batch-drift-families left_shift right_shift --batch-rows 100000 200000 --batch-seeds 20260529 20260530 20260531 --dim-rows-ratio 0.06 --min-dim-rows 5000 --output-dir results/postgres_batch_v3` |
| `exp2_sparse_v3_20260601` | single-config sparse sweep: `run_v3.py pg --drift-family left_shift --rows 40000 --dim-rows 4000 --seed 20260529 --num-feedback {2,4,6,8,16} --output-dir results/exp2_sparse_v3_20260601/k{K}` |
| `postgres_runtime_tpch_multiseed_20260601` | `tpch_setup_drift.sh` then `run_v3.py tpch` over three refresh-stream seeds (per-seed intermediates archived) |
| `mechanism_rank_v3` | `oasis_torch/mechanism_analysis.py` (rank/free-DOF sweep behind Table `tab:mr_rank`) |
| `ablation_objective_v3` | Stage-1 objective ablation behind Table `tab:mr_ablation` (oasis_torch training/eval) |
| `synthetic_paper_suite_rerun_20260529` | `run_synthetic_paper_suite.py` — shared data + base model; **input, keep** |
| `synthetic_paper_suite_tree_isomer_v3` | base tables for the conference `main.tex` |

> For targets driven by `run_v3_supporting*.sh`, the exact `--data-root`,
> `--model-path`, `--q-values`, `--max-cases-per-q`, and `--seed` are recorded
> in those scripts. Targets without a recorded command (`proj/comp/fj`) use the
> same shared `DATA`/`MODEL` and each script's defaults.

## Major-revision additions (2026-06-02)

| Result dir | Generator | Addresses |
|---|---|---|
| `exp2_sparse_multiconfig_v3_20260602` | `./run_sparse_multiconfig_v3.sh` — the 12-config grid (left/right × 100K/200K × 3 seeds) swept over K∈{2,4,6,8,16}. K=16 reproduces `postgres_batch_v3`. | MR2: multi-config sparse evidence |
| `router_diagnostics_v3_20260602` | `run_v3.py routerdiag --data-root .../compound_data --model-path .../oasis_k16.json --q-values 5 10 15 20 25 30 --max-cases-per-q 128 --seed 42 --output-dir results/router_diagnostics_v3_20260602` (script `router_diagnostics.py`) | MR3: Router vs oracle/random/always-X, residual tie rate |
| noise Router column | `run_v3_supporting2.sh` target `noise` (the table writer now emits the `calibrated_hybrid`/Router column) | MR4: noise-table / prose consistency |
