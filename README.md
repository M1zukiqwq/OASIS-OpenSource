# OASIS: Repairing Stale Optimizer Statistics in the Feedback Nullspace

Code and reproduction artifact for the paper:

> **OASIS: Repairing Stale Optimizer Statistics in the Feedback Nullspace**
> Qichu Tian and Heng Chen, Xi'an Jiaotong University.

The compiled manuscript is in [`paper/OASIS.pdf`](paper/OASIS.pdf).

## What this is

Cost-based optimizers plan with column statistics that grow stale between `ANALYZE`
refreshes; query feedback exposes the resulting selectivity errors. Repairing a histogram
from feedback is an *underdetermined inverse problem* — a few observed predicates constrain
the marginal only where they fall and leave the rest free, a region we call the **feedback
nullspace**. Classical feedback-consistency methods (STHoles, ISOMER, QuickSel-H) fill this
region with an optimizer-agnostic maximum-entropy default. **OASIS** is a statistics-layer
middleware that instead imputes the nullspace with a completion *learned against the
optimizer's own error*, then projects that completion onto the feedback and routes among
candidates from a deployment-visible signal so an imperfect prior is always safe to consume.

This repository contains the model, the drift generators and evaluation harness, and the
scripts that regenerate every table and figure in the paper.

## Repository layout

```
experiments/                 Evaluation harness, drift generators, and experiment scripts
  oasis_torch/               Stage-1 prior model + run_v3.py injection harness
    artifacts/ckpt_v3_it3.pt Trained composite-objective prior checkpoint (~1 MB)
  results/                   Per-experiment result tables the paper draws from (.tex/.json)
  REPRODUCE.md               Result-directory -> generator-script map (start here)
  RUNTIME_TPCH_PLAN.md       TPC-H runtime sanity-check protocol
  *.py, *.sh                 Experiment drivers
cdf_kll_ml_pipeline/         Shared histogram/model modules the experiments import
                             (added to sys.path automatically by the scripts)
paper/                       Compiled manuscript PDF
requirements.txt             Python dependencies (Python 3.9)
LICENSE                      MIT
```

> Large row-level intermediate outputs (`predicate_rows.*`, `decision_rows.*`, etc.) and
> archived exploratory runs are not committed (see `.gitignore`); the scripts regenerate
> them. The small per-experiment summary tables that the paper `\input`s are included.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate   # Python 3.9
pip install -r requirements.txt
```

The PostgreSQL planner-injection (paper §6.2.3) and TPC-H runtime (§6.2.4) experiments
additionally require a local PostgreSQL server reachable via the `psql` CLI; the cluster
paths/port the scripts expect are documented in `experiments/REPRODUCE.md`.

## Reproducing the results

Start from **[`experiments/REPRODUCE.md`](experiments/REPRODUCE.md)**, which gives a
**paper table/figure -> result directory -> command** map and then the full per-directory
generator commands. All v3 (composite-objective) numbers are produced through the
injection harness:

```bash
cd experiments
python oasis_torch/run_v3.py <target>     # e.g. proj | comp | fj | ood | stage1swap
                                          #      trace | public | odp | budget | noise
                                          #      pg | tpch | routerdiag | suite | smoke
```

Running a script **without** `run_v3.py` reproduces the older JSON-prior numbers and is
not what the paper reports — always go through `run_v3.py` (details in `REPRODUCE.md`).

**Verifying without a GPU or database.** Every result the paper draws from is committed —
the generated LaTeX tables (`experiments/results/<dir>/table_*.tex`) and per-experiment
summary CSV/JSON. You can confirm any number by opening the table listed in the mapping,
no re-run required. Only the planner-injection and TPC-H runtime experiments need a local
PostgreSQL 16.x server (and a TPC-H dataset); everything else is pure Python. The bulky
per-row intermediates (`predicate_rows.*`, `decision_rows.*`, ~0.2 GB) exceed GitHub's
size limits and are not committed — the scripts regenerate them.

## Full result data (release asset)

The complete per-experiment result tree — including the bulky row-level intermediates
(`predicate_rows.*`, `decision_rows.*`, `composition_family_results.json`, etc.) that
are too large to track in git — is published as a single compressed snapshot attached to
the [`v1.1.0`](https://github.com/M1zukiqwq/OASIS-OpenSource/releases/tag/v1.1.0)
release:

```bash
# download and unpack on top of the repo (restores experiments/results/ in full)
curl -L -o oasis-result-data.tar.gz \
  https://github.com/M1zukiqwq/OASIS-OpenSource/releases/download/v1.1.0/oasis-result-data.tar.gz
tar -xzf oasis-result-data.tar.gz -C experiments/
```

This lets you verify every number the paper reports against the raw outputs without
re-running any experiment. The archive excludes only the third-party NASA HTTP trace
(obtained separately, see License) and the exploratory `_archive/` runs that the paper
does not use.

## Citation

```bibtex
@unpublished{tian2026oasis,
  title  = {OASIS: Repairing Stale Optimizer Statistics in the Feedback Nullspace},
  author = {Tian, Qichu and Chen, Heng},
  note   = {Manuscript under review},
  year   = {2026}
}
```

## License

MIT — see [`LICENSE`](LICENSE). The NASA Kennedy Space Center HTTP trace used in the
public-trace case study is obtained separately from the Internet Traffic Archive and is
subject to its own terms.
