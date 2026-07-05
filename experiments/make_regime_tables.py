#!/usr/bin/env python3
"""Emit the two §regime LaTeX tables (tab:regime, tab:downstream) straight from the
oasis_prior_real result CSVs -- so the paper's hand-written numbers become reproducible
\\input files instead of transcribed constants.

  tab:regime      <- results/oasis_prior_real/summary_all_b{10,16,24,32}.csv
                     (learned residual prior vs ISOMER as output resolution B grows;
                      the reported cell is the learned prior's % reduction over ISOMER)
  tab:downstream  <- results/oasis_prior_real/downstream_summary.csv
                     (two histogram consumers fed stale/ISOMER/OASIS/AVI* marginals)

No recompute: numbers are read directly from the CSVs the remote runs produced.
"""
import csv
import os

ROOT = "results"
PR = os.path.join(ROOT, "oasis_prior_real")

# tab:regime : bucket -> row label
BUCKETS = [(10, r"$10$ (model default)"), (16, r"$16$"),
           (24, r"$24$"), (32, r"$32$ (grid)")]


def pct(v):
    return f"{float(v):+.1f}\\%"


def emit_regime():
    # gather (bucket, k2_delta, k6_delta, k16_delta) from each summary_all_b*.csv
    rows = []
    for b, _ in BUCKETS:
        p = os.path.join(PR, f"summary_all_b{b}.csv")
        rec = {r["method"]: r for r in csv.DictReader(open(p))}
        d = rec["learned:residual"]
        rows.append((b, float(d["k2_delta"]), float(d["k6_delta"]), float(d["k16_delta"])))
    best = max(v for _, k2, k6, k16 in rows for v in (k2, k6, k16))  # bold the single best cell

    def cell(v):
        s = pct(v)
        return f"$\\mathbf{{{s}}}$" if v == best else f"${s}$"

    lines = [
        r"\begin{table}[t]", r"  \centering\small",
        r"  \caption{Learned prior versus the ISOMER projection on real held-out columns as output",
        r"  resolution $B$ grows (residual prior, three seeds; geometric-mean Q-error on held-out range",
        r"  predicates; a positive number means the learned prior beats the projection by that fraction).",
        r"  The edge appears only once $B$ reaches the grid, and only at sparse feedback.}",
        r"  \label{tab:regime}", r"  \setlength{\tabcolsep}{10pt}",
        r"  \begin{tabular}{c r r r}", r"    \toprule",
        r"    Output buckets $B$ & $K{=}2$ & $K{=}6$ & $K{=}16$ \\", r"    \midrule",
    ]
    for b, label in BUCKETS:
        _, k2, k6, k16 = next(r for r in rows if r[0] == b)
        lines.append(f"    {label} & {cell(k2)} & {cell(k6)} & {cell(k16)} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return lines


def emit_downstream():
    R = list(csv.DictReader(open(os.path.join(PR, "downstream_summary.csv"))))
    by = {(r["metric"], int(r["K"])): r for r in R}

    def num(v):
        v = (v or "").strip()
        return f"${float(v):.3f}$" if v else "---"

    def body(metric, header):
        out = [f"    \\multirow{{3}}{{*}}{{{header}}}"]
        for i, k in enumerate((2, 6, 16)):
            r = by[(metric, k)]
            lead = out[-1] if i == 0 else "   "
            kcell = f"${k}$".rjust(4)
            row = (f" & {kcell}  & {num(r['stale'])} & {num(r['isomer'])} & "
                   f"{num(r['oasis'])} & {num(r.get('avistar'))} & ${pct(r['oasis_vs_isomer_pct'])}$ \\\\")
            if i == 0:
                out[-1] = lead + row
            else:
                out.append(lead + row)
        return out

    lines = [
        r"\begin{table}[t]", r"  \centering\small",
        r"  \caption{Downstream propagation. Geometric-mean Q-error of two histogram consumers when fed",
        r"  the stale, ISOMER-corrected, and learned-prior-corrected ($B{=}32$) marginals on real columns",
        r"  (lower is better). ``ours vs.\ ISOMER'' is the learned prior's reduction over the projection;",
        r"  AVI$^\ast$ uses the true marginals and is the independence ceiling.}",
        r"  \label{tab:downstream}", r"  \setlength{\tabcolsep}{6pt}",
        r"  \begin{tabular}{l c r r r r r}", r"    \toprule",
        r"    Downstream consumer & $K$ & stale & ISOMER & OASIS & AVI$^\ast$ & ours vs.\ ISOMER \\",
        r"    \midrule",
    ]
    lines += body("avi_joint", r"AVI two-column join")
    lines.append(r"    \addlinespace[2pt]")
    lines += body("selfjoin", r"Self-join $\sum_i h_i^2$")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return lines


def main():
    for name, gen in (("table_regime.tex", emit_regime),
                      ("table_downstream.tex", emit_downstream)):
        lines = gen()
        with open(os.path.join(PR, name), "w") as h:
            h.write("\n".join(lines) + "\n")
        print(f"wrote {PR}/{name}")
        print("\n".join(lines))
        print()


if __name__ == "__main__":
    main()
