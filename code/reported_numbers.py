"""Results summary: print the main estimates from the shipped aggregated results.

Needs only pandas and the CSV tables in results_aggregated/ (no IEA microdata). After a full run of
the pipeline, `--results` reads the same tables from results/ instead.

Run (from the repository root):
    python code/reported_numbers.py
    python code/reported_numbers.py --results
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / ("results" if "--results" in sys.argv else "results_aggregated")


def load(name, **kw):
    p = SRC / name
    if not p.exists():
        print(f"  [missing: {name}]")
        return None
    return pd.read_csv(p, **kw)


def ci(r, d=2):
    return f"{r.est:.{d}f} [{r.ci_lo:.{d}f}, {r.ci_hi:.{d}f}]"


def section(title):
    print()
    print(title)
    print("-" * len(title))


print(f"Results summary, read from {SRC.relative_to(ROOT)}/")

section("Sample and validation")
prep = load("prep_summary.csv")
if prep is not None:
    print(f"  education systems: {len(prep)}")
    print(f"  students: {prep.n_students.sum():,}   schools: {prep.n_schools.sum():,}")
pvc = load("published_vs_computed.csv", encoding="utf-8-sig")
if pvc is not None:
    m = pvc.dropna(subset=["published_mean"])
    print(f"  computed vs published CIL means ({len(m)} reported systems): "
          f"largest |difference| {m.diff_mean.abs().max():.1f} points, "
          f"SE ratio {m.se_ratio.min():.2f} to {m.se_ratio.max():.2f}")

section("Prediction (pooled out-of-sample R^2, 34 systems, 95% CI)")
dr = load("meta_delta_r2.csv")
if dr is not None:
    dr = dr[dr.analysis == "main"].set_index("quantity")
    for q, lab in [("R2_RIDGE", "weighted ridge"), ("R2_RLMM", "random-intercept model"),
                   ("R2_XGB", "XGBoost"), ("R2_MERF", "mixed-effects XGBoost")]:
        print(f"  {lab:<32s} {ci(dr.loc[q], 3)}")
    print(f"  {'XGBoost minus weighted ridge':<32s} {ci(dr.loc['dR2_XGB-RIDGE'], 3)}")

section("Opportunities to learn, design-based linear estimates (points per SD, 95% CI)")
oc = load("meta_otl_composites.csv")
if oc is not None:
    oc = oc.set_index(["feature", "step"])
    steps = [("a", "unadjusted"), ("b", "+ background and peers"), ("c", "+ ICT use"),
             ("d", "+ dispositions")]
    for f, lab in [("OTLO6", "outside school (six items)"), ("OTLS6", "at school (six items)")]:
        print(f"  {lab}")
        for s, slab in steps:
            print(f"    {slab:<26s} {ci(oc.loc[(f, s)])}")

section("Focal SHAP slopes, Holm adjustment over the 8 primary endpoints")
rx = load("reporting_extras.csv")
ms = load("meta_SLOPE.csv")
if rx is not None and ms is not None:
    p = rx[rx.estimator == "shap_slope"].set_index("feature").p_exact
    order = p.sort_values().index
    holm, run = {}, 0.0
    for rank, f in enumerate(order):
        run = max(run, min(1.0, (len(p) - rank) * p[f]))
        holm[f] = run
    ms = ms[ms.analysis == "main"].set_index("feature")
    for f in p.index:
        flag = "  significant after Holm" if holm[f] < 0.05 else ""
        print(f"  {f:<11s} slope {ci(ms.loc[f])}  Holm p = {holm[f]:.3g}{flag}")

section("Coverage simulation (ESP pseudo-population, schools drawn with replacement)")
cs = load("coverage_sim_summary.csv")
if cs is not None:
    r = cs[(cs.design == "wr") & (cs.stat == "SLOPE") & (cs.fpc == "none")].iloc[0]
    print("  SHAP slopes, median ratio of the estimated standard error to the true one:")
    print(f"    jackknife (JRR)   {r.median_ratio_jrr:.2f}")
    print(f"    school bootstrap  {r.median_ratio_boot:.2f}")
