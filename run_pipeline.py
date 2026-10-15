"""Run the analysis pipeline stage by stage, in order.

Every stage runs scripts in code/ with the current Python interpreter and writes its result tables
to results/. Per-system steps are cached in results/<step>/, so an interrupted stage resumes with the
unfinished systems.

    python run_pipeline.py                 all stages
    python run_pipeline.py --list          print the stages without running them
    python run_pipeline.py --from 12       stages 12 to the end
    python run_pipeline.py --only 8 10     selected stages
    python run_pipeline.py --npar 4        education systems in parallel (default 8)
    python run_pipeline.py --dry-run       print the commands without running them
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"


def per_system(script, npar, extra=()):
    """runner.py <script> <npar>, then <script> --collect and any extra calls."""
    return [["runner.py", script, str(npar)], [script, "--collect"]] + [[script, *e] for e in extra]


def stages(npar):
    return [
        (1, "prepare", "data/processed/analysis.parquet, prep_summary.csv",
         [["01_prepare.py"]]),
        (2, "validate", "validation_country_means.csv, published_cil_means.csv, published_vs_computed.csv",
         [["02_validate.py"], ["-I", "validation/build_published.py", ".."]]),
        (3, "cross-validation", "cv_metrics.csv, oof_predictions.parquet, cv_hparams.json",
         per_system("03_cv.py", npar)),
        (4, "design-based SHAP with JRR", "shap_country_stats.parquet, final_hparams.json",
         per_system("04_shap_jrr.py", npar)),
        (5, "naive contrasts", "robust_feature_stats.parquet, leakage_r2.csv",
         per_system("04b_robust.py", npar)),
        (6, "drop-out sensitivity", "dropone.csv",
         per_system("04c_dropone.py", npar)),
        (7, "school bootstrap", "clusterboot_stats.parquet, clusterboot_blocks.parquet",
         per_system("04d_cluster_boot.py", npar)),
        (8, "meta-analysis", "meta_SLOPE.csv, meta_IMP.csv, meta_BLOCK.csv",
         [["05_meta.py"]]),
        (9, "transfer (LOCO)", "loco_metrics.csv, loco_hparams.json",
         [["06_loco.py"]]),
        (10, "fairness and calibration", "fairness_gaps.csv",
         [["07_fairness.py"]]),
        (11, "sanity checks", "sanity_*.csv, local_explanations.parquet",
         per_system("04e_sanity_local.py", npar, extra=[["--local"]])),
        (12, "linear benchmarks", "meta_ols_full.csv, meta_focal_*.csv, meta_otl_composites.csv",
         [["09_additional_analyses.py"], ["10_linear_benchmarks.py"], ["10b_bivariate_correlations.py"]]),
        (13, "noise floor and drop-out importance", "noise_floor_*.csv, crossfit_shap_stats.csv, meta_dropout.csv",
         per_system("11_noise_floor.py", 6, extra=[["--local-oof"]]) + per_system("11b_dropout_pv.py", 6)),
        (14, "performance, calibration and fairness contrasts",
         "meta_delta_r2.csv, meta_fairness_extended.csv, gap_contrasts_pooled.csv, rmse_design.csv",
         [["12_performance_contrasts.py"]]
         + [["12b_delta_r2_foldseeds.py", "--run", str(pv), "6"] for pv in range(1, 6)]
         + [["12b_delta_r2_foldseeds.py", "--collect"], ["12c_gap_contrasts.py"], ["12d_rmse_design.py"]]),
        (15, "coverage simulation", "coverage_sim.csv, coverage_sim_summary.csv",
         [["13_coverage_runner.py", str(npar)], ["13_coverage_sim.py", "--collect"]]),
        (16, "tuning propagation", "tuning_propagation.csv",
         [["14_tuning_propagation.py", "--run", "5", "3"], ["14_tuning_propagation.py", "--collect"]]),
        (17, "nested transfer", "loco_nested.csv, loco_nested_meta.json",
         ["nested_target", "nested_workers", ["15_loco_nested.py", "--collect"]]),
        (18, "JRR mechanism", "jrr_mechanism.csv, jrr_mechanism_summary.csv",
         [["16_jrr_mechanism.py", "--run"], ["16_jrr_mechanism.py", "--collect"]]),
    ]


def run(cmd, dry):
    if cmd == "nested_target":
        # the nested transfer analysis covers the first 12 systems of the seeded order of 15_loco_nested.py
        print("  results/loco_nested/N_TARGET.txt = 12")
        if not dry:
            (ROOT / "results" / "loco_nested").mkdir(parents=True, exist_ok=True)
            (ROOT / "results" / "loco_nested" / "N_TARGET.txt").write_text("12\n")
        return
    if cmd == "nested_workers":
        argv = [sys.executable, "15_loco_nested.py", "--worker", "8", "cpu"]
        print("  2 x python", " ".join(argv[1:]))
        if not dry:
            procs = [subprocess.Popen(argv, cwd=CODE) for _ in range(2)]
            if any(p.wait() != 0 for p in procs):
                sys.exit("a nested-LOCO worker failed")
        return
    print("  python", " ".join(cmd))
    if not dry:
        r = subprocess.run([sys.executable, *cmd], cwd=CODE)
        if r.returncode != 0:
            sys.exit(f"failed: python {' '.join(cmd)} (exit {r.returncode})")


def main():
    ap = argparse.ArgumentParser(description="Run the analysis pipeline in order.")
    ap.add_argument("--list", action="store_true", help="print the stages and exit")
    ap.add_argument("--from", dest="start", type=int, default=1, help="first stage to run")
    ap.add_argument("--only", type=int, nargs="+", help="run only these stages")
    ap.add_argument("--npar", type=int, default=8, help="education systems in parallel")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    a = ap.parse_args()
    st = stages(a.npar)
    if a.list:
        for n, name, writes, _ in st:
            print(f"{n:>2}  {name:<48s} {writes}")
        return
    for n, name, _, cmds in st:
        if (a.only and n not in a.only) or (not a.only and n < a.start):
            continue
        print(f"[{n}] {name}", flush=True)
        for c in cmds:
            run(c, a.dry_run)
    print("done")


if __name__ == "__main__":
    main()
