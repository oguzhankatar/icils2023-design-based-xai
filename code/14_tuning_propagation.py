"""Propagation of tuning uncertainty into the PSU (school) bootstrap SEs.

The PSU bootstrap of 04d_cluster_boot.py refits XGBoost in every replicate with the
hyperparameters AND the number of boosting rounds fixed at the full-sample values
(results/final_hparams.json). Here we rerun the bootstrap on PV1 with N_BOOT = 30 replicates in
three variants that use the SAME resampled data sets (the first 30 draws of the
bootstrap stream of 04d_cluster_boot.py, rng = default_rng(SEED + 7)):

  (a) fixed  : as in 04d_cluster_boot.py -- fixed hyperparameters and fixed number of rounds;
  (b) rounds : fixed hyperparameters, number of rounds re-selected inside every replicate with
               the pipeline rule (models.xgb_cv_score: 3-fold school-grouped internal CV, early
               stopping with MAX_ROUNDS = 3000 and ES = 100, rounds = mean best iteration);
  (c) retune : full re-tuning inside every replicate with models.tune_xgb (Optuna TPE, 10 trials,
               same search space, enqueued default trial and inner CV as the pipeline, which used
               25 trials), only for 2 of the 5 systems.
Duplicated schools in a replicate keep their original IDSCHOOL, so all copies of a school fall in
the same internal fold (no leakage of a school copy into its own validation fold).

System selection (stated rule): the 34 analysed systems (DNW excluded, as in the meta-analysis)
are ranked by within-system XGBoost R^2 (results/cv_metrics.csv, mean over 5 PVs); the systems at
the 10th, 30th, 50th, 70th and 90th percentiles are taken with the inverted-CDF quantile (sorted
position ceil(q * 34), 1-based). Variant (c) is run in the 2 of these 5 systems with the fewest
students (computational feasibility).

All fits use CPU 'hist' (device = cpu) so that the three variants are exactly comparable; the
SHAP statistics (IMP, SLOPE, BLOCK; shapstats.stats with TOTWGTS) are those of the main analysis.

Run:   python 14_tuning_propagation.py --one <CNTRY> [nthread]      (one system, cached)
       python 14_tuning_propagation.py --part <CNTRY> <r0> <r1> <nthread> <ab|abc>
                                         (replicates r0..r1-1, one cache file per replicate)
       python 14_tuning_propagation.py --run [n_parallel] [nthread] (all 5, subprocesses)
       python 14_tuning_propagation.py --collect
Outputs: results/tuning_propagation/<CNTRY>.pkl (cache), results/tuning_propagation.csv,
         results/tuning_propagation_meta.json
Does not modify any existing script or results file.
"""
import json
import math
import pickle
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

import models as M
from config import PROC, RES, FEAT, FOCAL, PVS, SEED, WGT
from shapstats import stats, BLK_NAMES

CACHE = RES / "tuning_propagation"
CACHE.mkdir(exist_ok=True)
PARTS = CACHE / "parts"
PARTS.mkdir(exist_ok=True)
N_BOOT = 30
N_TRIALS_C = 10
PCTS = [0.1, 0.3, 0.5, 0.7, 0.9]
N_C = 2


def select_systems():
    cv = pd.read_csv(RES / "cv_metrics.csv")
    r2 = cv[(cv.model == "XGB") & (cv.CNTRY != "DNW")].groupby("CNTRY").r2.mean().sort_values()
    n = len(r2)
    sel = []
    for q in PCTS:
        pos = math.ceil(q * n)          # 1-based sorted position (inverted CDF)
        sel.append({"CNTRY": r2.index[pos - 1], "pct": q, "rank": pos, "r2_xgb": float(r2.iloc[pos - 1])})
    sizes = pd.read_parquet(PROC / "analysis.parquet", columns=["CNTRY"]).CNTRY.value_counts()
    for s in sel:
        s["n"] = int(sizes[s["CNTRY"]])
    c_sys = [s["CNTRY"] for s in sorted(sel, key=lambda s: s["n"])[:N_C]]
    for s in sel:
        s["retune"] = s["CNTRY"] in c_sys
    return sel, n


def run_country(c, nthread, reps=None, part_variants=None):
    """reps: replicate indices to compute (default all); with part_variants (e.g. "ab" or "abc")
    every replicate is saved as its own file in CACHE/parts (resumable, parallelisable)."""
    M.XGB_FIXED.update(device="cpu", nthread=nthread)
    sel, _ = select_systems()
    do_c = {s["CNTRY"]: s["retune"] for s in sel}[c]
    if part_variants is not None:
        do_c = do_c and "c" in part_variants
    reps = set(range(N_BOOT)) if reps is None else set(reps)
    df = pd.read_parquet(PROC / "analysis.parquet")
    g = df[df.CNTRY == c].reset_index(drop=True)
    X = g[FEAT].to_numpy(float)
    y = g[PVS[0]].to_numpy(float)
    s = g.IDSCHOOL.to_numpy()
    wt = g[WGT].to_numpy(float)
    w = M.norm_w(wt)
    F = len(FEAT)
    best = json.load(open(RES / "final_hparams.json"))[c]
    best = {k: v for k, v in best.items() if k != "inner_r2"}
    best["n_rounds"], best["max_depth"] = int(best["n_rounds"]), int(best["max_depth"])
    hp = {k: v for k, v in best.items() if k != "n_rounds"}

    rng = np.random.default_rng(SEED + 7)    # same stream as 04d_cluster_boot.py
    schools = np.unique(s)
    rows_of = {sc: np.where(s == sc)[0] for sc in schools}
    res = {v: {"imp": [], "slope": [], "blk": [], "rounds": [], "hp": []} for v in "abc"}

    def shap_stats(model, Xb, wb):
        ph = model.predict(xgb.DMatrix(Xb), pred_contribs=True)[:, :F]
        return stats(ph, Xb, wb)

    t0 = time.time()
    for r in range(N_BOOT):
        pick = rng.choice(schools, size=len(schools), replace=True)   # always drawn: same stream
        if r not in reps:
            continue
        pf = PARTS / f"{c}_{'abc' if do_c else 'ab'}_{r:02d}.pkl"
        if part_variants is not None and pf.exists():
            continue
        idx = np.concatenate([rows_of[sc] for sc in pick])
        Xb, yb, wb, sb, wtb = X[idx], y[idx], w[idx], s[idx], wt[idx]
        fits = {"a": dict(best)}
        # (b) re-select rounds with the pipeline rule, hyperparameters fixed
        _, nr = M.xgb_cv_score(Xb, yb, wb, sb, hp, k=3, seed=SEED)
        fits["b"] = dict(hp, n_rounds=int(nr))
        # (c) full re-tuning (10 Optuna trials)
        if do_c:
            bc, _ = M.tune_xgb(Xb, yb, wb, sb, N_TRIALS_C, seed=SEED + 1000 + r)
            bc["n_rounds"] = int(bc["n_rounds"])
            fits["c"] = bc
        for v, par in fits.items():
            a_, b_, k_ = shap_stats(M.fit_xgb(Xb, yb, wb, par), Xb, wtb)
            res[v]["imp"].append(a_)
            res[v]["slope"].append(b_)
            res[v]["blk"].append(k_)
            res[v]["rounds"].append(par["n_rounds"])
            res[v]["hp"].append(par)
        print(c, r, {v: fits[v]["n_rounds"] for v in fits}, f"{time.time() - t0:.0f}s", flush=True)
        if part_variants is not None:
            one = {v: {k: (np.array(x[-1:]) if k != "hp" else x[-1:]) for k, x in res[v].items()}
                   for v in "abc" if res[v]["imp"]}
            with open(pf, "wb") as f:
                pickle.dump({"r": r, "secs": time.time() - t0, **one}, f)
    out = {"CNTRY": c, "n_boot": N_BOOT, "retune": do_c, "best": best, "secs": time.time() - t0}
    for v in "abc":
        if res[v]["imp"]:
            out[v] = {k: (np.array(x) if k != "hp" else x) for k, x in res[v].items()}
    return out


def load_country(c):
    """Complete per-replicate 'abc' parts if present, else the full cache file, else assemble
    per-replicate part files (variants a, b from the 'ab' or 'abc' parts; c only from complete
    'abc' parts)."""
    full_abc_parts = all((PARTS / f"{c}_abc_{r:02d}.pkl").exists() for r in range(N_BOOT))
    if (CACHE / f"{c}.pkl").exists() and not full_abc_parts:
        o = pickle.load(open(CACHE / f"{c}.pkl", "rb"))
        o["source"] = "full"
        return o
    parts = {}
    for f in sorted(PARTS.glob(f"{c}_*_*.pkl")):
        q = pickle.load(open(f, "rb"))
        if q["r"] not in parts or "c" in q:
            parts[q["r"]] = parts.get(q["r"], {}) | q
    assert sorted(parts) == list(range(N_BOOT)), (c, sorted(parts))
    out = {"CNTRY": c, "n_boot": N_BOOT, "source": "parts",
           "secs": float(max(q["secs"] for q in parts.values()))}   # slowest chunk (cumulative)
    for v in "abc":
        have = [r for r in range(N_BOOT) if v in parts[r]]
        if v == "c" and len(have) < N_BOOT:
            out["c_reps_done"] = len(have)
            continue
        out[v] = {k: (np.concatenate([parts[r][v][k] for r in have]) if k != "hp"
                      else sum([parts[r][v][k] for r in have], [])) for k in ["imp", "slope", "blk", "rounds", "hp"]}
    out["retune"] = "c" in out
    return out


def collect():
    sel, nsys = select_systems()
    cb = pd.read_parquet(RES / "clusterboot_stats.parquet")
    cbb = pd.read_parquet(RES / "clusterboot_blocks.parquet")
    sh = pd.read_parquet(RES / "shap_country_stats.parquet")
    mi = pd.read_csv(RES / "meta_IMP.csv")
    top10 = mi[mi.analysis == "main"].sort_values("est", ascending=False).feature.head(10).tolist()
    rows, meta = [], {"systems": sel, "n_systems_ranked": nsys, "n_boot": N_BOOT,
                      "n_trials_retune": N_TRIALS_C, "top10": top10, "focal": FOCAL, "rounds": {}}
    for s in sel:
        c = s["CNTRY"]
        o = load_country(c)
        s["retune_done"] = "c" in o
        s["source"] = o["source"]
        meta["rounds"][c] = {v: {"fixed_or_mean": float(np.mean(o[v]["rounds"])),
                                 "sd": float(np.std(o[v]["rounds"], ddof=1)),
                                 "min": int(np.min(o[v]["rounds"])), "max": int(np.max(o[v]["rounds"]))}
                             for v in "abc" if v in o}
        meta["rounds"][c]["secs"] = o["secs"]
        for stat, key, names in [("IMP", "imp", FEAT), ("SLOPE", "slope", FEAT), ("BLOCK", "blk", BLK_NAMES)]:
            for j, f in enumerate(names):
                row = {"CNTRY": c, "pct": s["pct"], "r2_xgb": s["r2_xgb"], "n": s["n"], "retune": s["retune_done"],
                       "stat": stat, "feature": f, "focal": f in FOCAL, "top10": f in top10}
                for v in "abc":
                    if v in o:
                        x = o[v][key][:, j]
                        row[f"mean_{v}"] = np.nanmean(x)
                        row[f"se_{v}"] = np.nanstd(x, ddof=1)
                    else:
                        row[f"mean_{v}"] = row[f"se_{v}"] = np.nan
                rows.append(row)
    d = pd.DataFrame(rows)
    # 75-replicate PSU-bootstrap SE of 04d (GPU for n > 5000; none of the 5 systems) and the
    # main-analysis total SE (PSU bootstrap + 1.2 x between-PV variance) for reference
    cbl = pd.concat([cb[["CNTRY", "feature", "cb_var_imp"]].rename(columns={"cb_var_imp": "cb_var"}).assign(stat="IMP"),
                     cb[["CNTRY", "feature", "cb_var_slope"]].rename(columns={"cb_var_slope": "cb_var"}).assign(stat="SLOPE"),
                     cbb.rename(columns={"cb_var_block": "cb_var"}).assign(stat="BLOCK")])
    d = d.merge(cbl, on=["CNTRY", "feature", "stat"], how="left").merge(
        sh[["CNTRY", "feature", "stat", "est", "imp_var"]], on=["CNTRY", "feature", "stat"], how="left")
    d["se_cb75"] = np.sqrt(d.cb_var)
    for v in "bc":
        d[f"se_ratio_{v}a"] = d[f"se_{v}"] / d["se_a"]
        d[f"mean_ratio_{v}a"] = d[f"mean_{v}"] / d["mean_a"]
        # main-analysis total SE (cb_var + 1.2 imp_var) if the PSU part were inflated by se_v/se_a
        d[f"tot_se_ratio_{v}a"] = np.sqrt(d.cb_var * d[f"se_ratio_{v}a"] ** 2 + 1.2 * d.imp_var) / \
            np.sqrt(d.cb_var + 1.2 * d.imp_var)
    d["se_ratio_a_cb75"] = d.se_a / d.se_cb75
    d = d.drop(columns=["cb_var"]).rename(columns={"est": "main_est_pvmean", "imp_var": "main_imp_var"})
    d.to_csv(RES / "tuning_propagation.csv", index=False)

    # summaries
    fs = d[(d.focal | d.top10) & d.stat.isin(["IMP", "SLOPE"])]
    summ = {}
    for lab, sub in [("focal", fs[fs.focal]), ("top10", fs[fs.top10]), ("focal_top10", fs)]:
        for v in "bc":
            sl = sub[sub.stat == "SLOPE"].dropna(subset=[f"se_ratio_{v}a"])
            im = sub[sub.stat == "IMP"].dropna(subset=[f"mean_ratio_{v}a"])
            if len(sl) == 0:
                continue
            summ[f"{lab}_{v}"] = {
                "n_pairs": int(len(sl)), "n_sys": int(sl.CNTRY.nunique()),
                "slope_se_ratio_median": float(sl[f"se_ratio_{v}a"].median()),
                "slope_se_ratio_q1": float(sl[f"se_ratio_{v}a"].quantile(.25)),
                "slope_se_ratio_q3": float(sl[f"se_ratio_{v}a"].quantile(.75)),
                "imp_se_ratio_median": float(im[f"se_ratio_{v}a"].median()),
                "imp_mean_ratio_mean": float(im[f"mean_ratio_{v}a"].mean()),
                "imp_mean_ratio_min": float(im[f"mean_ratio_{v}a"].min()),
                "imp_mean_ratio_max": float(im[f"mean_ratio_{v}a"].max()),
                "slope_mean_absdiff_over_se_a_median": float(
                    ((sl[f"mean_{v}"] - sl.mean_a).abs() / sl.se_a).median()),
                "slope_tot_se_ratio_median": float(sl[f"tot_se_ratio_{v}a"].median()),
            }
    bl = d[d.stat == "BLOCK"]
    for v in "bc":
        b2 = bl.dropna(subset=[f"mean_ratio_{v}a"])
        summ[f"block_{v}"] = {"n_pairs": int(len(b2)), "mean_ratio_mean": float(b2[f"mean_ratio_{v}a"].mean()),
                              "mean_ratio_min": float(b2[f"mean_ratio_{v}a"].min()),
                              "mean_ratio_max": float(b2[f"mean_ratio_{v}a"].max()),
                              "se_ratio_median": float(b2[f"se_ratio_{v}a"].median())}
    sa = fs[fs.stat == "SLOPE"]
    summ["check_a_vs_cb75_slope_se_ratio_median"] = float(sa.se_ratio_a_cb75.median())
    meta["summary"] = summ
    json.dump(meta, open(RES / "tuning_propagation_meta.json", "w"), indent=1, default=str)
    print(json.dumps(summ, indent=1))
    print(json.dumps(meta["rounds"], indent=1))


def run_all(n_par, nthread):
    sel, _ = select_systems()
    order = sorted(sel, key=lambda s: (not s["retune"], -s["n"]))   # retune systems first
    logdir = RES / "logs"
    logdir.mkdir(exist_ok=True)
    here = Path(__file__).parent

    def run(c):
        if (CACHE / f"{c}.pkl").exists():
            return c, 0
        t0 = time.time()
        with open(logdir / f"14_tuning_propagation_{c}.log", "w") as f:
            r = subprocess.run([sys.executable, str(here / Path(__file__).name), "--one", c, str(nthread)],
                               stdout=f, stderr=subprocess.STDOUT, cwd=here)
        print(f"{c}: exit={r.returncode} {time.time() - t0:.0f}s", flush=True)
        return c, r.returncode

    with ThreadPoolExecutor(n_par) as ex:
        res = list(ex.map(run, [s["CNTRY"] for s in order]))
    print("FAILED:", [c for c, rc in res if rc != 0] or "none")


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        c = sys.argv[2]
        nth = int(sys.argv[3]) if len(sys.argv) > 3 else 4
        out = run_country(c, nth)      # compute first, then write the cache file
        with open(CACHE / f"{c}.pkl", "wb") as f:
            pickle.dump(out, f)
    elif sys.argv[1] == "--run":
        t = time.time()
        run_all(int(sys.argv[2]) if len(sys.argv) > 2 else 5, int(sys.argv[3]) if len(sys.argv) > 3 else 2)
        print(f"total {time.time() - t:.0f}s")
    elif sys.argv[1] == "--part":
        # --part <CNTRY> <r0> <r1> <nthread> <ab|abc>: replicates r0..r1-1, one file per replicate
        c, r0, r1, nth, pv = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
        run_country(c, nth, reps=range(r0, r1), part_variants=pv)
    elif sys.argv[1] == "--collect":
        collect()
    elif sys.argv[1] == "--select":
        print(json.dumps(select_systems(), indent=1))
