"""Mechanism of the JRR / PSU-bootstrap variance ratio of SHAP SLOPE.

Question: is JRR variance / PSU-bootstrap variance (median ~3 for SLOPE, results/
seed_jitter_decomposition.csv) driven by (i) Monte-Carlo refit noise of the stochastic learner
(row/column subsampling) that the unscaled JRR sum SV = sum_r (theta_r - theta)^2 adds 75 times,
or (ii) non-smoothness (instability) of refitted trees under small weight perturbations, which
would remain for a deterministic learner?

System selection (stated rule, fixed before any refit): per system, the median over predictors of
jrr_over_boot in results/seed_jitter_decomposition.csv (in_subset rows; 34 systems, DNW excluded).
Sort the 34 medians and take the systems at sorted position np.percentile(..., method='nearest')
for q = 10, 30, 50, 70, 90 (numpy 'nearest' order statistic), plus Kosovo (XKX), the system with
the largest median ratio.  -> PRT, USA, DEU, LUX, CHL, XKX. (DEU, n = 5065, was run on the GPU in
04b/04d, so its A_saved differs from the CPU reproduction A_cpu; all new refits here are CPU.)

Data and estimators follow the main analysis exactly: PV1, X = 79 predictors, tuned hyperparameters and
number of rounds of results/final_hparams.json, models.fit_xgb (CPU 'hist'; nthread see NTHREAD),
SHAP statistics shapstats.stats.
  JRR  (as 04b_robust.py / 04_shap_jrr.py): full fit with norm_w(TOTWGTS); replicate r refitted on
       ALL rows with norm_w(SRWGTr) (zero-weight rows kept, weight 0), same XGBoost seed;
       statistics with the raw replicate weights; SV = sum_r (theta_r - theta)^2 (no 1/2).
  PSU bootstrap (as 04d_cluster_boot.py): rng = default_rng(SEED + 7), 75 draws of
       rng.choice(np.unique(IDSCHOOL), n_schools, replace=True) (strata ignored), rows stacked,
       weights norm_w(TOTWGTS)[idx] (not re-normalised), statistics with TOTWGTS[idx];
       variance = nanvar(ddof=1). The same 75 school draws are used in every variant.
Variants:
  A  stochastic, as in the main analysis: tuned subsample / colsample_bytree, seed SEED. Saved PV1 results
     (robust_feature_stats.parquet jrr_var_*, clusterboot_stats.parquet cb_var_*) are reused
     (A_saved); the seed-SEED refits of variant C reproduce them on the CPU (A_cpu, check).
  B  deterministic: subsample = colsample_bytree = colsample_bylevel = colsample_bynode = 1,
     everything else unchanged. Check: full-sample fit with seeds SEED+0..4 -> across-seed var ~0.
  C  seed-averaged stochastic: every fit (full sample, each JRR replicate, each bootstrap replicate)
     repeated with XGBoost seeds SEED+0..SEED+4 (the seeds of 04e_sanity_local.py); statistic =
     mean over the 5 seeds. JRR and bootstrap variances of the seed means.
  C_debiased  noise-free extrapolation from the 5 seeds: with d_r^s = theta_r^s - theta_0^s,
     JRR_db = sum_r [ mean_s(d_r^s)^2 - var_s(d_r^s)/5 ],
     BOOT_db = var_b(mean_s theta_b^s) - mean_b var_s(theta_b^s)/5
     (var_s with ddof=1; unbiased for the seed-free variances; uses the same seeds across
     replicates, so common-random-number correlation is accounted for).
  Noise share of A: expected single-seed noise contribution to SV, sum_r var_s(d_r^s), relative to
     A_cpu SV (and mean_b var_s(theta_b^s) relative to the A_cpu bootstrap variance).
  L  smooth reference (no model): weighted bivariate slope Cov_w(x, PV1)/SD_w(x) of each predictor,
     same JRR and PSU bootstrap (shows the ratio a smooth statistic gets under this bootstrap,
     which ignores the stratification that the paired-zone JRR reflects).
Instability (variant B): per predictor, median_r |theta_r - theta_{r+1}| (adjacent JRR replicate
refits) and median_r |theta_r - theta| divided by the bootstrap SD of B; same metric for L.
If JRR and bootstrap agree and replicate deviations are Gaussian, the expected values are
0.674*sqrt(2/75) = 0.110 and 0.674/sqrt(75) = 0.078.

Run:   python 16_jrr_mechanism.py --run         (dispatches per-task subprocesses, <= 5 parallel,
                                                 cache results/jrr_mechanism/*.npz)
       python 16_jrr_mechanism.py --collect     (results/jrr_mechanism.csv, jrr_mechanism_summary.csv)
"""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from config import PROC, RES, FEAT, PVS, REPW, SEED, WGT

CACHE = RES / "jrr_mechanism"
CACHE.mkdir(exist_ok=True)
N_BOOT = 75
SEEDS = [SEED + k for k in range(5)]
F = len(FEAT)
N_PAR = 5
DET = dict(subsample=1.0, colsample_bytree=1.0, colsample_bylevel=1.0, colsample_bynode=1.0)
QS = [10, 30, 50, 70, 90]
# XGBoost threads per worker. The CPU was shared with other jobs; with nthread = 4 the OpenMP
# synchronisation of these small fits slowed them 5-10x under oversubscription, so tasks started
# after the first ones use nthread = 1 (env JRRM_NTHREAD). nthread changes only the random stream of
# the row/column subsampling (the deterministic variant B is unaffected up to floating point);
# each task (full fit + its 75 replicates) uses one setting, recorded in its .npz.
NTHREAD = int(os.environ.get("JRRM_NTHREAD", "1"))


def select_systems():
    d = pd.read_csv(RES / "seed_jitter_decomposition.csv")
    med = d[d.in_subset].groupby("CNTRY").jrr_over_boot.median().sort_values()
    vals = med.to_numpy()
    pick = []
    for q in QS:
        v = np.percentile(vals, q, method="nearest")
        pick.append((f"P{q}", med.index[np.where(vals == v)[0][0]], float(v)))
    pick.append(("max", med.index[-1], float(vals[-1])))
    return pick, med


def load_best(c):
    best = json.load(open(RES / "final_hparams.json"))[c]
    best = {k: v for k, v in best.items() if k != "inner_r2"}
    best["n_rounds"] = int(best["n_rounds"])
    best["max_depth"] = int(best["max_depth"])
    return best


# ----------------------------------------------------------------------------- worker
def task(c, kind, var):
    import xgboost as xgb
    import models as M
    from shapstats import stats
    M.XGB_FIXED.update(device="cpu", tree_method="hist", nthread=NTHREAD)
    df = pd.read_parquet(PROC / "analysis.parquet")
    g = df[df.CNTRY == c].reset_index(drop=True)
    X = g[FEAT].to_numpy(float)
    y = g[PVS[0]].to_numpy(float)
    s = g.IDSCHOOL.to_numpy()
    wt = g[WGT].to_numpy(float)
    w = M.norm_w(wt)
    best = load_best(c)
    if var == "det":
        best = {**best, **DET}
        seed = SEED
    else:
        seed = SEEDS[int(var[1:])]
    M.XGB_FIXED["seed"] = seed
    dX = xgb.DMatrix(X)

    def one(Xa, ya, wfit, wstat, dm=None):
        b = M.fit_xgb(Xa, ya, wfit, best)
        ph = b.predict(dm if dm is not None else xgb.DMatrix(Xa), pred_contribs=True)[:, :F]
        imp, sl, _ = stats(ph, Xa, wstat)
        return imp, sl

    out = {}
    t0 = time.time()
    if kind == "jrr":
        out["full_imp"], out["full_slope"] = one(X, y, w, wt, dX)
        ri, rs = [], []
        for r in REPW:
            wr = g[r].to_numpy(float)
            a, b_ = one(X, y, M.norm_w(wr), wr, dX)
            ri.append(a)
            rs.append(b_)
        out["rep_imp"], out["rep_slope"] = np.array(ri), np.array(rs)
    elif kind == "boot":
        rng = np.random.default_rng(SEED + 7)
        schools = np.unique(s)
        rows_of = {sc: np.where(s == sc)[0] for sc in schools}
        ri, rs = [], []
        for _ in range(N_BOOT):
            pick = rng.choice(schools, size=len(schools), replace=True)
            idx = np.concatenate([rows_of[sc] for sc in pick])
            a, b_ = one(X[idx], y[idx], w[idx], wt[idx])
            ri.append(a)
            rs.append(b_)
        out["rep_imp"], out["rep_slope"] = np.array(ri), np.array(rs)
    elif kind == "detseed":          # deterministic learner, full sample, seeds SEED+1..SEED+4
        ri, rs = [], []
        for sd in SEEDS[1:]:
            M.XGB_FIXED["seed"] = sd
            a, b_ = one(X, y, w, wt, dX)
            ri.append(a)
            rs.append(b_)
        out["rep_imp"], out["rep_slope"] = np.array(ri), np.array(rs)
    out["seconds"] = np.array(time.time() - t0)
    out["nthread"] = np.array(NTHREAD)
    np.savez(CACHE / f"{c}_{kind}_{var}.npz", **out)
    print(c, kind, var, f"{time.time() - t0:.0f}s", flush=True)


def all_tasks(systems):
    t = []
    for c in systems:
        for kind in ["jrr", "boot"]:
            for var in ["det"] + [f"s{k}" for k in range(5)]:
                t.append((c, kind, var))
        t.append((c, "detseed", "det"))
    return t


def run():
    pick, _ = select_systems()
    systems = [p[1] for p in pick]
    print("systems:", pick, flush=True)
    sizes = pd.read_parquet(PROC / "analysis.parquet", columns=["CNTRY"]).CNTRY.value_counts()
    pending = [t for t in all_tasks(systems) if not (CACHE / f"{t[0]}_{t[1]}_{t[2]}.npz").exists()]
    pending.sort(key=lambda t: -sizes[t[0]] * load_best(t[0])["n_rounds"])   # heaviest first
    logdir = RES / "logs"
    logdir.mkdir(exist_ok=True)
    here = Path(__file__)

    def go(t):
        t0 = time.time()
        with open(logdir / f"16_jrr_mechanism_{'_'.join(t)}.log", "w") as f:
            r = subprocess.run([sys.executable, str(here), "--task", *t], stdout=f,
                               stderr=subprocess.STDOUT, cwd=here.parent)
        print(*t, f"exit={r.returncode} {time.time() - t0:.0f}s", flush=True)
        return t, r.returncode

    t0 = time.time()
    with ThreadPoolExecutor(N_PAR) as ex:
        res = list(ex.map(go, pending))
    bad = [t for t, rc in res if rc != 0]
    print("FAILED:", bad if bad else "none", f"total {time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------- collect
def wslope_all(X, y, w):
    out = np.full(X.shape[1], np.nan)
    for j in range(X.shape[1]):
        ok = ~np.isnan(X[:, j]) & (w > 0)
        if ok.sum() < 30:
            continue
        x, yy, ww = X[ok, j], y[ok], w[ok]
        mx, my = np.average(x, weights=ww), np.average(yy, weights=ww)
        vx = np.average((x - mx) ** 2, weights=ww)
        if vx > 0:
            out[j] = np.average((x - mx) * (yy - my), weights=ww) / np.sqrt(vx)
    return out


def smooth_reference(c, df):
    g = df[df.CNTRY == c].reset_index(drop=True)
    X = g[FEAT].to_numpy(float)
    y = g[PVS[0]].to_numpy(float)
    s = g.IDSCHOOL.to_numpy()
    wt = g[WGT].to_numpy(float)
    full = wslope_all(X, y, wt)
    reps = np.array([wslope_all(X, y, g[r].to_numpy(float)) for r in REPW])
    rng = np.random.default_rng(SEED + 7)
    schools = np.unique(s)
    rows_of = {sc: np.where(s == sc)[0] for sc in schools}
    bs = []
    for _ in range(N_BOOT):
        idx = np.concatenate([rows_of[sc] for sc in rng.choice(schools, len(schools), replace=True)])
        bs.append(wslope_all(X[idx], y[idx], wt[idx]))
    return full, reps, np.array(bs)


def instab(full, reps, sd):
    adj = np.nanmedian(np.abs(np.diff(reps, axis=0)), axis=0) / sd
    dev = np.nanmedian(np.abs(reps - full), axis=0) / sd
    return adj, dev


def collect():
    pick, med = select_systems()
    df = pd.read_parquet(PROC / "analysis.parquet")
    rob = pd.read_parquet(RES / "robust_feature_stats.parquet").set_index(["CNTRY", "feature"])
    cb = pd.read_parquet(RES / "clusterboot_stats.parquet").set_index(["CNTRY", "feature"])
    L = lambda c, k, v: np.load(CACHE / f"{c}_{k}_{v}.npz")
    rows, secs = [], 0.0
    for tag, c, m in pick:
        J = {v: L(c, "jrr", v) for v in ["det"] + [f"s{k}" for k in range(5)]}
        Bt = {v: L(c, "boot", v) for v in ["det"] + [f"s{k}" for k in range(5)]}
        DS = L(c, "detseed", "det")
        secs += sum(float(z["seconds"]) for z in list(J.values()) + list(Bt.values()) + [DS])
        nt = lambda z: int(z["nthread"]) if "nthread" in z else 4     # first tasks: nthread 4
        nthr = (f"jrr:{''.join(str(nt(J[v])) for v in J)};boot:{''.join(str(nt(Bt[v])) for v in Bt)}"
                f";detseed:{nt(DS)}")
        lin_full, lin_reps, lin_bs = smooth_reference(c, df)
        res = {}
        for st in ["imp", "slope"]:
            jd, bd = J["det"], Bt["det"]
            # A_cpu (seed SEED) and B
            for nm, jj, bb in [("A_cpu", J["s0"], Bt["s0"]), ("B_det", jd, bd)]:
                res[(nm, st)] = (((jj[f"rep_{st}"] - jj[f"full_{st}"]) ** 2).sum(0),
                                 np.nanvar(bb[f"rep_{st}"], axis=0, ddof=1))
            # C: seed averages
            fs = np.array([J[f"s{k}"][f"full_{st}"] for k in range(5)])          # (5, F)
            rs = np.array([J[f"s{k}"][f"rep_{st}"] for k in range(5)])           # (5, 75, F)
            bs = np.array([Bt[f"s{k}"][f"rep_{st}"] for k in range(5)])          # (5, 75, F)
            d = rs - fs[:, None, :]                                              # d_r^s
            jC = (d.mean(0) ** 2).sum(0)
            bC = np.nanvar(bs.mean(0), axis=0, ddof=1)
            noise_j = d.var(0, ddof=1).sum(0)                  # single-seed noise in SV
            noise_b = np.nanmean(bs.var(0, ddof=1), axis=0)     # single-seed noise in boot var
            res[("C_seedavg", st)] = (jC, bC)
            res[("C_debiased", st)] = (jC - noise_j / 5, bC - noise_b / 5)
            res[("noise", st)] = (noise_j, noise_b)
            # A_saved
            jA = np.array([rob.loc[(c, f), f"jrr_var_{st}"] for f in FEAT])
            bA = np.array([cb.loc[(c, f), f"cb_var_{st}"] for f in FEAT])
            res[("A_saved", st)] = (jA, bA)
            # det seed check
            dsf = np.vstack([jd[f"full_{st}"][None], DS[f"rep_{st}"]])
            res[("detseed", st)] = (dsf.var(0, ddof=1), np.abs(dsf - dsf[0]).max(0))
            res[("full", st)] = (jd[f"full_{st}"], J["s0"][f"full_{st}"])
            res[("instab", st)] = instab(jd[f"full_{st}"], jd[f"rep_{st}"],
                                         np.sqrt(res[("B_det", st)][1]))
        res[("L_smooth", "slope")] = (((lin_reps - lin_full) ** 2).sum(0),
                                      np.nanvar(lin_bs, axis=0, ddof=1))
        res[("instabL", "slope")] = instab(lin_full, lin_reps, np.sqrt(res[("L_smooth", "slope")][1]))
        for j, f in enumerate(FEAT):
            for st in ["imp", "slope"]:
                r = dict(CNTRY=c, sel_rule=tag, sel_median_ratio=m, nthread_det_s0_s4=nthr,
                         feature=f, stat=st.upper(),
                         est_A=res[("full", st)][1][j], est_B=res[("full", st)][0][j])
                for nm in ["A_saved", "A_cpu", "B_det", "C_seedavg", "C_debiased"]:
                    jv, bv = res[(nm, st)]
                    r[f"jrr_{nm}"], r[f"boot_{nm}"] = jv[j], bv[j]
                    r[f"ratio_{nm}"] = jv[j] / bv[j] if (bv[j] > 0 and jv[j] >= 0) else np.nan
                if st == "slope":
                    jv, bv = res[("L_smooth", "slope")]
                    r["jrr_L_smooth"], r["boot_L_smooth"] = jv[j], bv[j]
                    r["ratio_L_smooth"] = jv[j] / bv[j] if bv[j] > 0 else np.nan
                    r["instab_adj_L"], r["instab_dev_L"] = (x[j] for x in res[("instabL", "slope")])
                r["noise_jrr_A"], r["noise_boot_A"] = (x[j] for x in res[("noise", st)])
                r["noise_share_jrr_A"] = r["noise_jrr_A"] / r["jrr_A_cpu"] if r["jrr_A_cpu"] > 0 else np.nan
                r["noise_share_boot_A"] = r["noise_boot_A"] / r["boot_A_cpu"] if r["boot_A_cpu"] > 0 else np.nan
                r["detseed_var"], r["detseed_maxabs"] = (x[j] for x in res[("detseed", st)])
                r["instab_adj_B"], r["instab_dev_B"] = (x[j] for x in res[("instab", st)])
                rows.append(r)
    d = pd.DataFrame(rows)
    d = d.replace([np.inf, -np.inf], np.nan)
    d.to_csv(RES / "jrr_mechanism.csv", index=False)

    # ---------------------------------------------------------------- summary
    V = ["A_saved", "A_cpu", "B_det", "C_seedavg", "C_debiased", "L_smooth"]
    srows = []
    for st, x in d.groupby("stat"):
        for c in list(dict.fromkeys(d.CNTRY)) + ["ALL"]:
            xx = x if c == "ALL" else x[x.CNTRY == c]
            r = dict(stat=st, CNTRY=c, sel_rule="" if c == "ALL" else xx.sel_rule.iloc[0],
                     nthread_det_s0_s4="" if c == "ALL" else xx.nthread_det_s0_s4.iloc[0])
            for v in V:
                if f"ratio_{v}" not in xx or xx[f"ratio_{v}"].notna().sum() == 0:
                    continue
                r[f"med_ratio_{v}"] = xx[f"ratio_{v}"].median()
                r[f"n_{v}"] = int(xx[f"ratio_{v}"].notna().sum())
                # pairs with a finite, positive bootstrap variance whose ratio is undefined
                # (negative debiased JRR) and the ratio of summed variances over predictors
                ok = xx[f"boot_{v}"] > 0
                r[f"n_negjrr_{v}"] = int((ok & (xx[f"jrr_{v}"] < 0)).sum())
                both = ok & xx[f"jrr_{v}"].notna()
                r[f"sumratio_{v}"] = xx.loc[both, f"jrr_{v}"].sum() / xx.loc[both, f"boot_{v}"].sum()
                if c == "ALL":
                    r[f"med_of_sys_med_{v}"] = xx.groupby("CNTRY")[f"ratio_{v}"].median().median()
            # ratio of summed variances (robust to per-predictor noise)
            r["med_noise_share_jrr_A"] = xx.noise_share_jrr_A.median()
            r["med_noise_share_boot_A"] = xx.noise_share_boot_A.median()
            r["max_detseed_var"] = xx.detseed_var.max()
            r["max_detseed_maxabs"] = xx.detseed_maxabs.max()
            r["med_instab_adj_B"] = xx.instab_adj_B.median()
            r["med_instab_dev_B"] = xx.instab_dev_B.median()
            if st == "SLOPE":
                r["med_instab_adj_L"] = xx.instab_adj_L.median()
                r["med_instab_dev_L"] = xx.instab_dev_L.median()
            r["med_absdiff_Asaved_Acpu_rel"] = (np.abs(xx.jrr_A_saved - xx.jrr_A_cpu) /
                                                xx.jrr_A_saved).median()
            srows.append(r)
    s = pd.DataFrame(srows)
    s.attrs["cpu_seconds"] = secs
    s.to_csv(RES / "jrr_mechanism_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print("systems:", pick)
    print(f"summed task time: {secs / 3600:.2f} CPU-process hours")
    cols = ["stat", "CNTRY", "sel_rule"] + [f"med_ratio_{v}" for v in V] + [
        "med_noise_share_jrr_A", "med_noise_share_boot_A", "max_detseed_var", "med_instab_adj_B",
        "med_instab_dev_B", "med_instab_adj_L", "med_instab_dev_L", "med_absdiff_Asaved_Acpu_rel"]
    print(s[[c for c in cols if c in s]].round(3).to_string())
    print(s[s.CNTRY == "ALL"].filter(like="med_of_sys").round(3).to_string())
    print(s[["stat", "CNTRY"] + [c for c in s if c.startswith(("sumratio_", "n_negjrr_"))]]
          .round(3).to_string())


if __name__ == "__main__":
    if sys.argv[1] == "--task":
        task(*sys.argv[2:5])
    elif sys.argv[1] == "--run":
        run()
    elif sys.argv[1] == "--collect":
        collect()
    elif sys.argv[1] == "--plan":
        p, med = select_systems()
        print(p)
        print(med.round(3).to_string())
