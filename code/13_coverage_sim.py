"""Coverage simulation: is the delete-a-zone jackknife (JRR) or the school (PSU) bootstrap
calibrated for SHAP statistics of refitted XGBoost?

Pseudo-population: all sampled students of one large system (default ESP: 508 schools, 11,799
students) are treated as a finite population of schools; outcome PV1CIL; all 79 predictors.

Estimand theta*: the SHAP statistic (SLOPE, IMP; shapstats.stats) of XGBoost fitted to the whole
pseudo-population (every student weight 1) with the system's tuned hyperparameters and number of
rounds from results/final_hparams.json (the configuration used by 04_shap_jrr.py). Because
row/column subsampling makes a single fit seed-dependent, theta* is the mean over N_POP_SEEDS
population fits (seeds SEED..SEED+N_POP_SEEDS-1); the seed SD of theta* is stored.

Sampling design (repeated R_TOTAL times), CSIM_DESIGN:
  wr  (primary)   simple random sampling of N_SCH schools WITH replacement (a school drawn twice
                  enters as two PSUs), all students of each sampled school; Hansen-Hurwitz weight
                  N_schools/N_SCH (constant). Mirrors the negligible sampling fraction of the real
                  ICILS school samples, which JRR and the bootstrap (no fpc) assume.
  wor (secondary) the same WITHOUT replacement; sampling fraction f = N_SCH/N_schools = 0.30. The N_SCH schools
are paired at random into N_SCH/2 JRR zones; in zone k one randomly chosen school gets weight 0
and its partner weight x2 (ICILS JRR, SV = sum_k (theta_k - theta)^2, no 1/2).

Per sample: the main-analysis estimator (fit XGBoost with the normalised design weights, same
hyperparameters, seed SEED; TreeSHAP; stats with the design weights) -> theta_hat.  For the first
N_VAR samples additionally (a) JRR: 75 refits with the replicate weights, exactly as 04_shap_jrr.py
(all rows kept, zero-weight rows inert, SHAP statistics with the replicate weights); (b) PSU
bootstrap: N_BOOT refits on N_SCH schools drawn with replacement, exactly as 04d_cluster_boot.py
(variance = replicate variance, ddof=1).
Smooth benchmarks (cheap, computed for ALL samples with the same JRR and bootstrap replicates):
standardised weighted OLS coefficient of every predictor (weighted-mean imputation + missing
indicators; coefficient x weighted SD, score points per SD) and the weighted mean of PV1.

Finite-population correction (wor only): neither JRR nor the bootstrap includes an fpc; --collect
reports them unadjusted ('none') and multiplied by (1 - f) ('fpc'). The fpc is exact for the
sampling component of smooth statistics but not for the refit-instability ("jitter") component of
tree ensembles, which does not shrink with f -- hence the with-replacement primary design.

Usage
  python 13_coverage_sim.py --time [device]       rough per-fit timing (no files written)
  python 13_coverage_sim.py --pop                 population fits -> results/coverage_sim/<C>_pop.pkl
  python 13_coverage_sim.py --one r               sample r       -> results/coverage_sim/<C>_<design>_s<r>.pkl
  python 13_coverage_sim.py --var r               variance reps  -> results/coverage_sim/<C>_<design>_v<r>.pkl
  python 13_coverage_runner.py <n_par>            pop + all samples in isolated subprocesses
  python 13_coverage_sim.py --collect             results/coverage_sim.csv, coverage_sim_summary.csv
Environment: CSIM_DEVICE (fit device, default cpu), CSIM_NTHREAD (default 1), CSIM_SHAP_DEVICE
(TreeSHAP device, default cuda), CSIM_CNTRY (ESP), CSIM_DESIGN (wr|wor, default wr).
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb

import models as M
from config import FEAT, FOCAL, PROC, RES, SEED
from shapstats import stats

CNTRY = os.environ.get("CSIM_CNTRY", "ESP")
DESIGN = os.environ.get("CSIM_DESIGN", "wr")
assert DESIGN in ("wr", "wor")
TAG = f"{CNTRY}_{DESIGN}"
N_SCH = 150          # schools per sample (=> 75 JRR zones)
R_TOTAL = 200        # Monte Carlo samples (sampling distribution of theta_hat)
N_VAR = 96           # samples r = 0..N_VAR-1 queued for XGBoost JRR + bootstrap (all completed are used)
N_BOOT = 75          # PSU bootstrap replicates (as 04d_cluster_boot.py)
N_POP_SEEDS = 5      # population fits averaged for theta*
N_TOP = 10           # most important non-focal predictors (ESP IMP ranking of the main analysis) in the eval set
Y = "PV1CIL"
OUT = RES / "coverage_sim"
OUT.mkdir(exist_ok=True)
F = len(FEAT)

M.XGB_FIXED["device"] = os.environ.get("CSIM_DEVICE", "cpu")
M.XGB_FIXED["nthread"] = int(os.environ.get("CSIM_NTHREAD", "1"))
# TreeSHAP device. Under heavy machine load (other jobs), multi-threaded CPU fits spin on OpenMP
# barriers (a 1 s fit took 60-350 s), so fits run single-threaded on the CPU and TreeSHAP runs on
# the GPU (GPUTreeShap; max |difference| to CPU TreeSHAP 5e-5 score points on sample 0).
SHAP_DEVICE = os.environ.get("CSIM_SHAP_DEVICE", "cuda")


def _no_power_throttling():
    """Windows 11 EcoQoS throttles background processes on hybrid CPUs (measured: a 1 s fit took
    20-30 s). Opt this process out (process attribute only; no system setting is changed)."""
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes

    class PPTS(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG),
                    ("StateMask", wintypes.ULONG)]
    st = PPTS(1, 1, 0)
    k = ctypes.windll.kernel32
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.SetProcessInformation(wintypes.HANDLE(k.GetCurrentProcess()), 4, ctypes.byref(st),
                            ctypes.sizeof(st))


_no_power_throttling()


def load():
    df = pd.read_parquet(PROC / "analysis.parquet", columns=["CNTRY", "IDSCHOOL", "IDSTUD", Y] + FEAT)
    g = df[df.CNTRY == CNTRY].sort_values(["IDSCHOOL", "IDSTUD"]).reset_index(drop=True)
    return g, g[FEAT].to_numpy(float), g[Y].to_numpy(float), g.IDSCHOOL.to_numpy()


def hparams():
    best = json.load(open(RES / "final_hparams.json"))[CNTRY]
    best = {k: v for k, v in best.items() if k != "inner_r2"}
    best["n_rounds"], best["max_depth"] = int(best["n_rounds"]), int(best["max_depth"])
    return best


def eval_set():
    s = pd.read_parquet(RES / "shap_country_stats.parquet")
    s = s[(s.CNTRY == CNTRY) & (s.stat == "IMP")].sort_values("est", ascending=False)
    top = [f for f in s.feature if f not in FOCAL][:N_TOP]
    return list(FOCAL) + top


# ------------------------------------------------------------------ statistics
def shap_stats(X, y, wfit, wstat, best, Xeval=None):
    """Main-analysis estimator: fit with normalised weights, TreeSHAP, stats with the raw weights."""
    b = M.fit_xgb(X, y, M.norm_w(wfit), best)
    if SHAP_DEVICE != M.XGB_FIXED["device"]:
        b.set_param({"device": SHAP_DEVICE})
    phi = b.predict(xgb.DMatrix(X), pred_contribs=True)[:, :F]
    imp, slope, _ = stats(phi, X, wstat)
    return imp, slope


def make_ols(Xpop):
    miss_cols = np.where(np.isnan(Xpop).any(axis=0))[0]   # fixed design (from the population)

    def ols(X, y, w):
        ok = w > 0
        X, y, w = X[ok], y[ok], w[ok]
        nan = np.isnan(X)
        wn = (~nan) * w[:, None]
        mu = np.nansum(np.where(nan, 0, X) * w[:, None], 0) / np.maximum(wn.sum(0), 1e-12)
        Z = np.where(nan, mu, X)
        D = np.hstack([np.ones((len(X), 1)), Z, nan[:, miss_cols].astype(float)])
        sw = np.sqrt(w)
        beta = np.linalg.lstsq(D * sw[:, None], y * sw, rcond=None)[0][1:1 + F]
        mz = np.average(Z, axis=0, weights=w)
        sd = np.sqrt(np.average((Z - mz) ** 2, axis=0, weights=w))
        return beta * sd, np.average(y, weights=w)
    return ols


# ------------------------------------------------------------------ design
def draw(r, schools):
    rs = np.random.default_rng([SEED, 13, r])
    sch = rs.choice(schools, N_SCH, replace=DESIGN == "wr")
    perm = rs.permutation(N_SCH)                    # random pairing into zones
    zone = np.empty(N_SCH, int)
    drop = np.zeros(N_SCH, bool)
    for k in range(N_SCH // 2):
        a, b = perm[2 * k], perm[2 * k + 1]
        zone[a] = zone[b] = k
        drop[a] = True                              # perm is random -> dropped school is random
    rb = np.random.default_rng([SEED, 13, r, 1])
    boot = [rb.choice(N_SCH, N_SCH, replace=True) for _ in range(N_BOOT)]
    return sch, zone, drop, boot


def design(r, s):
    schools = np.unique(s)
    sch, zone, drop, boot = draw(r, schools)
    rows = [np.where(s == sc)[0] for sc in sch]
    idx = np.concatenate(rows)
    pos = np.concatenate([np.full(len(rw), j) for j, rw in enumerate(rows)])   # school position
    w = np.full(len(idx), len(schools) / N_SCH)
    wj = []
    for k in range(N_SCH // 2):
        f = np.ones(N_SCH)
        f[(zone == k) & drop] = 0.0
        f[(zone == k) & ~drop] = 2.0
        wj.append(w * f[pos])
    bidx = [np.concatenate([np.where(pos == j)[0] for j in pick]) for pick in boot]
    return sch, idx, w, wj, bidx


def run_sample(r, X, y, s, best, ols):
    """Light task: theta_hat (XGBoost) and the smooth statistics with their JRR/bootstrap reps."""
    sch, idx, w, wj, bidx = design(r, s)
    Xs, ys = X[idx], y[idx]
    out = {"r": r, "n": len(idx), "schools": sch}
    t0 = time.time()
    out["imp"], out["slope"] = shap_stats(Xs, ys, w, w, best)
    out["t_fit"] = time.time() - t0
    out["ols"], out["mean"] = ols(Xs, ys, w)
    o = [ols(Xs, ys, wr) for wr in wj]
    out["ols_jrr"], out["mean_jrr"] = np.array([a for a, _ in o]), np.array([b for _, b in o])
    o = [ols(Xs[bi], ys[bi], w[bi]) for bi in bidx]
    out["ols_boot"], out["mean_boot"] = np.array([a for a, _ in o]), np.array([b for _, b in o])
    out["t_total"] = time.time() - t0
    return out


def run_var(r, X, y, s, best):
    """Heavy task: XGBoost JRR (75 refits) and PSU bootstrap (N_BOOT refits) for sample r."""
    sch, idx, w, wj, bidx = design(r, s)
    Xs, ys = X[idx], y[idx]
    t0 = time.time()
    out = {"r": r}
    out["imp"], out["slope"] = shap_stats(Xs, ys, w, w, best)
    o = [shap_stats(Xs, ys, wr, wr, best) for wr in wj]
    out["imp_jrr"], out["slope_jrr"] = np.array([a for a, _ in o]), np.array([b for _, b in o])
    o = [shap_stats(Xs[bi], ys[bi], w[bi], w[bi], best) for bi in bidx]
    out["imp_boot"], out["slope_boot"] = np.array([a for a, _ in o]), np.array([b for _, b in o])
    out["t_total"] = time.time() - t0
    return out


def run_pop(g, X, y, s, best, ols):
    w = np.ones(len(y))
    imps, slopes = [], []
    for k in range(N_POP_SEEDS):
        M.XGB_FIXED["seed"] = SEED + k
        a, b = shap_stats(X, y, w, w, best)
        imps.append(a)
        slopes.append(b)
    M.XGB_FIXED["seed"] = SEED
    o, m = ols(X, y, w)
    return {"imp_seeds": np.array(imps), "slope_seeds": np.array(slopes), "ols": o, "mean": m,
            "N": len(y), "N_schools": len(np.unique(s)), "best": best}


# ------------------------------------------------------------------ collect
def _cov(est, se, target):
    return np.mean(np.abs(est - target) <= 1.959964 * se, axis=0)


def collect_design(design, pop):
    tag = f"{CNTRY}_{design}"
    files = sorted(OUT.glob(f"{tag}_s*.pkl"))
    S = [pickle.load(open(p, "rb")) for p in files]
    S = sorted(S, key=lambda d: d["r"])
    R = len(S)
    f_samp = N_SCH / pop["N_schools"] if design == "wor" else 0.0
    fpc = 1 - f_samp
    ev = eval_set()
    jx = [FEAT.index(f) for f in ev]
    V = {d["r"]: d for d in (pickle.load(open(p, "rb")) for p in sorted(OUT.glob(f"{tag}_v*.pkl")))}
    heavy = [V[k] for k in sorted(V)]
    nv = len(heavy)
    byr = {d["r"]: d for d in S}
    dev = max(np.nanmax(np.abs(d["slope"] - byr[d["r"]]["slope"])) for d in heavy)
    print(f"max |theta_hat(var task) - theta_hat(light task)| for SLOPE: {dev:.2e}")
    rows = []
    specs = [("SLOPE", "slope", np.mean(pop["slope_seeds"], 0), np.std(pop["slope_seeds"], 0, ddof=1)),
             ("IMP", "imp", np.mean(pop["imp_seeds"], 0), np.std(pop["imp_seeds"], 0, ddof=1)),
             ("OLS", "ols", pop["ols"], np.zeros(F)),
             ("MEAN", "mean", np.atleast_1d(pop["mean"]), np.zeros(1))]
    for stat, key, tstar, tsd in specs:
        feats = ["PV1CIL"] if stat == "MEAN" else ev
        cols = [0] if stat == "MEAN" else jx
        full = np.array([np.atleast_1d(d[key]) for d in S])[:, cols]           # R x K
        VS = heavy if stat in ("SLOPE", "IMP") else S                            # variance samples
        fv = np.array([np.atleast_1d(d[key]) for d in VS])[:, cols]
        rj = np.array([np.asarray(d[f"{key}_jrr"]).reshape(len(d[f"{key}_jrr"]), -1) for d in VS])[:, :, cols]
        rb = np.array([np.asarray(d[f"{key}_boot"]).reshape(len(d[f"{key}_boot"]), -1) for d in VS])[:, :, cols]
        v_j = ((rj - fv[:, None, :]) ** 2).sum(1)
        v_b = rb.var(1, ddof=1)
        mean_hat = full.mean(0)
        sd = full.std(0, ddof=1)
        sd_sub = fv.std(0, ddof=1)
        for adj, a in ([("none", 1.0), ("fpc", fpc)] if design == "wor" else [("none", 1.0)]):
            se_j, se_b = np.sqrt(a * v_j), np.sqrt(a * v_b)
            for k, f in enumerate(feats):
                rows.append({
                    "CNTRY": CNTRY, "design": design, "stat": stat, "feature": f, "focal": f in FOCAL, "fpc": adj,
                    "R": R, "n_var": len(VS), "theta_star": tstar[cols[k]],
                    "theta_star_seed_sd": tsd[cols[k]], "mean_hat": mean_hat[k],
                    "bias": mean_hat[k] - tstar[cols[k]], "bias_mcse": sd[k] / np.sqrt(R),
                    "emp_sd": sd[k], "emp_sd_mcse": sd[k] / np.sqrt(2 * (R - 1)),
                    "emp_sd_varsubset": sd_sub[k],
                    "mean_se_jrr": se_j[:, k].mean(), "mean_se_boot": se_b[:, k].mean(),
                    "rmse_se_jrr": np.sqrt((a * v_j[:, k]).mean()),
                    "rmse_se_boot": np.sqrt((a * v_b[:, k]).mean()),
                    "ratio_jrr": se_j[:, k].mean() / sd[k], "ratio_boot": se_b[:, k].mean() / sd[k],
                    "ratio_jrr_sqrtvar": np.sqrt((a * v_j[:, k]).mean()) / sd[k],
                    "ratio_boot_sqrtvar": np.sqrt((a * v_b[:, k]).mean()) / sd[k],
                    "cov_star_jrr": _cov(fv[:, k], se_j[:, k], tstar[cols[k]]),
                    "cov_star_boot": _cov(fv[:, k], se_b[:, k], tstar[cols[k]]),
                    "cov_mean_jrr": _cov(fv[:, k], se_j[:, k], mean_hat[k]),
                    "cov_mean_boot": _cov(fv[:, k], se_b[:, k], mean_hat[k]),
                    "rel_bias": (mean_hat[k] - tstar[cols[k]]) / sd[k]})
    res = pd.DataFrame(rows)
    summ = []
    for (stat, adj), d in res.groupby(["stat", "fpc"], sort=False):
        nvar = int(d.n_var.iloc[0])
        row = {"CNTRY": CNTRY, "design": design, "stat": stat, "fpc": adj, "n_features": len(d), "R": int(d.R.iloc[0]),
               "n_var": nvar, "N_SCH": N_SCH, "N_schools_pop": pop["N_schools"],
               "N_students_pop": pop["N"], "fpc_factor": fpc if adj == "fpc" else 1.0,
               "samp_frac": f_samp, "mean_n_students": np.mean([d["n"] for d in S]),
               "mean_heavy_task_s": np.mean([d["t_total"] for d in heavy])}
        for c in ["ratio_jrr", "ratio_boot", "ratio_jrr_sqrtvar", "ratio_boot_sqrtvar", "cov_star_jrr",
                  "cov_star_boot", "cov_mean_jrr", "cov_mean_boot", "rel_bias"]:
            row[f"median_{c}"] = d[c].median()
            row[f"min_{c}"] = d[c].min()
            row[f"max_{c}"] = d[c].max()
        for c in ["cov_star_jrr", "cov_star_boot", "cov_mean_jrr", "cov_mean_boot"]:
            row[f"pooled_{c}"] = d[c].mean()
        row["median_abs_rel_bias"] = d.rel_bias.abs().median()
        row["cov_mcse_per_feature"] = np.sqrt(0.95 * 0.05 / nvar)
        row["median_emp_sd_mcse_rel"] = (d.emp_sd_mcse / d.emp_sd).median()
        for c in ["ratio_jrr", "ratio_boot"]:
            row[f"median_{c}_focal"] = d[d.focal][c].median() if d.focal.any() else np.nan
        summ.append(row)
    t = np.array([d["t_total"] for d in heavy])
    print(f"design={design}: R={R}, n_var={nv}, heavy-sample time mean {t.mean():.0f}s")
    print(pd.DataFrame(summ)[["stat", "fpc", "median_ratio_jrr", "median_ratio_boot",
                              "median_cov_star_jrr", "median_cov_star_boot",
                              "median_cov_mean_jrr", "median_cov_mean_boot"]].round(3).to_string())
    return res, pd.DataFrame(summ)


def collect():
    pop = pickle.load(open(OUT / f"{CNTRY}_pop.pkl", "rb"))
    parts = [collect_design(d, pop) for d in ("wr", "wor")
             if list(OUT.glob(f"{CNTRY}_{d}_v*.pkl"))]
    pd.concat([p[0] for p in parts]).to_csv(RES / "coverage_sim.csv", index=False)
    pd.concat([p[1] for p in parts]).to_csv(RES / "coverage_sim_summary.csv", index=False)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "--collect":
        collect()
        sys.exit()
    g, X, y, s = load()
    best = hparams()
    ols = make_ols(X)
    if cmd == "--time":
        if len(sys.argv) > 2:
            M.XGB_FIXED["device"] = sys.argv[2]
        schools = np.unique(s)
        sch, zone, drop, boot = draw(0, schools)
        idx = np.concatenate([np.where(s == sc)[0] for sc in sch])
        w = np.full(len(idx), len(schools) / N_SCH)
        shap_stats(X[idx], y[idx], w, w, best)   # warm-up
        t0 = time.time()
        for _ in range(3):
            shap_stats(X[idx], y[idx], w, w, best)
        t1 = time.time()
        for _ in range(3):
            ols(X[idx], y[idx], w)
        print(f"device={M.XGB_FIXED['device']} nthread={M.XGB_FIXED['nthread']} n={len(idx)}: "
              f"xgb+shap {(t1 - t0) / 3:.2f}s/fit, ols {(time.time() - t1) / 3:.3f}s", flush=True)
    elif cmd == "--pop":
        res = run_pop(g, X, y, s, best, ols)
        with open(OUT / f"{CNTRY}_pop.pkl", "wb") as f:
            pickle.dump(res, f)
    elif cmd == "--one":
        r = int(sys.argv[2])
        res = run_sample(r, X, y, s, best, ols)   # compute first, then write the cache file
        with open(OUT / f"{TAG}_s{r:03d}.pkl", "wb") as f:
            pickle.dump(res, f)
        print(f"sample {r}: n={res['n']} {res['t_total']:.0f}s", flush=True)
    elif cmd == "--var":
        r = int(sys.argv[2])
        res = run_var(r, X, y, s, best)
        with open(OUT / f"{TAG}_v{r:03d}.pkl", "wb") as f:
            pickle.dump(res, f)
        print(f"variance sample {r}: {res['t_total']:.0f}s", flush=True)
