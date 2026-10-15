"""Focal linear estimates, JRR vs bootstrap for smooth statistics and seed-jitter decomposition.
Does not modify any other script or result file.

(A) Unadjusted and background-adjusted design-based linear estimates for the 8 focal predictors.
    Per system, plausible value (5) and weight set (TOTWGTS + 75 JRR replicate weights SRWGT01-75):
      bivariate : weighted slope of PVxCIL on the predictor, per weighted SD of the predictor
                  (= Cov_w(x, y) / SD_w(x)), students with the predictor observed;
      background: coefficient of the predictor (times its weighted SD) in a weighted least-squares
                  regression of PVxCIL on the predictor plus the student background block except
                  expected education (S_SEX, S_AGE, S_IMMBGR, S_TLANG, S_HISEI, S_HISCED, S_HOMLIT),
                  students with the predictor observed. Missing background values: mean imputation
                  (TOTWGTS-weighted mean of the system, computed once on all students of the system)
                  plus a 0/1 missing indicator per background variable with any missing value in
                  the system. Imputation values are held fixed across replicate weights.
    The weighted SD used for standardisation is computed with the same weight vector as the fit.
    Combination: JRR sampling variance SV_p = sum_r (theta_pr - theta_p)^2 (no 1/2 factor), Rubin:
    Var = mean_p SV_p + (1 + 1/5) Var_p(theta_p) (survey.combine_pv). Pooling across the 34 systems
    (DNW excluded) with 05_meta.reml (REML tau^2, truncated HKSJ CI, PI on t_{k-2}, I^2).
    Outputs: results/focal_unadjusted.csv, results/meta_focal_unadjusted.csv
(B) Smooth-statistic benchmark of JRR vs the PSU bootstrap of 04d_cluster_boot.py (PV1): weighted
    mean CIL and weighted standardised bivariate slopes of CIL on OTL_O_SCAM and S_GENEFF. The
    bootstrap replicates 04d exactly: rows of the system in analysis.parquet order, schools =
    np.unique(IDSCHOOL), rng = default_rng(SEED + 7), 75 draws of rng.choice(schools, n_schools,
    replace=True), TOTWGTS carried along, variance = nanvar(ddof=1) of the replicate statistics.
    Diagnostic (weighted mean only): Taylor-linearised school-cluster variance, unstratified and
    stratified by JK zone, to show which design features each replication method reflects.
    Output: results/jrr_smooth_benchmark.csv
(C) Refit-noise (seed-jitter) decomposition of the PV1 SHAP SLOPE: s2_seed = across-seed variance
    (ddof=1, 5 XGBoost seeds; results/sanity_seeds_long.parquet) vs JRR PV1 sampling variance
    (results/robust_feature_stats.parquet, jrr_var_slope; 75 replicate refits, SV without 1/2) and
    PSU-bootstrap PV1 variance (results/clusterboot_stats.parquet, cb_var_slope).
    Output: results/seed_jitter_decomposition.csv

Pure numpy/pandas (no native ML library), so all systems run in one process.
Run: python 09_additional_analyses.py
"""
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import PROC, RES, FEAT, FEATURES, FOCAL, PVS, REPW, SEED, WGT
from survey import combine_pv

spec = importlib.util.spec_from_file_location("meta05", Path(__file__).parent / "05_meta.py")
meta05 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(meta05)
reml, EXCL = meta05.reml, meta05.EXCL

BG = [f for f in FEAT if FEATURES[f][0] == "Background" and f != "S_ISCED"]
assert BG == ["S_SEX", "S_AGE", "S_IMMBGR", "S_TLANG", "S_HISEI", "S_HISCED", "S_HOMLIT"], BG
MIN_N = 30
N_BOOT = 75


# ------------------------------------------------------------------ helpers
def wslope(x, Y, w):
    """Weighted Cov(x, Y) / SD(x) for each column of Y (x observed for all rows)."""
    sw = w.sum()
    mx = (w * x).sum() / sw
    dx = x - mx
    vx = (w * dx ** 2).sum() / sw
    if not vx > 0:
        return np.full(Y.shape[1], np.nan)
    my = (w[:, None] * Y).sum(0) / sw
    return (w[:, None] * dx[:, None] * (Y - my)).sum(0) / sw / np.sqrt(vx)


def wls_slope(x, Z, Y, w):
    """Coefficient of x (times weighted SD of x) in WLS of Y on [1, x, Z]."""
    sw = w.sum()
    mx = (w * x).sum() / sw
    vx = (w * (x - mx) ** 2).sum() / sw
    if not vx > 0:
        return np.full(Y.shape[1], np.nan)
    D = np.column_stack([np.ones(len(x)), x, Z])
    r = np.sqrt(w)
    coef = np.linalg.lstsq(D * r[:, None], Y * r[:, None], rcond=None)[0]
    return coef[1] * np.sqrt(vx)


def background_design(g, wt):
    """Mean-imputed background matrix + missing indicators (system-level, TOTWGTS means)."""
    cols, names = [], []
    for f in BG:
        v = g[f].to_numpy(float)
        miss = np.isnan(v)
        if miss.all():
            continue
        m = np.average(v[~miss], weights=wt[~miss])
        cols.append(np.where(miss, m, v))
        names.append(f)
        if miss.any():
            cols.append(miss.astype(float))
            names.append(f + "_miss")
    return np.column_stack(cols), names


# ------------------------------------------------------------------ (A)
def part_a(df):
    rows = []
    for c, g in df.groupby("CNTRY", sort=True):
        g = g.reset_index(drop=True)
        Y = g[PVS].to_numpy(float)
        wt = g[WGT].to_numpy(float)
        wsets = [wt] + [g[r].to_numpy(float) for r in REPW]
        Zall, znames = background_design(g, wt)
        for f in FOCAL:
            x = g[f].to_numpy(float)
            ok = ~np.isnan(x)
            if ok.sum() < MIN_N or np.nanstd(x) == 0:
                for model in ["bivariate", "background"]:
                    rows.append(dict(CNTRY=c, feature=f, model=model, n_obs=int(ok.sum())))
                continue
            xo, Yo, Zo = x[ok], Y[ok], Zall[ok]
            keep = Zo.std(0) > 0          # drop columns constant in the analytic sample
            Zo = Zo[:, keep]
            res = {"bivariate": np.zeros((len(wsets), len(PVS))),
                   "background": np.zeros((len(wsets), len(PVS)))}
            for r, w in enumerate(wsets):
                wo = w[ok]
                res["bivariate"][r] = wslope(xo, Yo, wo)
                res["background"][r] = wls_slope(xo, Zo, Yo, wo)
            for model, A in res.items():
                full = A[0]                                  # (P,)
                sv = ((A[1:] - full) ** 2).sum(0)            # (P,)
                est, se, u, b = combine_pv(full, sv)
                rows.append(dict(CNTRY=c, feature=f, model=model, n_obs=int(ok.sum()),
                                 n_bg_cols=int(keep.sum()) if model == "background" else 0,
                                 est=float(est), se=float(se), samp_var=float(u), imp_var=float(b),
                                 est_pv1=float(full[0]), jrr_var_pv1=float(sv[0])))
    d = pd.DataFrame(rows)
    d.to_csv(RES / "focal_unadjusted.csv", index=False)
    meta = []
    x = d[~d.CNTRY.isin(EXCL)].dropna(subset=["est", "se"])
    x = x[x.se > 0]
    for (f, model), gg in x.groupby(["feature", "model"]):
        r = reml(gg.est, gg.se ** 2)
        z = gg.est / gg.se
        meta.append(dict(feature=f, model=model, est=r["est"], se=r["se"], ci_lo=r["ci_lo"],
                         ci_hi=r["ci_hi"], pi_lo=r["pi_lo"], pi_hi=r["pi_hi"], tau=r["tau"],
                         I2=r["I2"], n_pos_sig=int((z > 1.96).sum()),
                         n_neg_sig=int((z < -1.96).sum()), k=r["k"], p=r["p"]))
    m = pd.DataFrame(meta)
    m["feature"] = pd.Categorical(m.feature, FOCAL, ordered=True)
    m = m.sort_values(["feature", "model"], ascending=[True, False]).reset_index(drop=True)
    m.to_csv(RES / "meta_focal_unadjusted.csv", index=False)
    print(m.round(3).to_string())
    return d, m


# ------------------------------------------------------------------ (B)
def smooth_stats(y, w, xs):
    out = [np.sum(w * y) / np.sum(w)]
    for x in xs:
        ok = ~np.isnan(x) & (w > 0)
        out.append(wslope(x[ok], y[ok, None], w[ok])[0] if ok.sum() >= MIN_N else np.nan)
    return np.array(out)


B_STATS = ["mean_CIL", "slope_OTL_O_SCAM", "slope_S_GENEFF"]


def part_b(df):
    rows = []
    for c in sorted(df.CNTRY.unique()):
        g = df[df.CNTRY == c].reset_index(drop=True)      # same row order as 04d
        y = g[PVS[0]].to_numpy(float)
        wt = g[WGT].to_numpy(float)
        xs = [g["OTL_O_SCAM"].to_numpy(float), g["S_GENEFF"].to_numpy(float)]
        full = smooth_stats(y, wt, xs)
        reps = np.array([smooth_stats(y, g[r].to_numpy(float), xs) for r in REPW])
        jrr = ((reps - full) ** 2).sum(0)
        # PSU bootstrap exactly as in 04d_cluster_boot.run_country
        s = g.IDSCHOOL.to_numpy()
        rng = np.random.default_rng(SEED + 7)
        schools = np.unique(s)
        rows_of = {sc: np.where(s == sc)[0] for sc in schools}
        bs = []
        for _ in range(N_BOOT):
            pick = rng.choice(schools, size=len(schools), replace=True)
            idx = np.concatenate([rows_of[sc] for sc in pick])
            bs.append(smooth_stats(y[idx], wt[idx], [x[idx] for x in xs]))
        boot = np.nanvar(np.array(bs), axis=0, ddof=1)
        # diagnostic for the weighted mean: Taylor-linearised with-replacement school-cluster
        # variance, (a) unstratified (what the 04d bootstrap targets) and (b) stratified by JK zone
        # (what the paired-zone JRR targets): V = sum_h n_h/(n_h-1) sum_s (t_hs - tbar_h)^2,
        # t_hs = school total of w_i (y_i - ybar) / sum(w)
        zsc = pd.Series(wt * (y - full[0]) / wt.sum()).groupby(s).sum()
        zone = g.groupby("IDSCHOOL").JKZONES.first().loc[zsc.index].to_numpy()
        lin_un = len(zsc) / (len(zsc) - 1) * ((zsc - zsc.mean()) ** 2).sum()
        lin_st = sum(len(t) / (len(t) - 1) * ((t - t.mean()) ** 2).sum()
                     for _, t in zsc.groupby(zone) if len(t) > 1)
        for k, nm in enumerate(B_STATS):
            rows.append(dict(CNTRY=c, stat=nm, est_pv1=full[k], jrr_var=jrr[k], boot_var=boot[k],
                             ratio_jrr_boot=jrr[k] / boot[k], n_schools=len(schools),
                             n_zones=int(g.JKZONES.nunique()),
                             lin_var_unstrat=lin_un if k == 0 else np.nan,
                             lin_var_strat_zone=lin_st if k == 0 else np.nan))
    d = pd.DataFrame(rows)
    d.to_csv(RES / "jrr_smooth_benchmark.csv", index=False)
    x = d[~d.CNTRY.isin(EXCL)]
    print(x.groupby("stat").ratio_jrr_boot.describe(percentiles=[.25, .5, .75]).round(3).to_string())
    m = x[x.stat == "mean_CIL"]
    print("mean CIL, medians: JRR/lin_strat %.3f  boot/lin_unstrat %.3f  lin_strat/lin_unstrat %.3f" % (
        (m.jrr_var / m.lin_var_strat_zone).median(), (m.boot_var / m.lin_var_unstrat).median(),
        (m.lin_var_strat_zone / m.lin_var_unstrat).median()))
    return d


# ------------------------------------------------------------------ (C)
def part_c():
    s = pd.read_parquet(RES / "sanity_seeds_long.parquet")
    sd = s.groupby(["CNTRY", "feature"]).SLOPE.agg(s2_seed=lambda v: np.var(v.dropna(), ddof=1),
                                                    n_seed=lambda v: int(v.notna().sum()))
    sd = sd.reset_index()
    seed0 = s[s.seed == SEED][["CNTRY", "feature", "SLOPE"]].rename(columns={"SLOPE": "slope_seed0"})
    rob = pd.read_parquet(RES / "robust_feature_stats.parquet")[["CNTRY", "feature", "pv1_slope",
                                                                 "jrr_var_slope"]]
    cb = pd.read_parquet(RES / "clusterboot_stats.parquet")[["CNTRY", "feature", "cb_var_slope"]]
    d = (sd.merge(seed0, on=["CNTRY", "feature"], validate="one_to_one")
         .merge(rob, on=["CNTRY", "feature"], how="left", validate="one_to_one")
         .merge(cb, on=["CNTRY", "feature"], how="left", validate="one_to_one"))
    d["jitter75_over_jrr"] = 75 * d.s2_seed / d.jrr_var_slope
    d["jitter_over_boot"] = d.s2_seed / d.cb_var_slope
    d["jrr_over_boot"] = d.jrr_var_slope / d.cb_var_slope
    d["jrr_over_boot_jitter_removed"] = (d.jrr_var_slope - 75 * d.s2_seed) / (d.cb_var_slope - d.s2_seed)
    d["excluded_DNW"] = d.CNTRY.isin(EXCL)
    d["in_subset"] = (~d.excluded_DNW & (d.n_seed == 5) & np.isfinite(d.s2_seed)
                      & (d.jrr_var_slope > 0) & (d.cb_var_slope > 0))
    d.to_csv(RES / "seed_jitter_decomposition.csv", index=False)
    x = d[d.in_subset]
    print("C: n pairs", len(x), "systems", x.CNTRY.nunique())
    print(x[["jitter75_over_jrr", "jitter_over_boot", "jrr_over_boot"]].describe(
        percentiles=[.25, .5, .75]).round(3).to_string())
    # check: seed-0 SLOPE of 04e reproduces the PV1 SLOPE of 04b (CPU vs GPU for large systems)
    print("max |seed0 - pv1_slope|:", np.nanmax(np.abs(d.slope_seed0 - d.pv1_slope)).round(4))
    return d


if __name__ == "__main__":
    t0 = time.time()
    df = pd.read_parquet(PROC / "analysis.parquet")
    print("A ...", flush=True)
    part_a(df)
    print(f"A done {time.time() - t0:.0f}s\nB ...", flush=True)
    part_b(df)
    print(f"B done {time.time() - t0:.0f}s\nC ...", flush=True)
    part_c()
    print(f"total {time.time() - t0:.0f}s")
