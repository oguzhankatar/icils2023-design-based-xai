"""Design-based uncertainty for model
performance differences, calibration slopes and (paired) group calibration gaps.

All quantities are computed from the existing out-of-fold predictions of 03_cv.py
(results/oof_predictions.parquet), which are held FIXED: for every system, plausible value p (5) and
weight vector (TOTWGTS + the 75 JRR replicate weights SRWGT01-75) the smooth statistic is
re-evaluated with that weight vector. JRR sampling variance SV_p = sum_r (theta_pr - theta_p)^2
(no 1/2 factor); Rubin: Var = mean_p SV_p + (1 + 1/5) Var_p(theta_p) (survey.combine_pv). The
residual for PV p uses the model refitted to PV p (column <MODEL>_pv<p>). The variance is therefore
conditional on the fitted models (refit variability across replicate samples is not included).

(1) Weighted R^2 = 1 - SSE_w / SST_w (SST_w around the weighted mean of the same weight vector) of
    each model, and the paired differences XGB-RIDGE, MERF-XGB (ME-XGB minus XGBoost), RLMM-RIDGE.
    Outputs: results/delta_r2_design.csv (system level), results/meta_delta_r2.csv (pooled).
(2) Calibration slope: weighted OLS slope of PV on the prediction (as in 07_fairness.py), with
    JRR x PV SE; z for H0: slope = 1. Output: results/cal_slope_design.csv (system rows and pooled
    rows, column 'level').
(3) Group calibration gaps = weighted mean residual (PV - prediction) in group A minus group B
    (positive = model under-predicts A relative to B). Groupings (fixed across replicate weights):
      gender     : girls - boys (S_SEX 1 vs 0)
      ses_hisei  : lowest - highest within-system TOTWGTS-weighted HISEI quartile (ties included;
                   identical to 07_fairness.py)
      ses_books  : lowest - highest within-system TOTWGTS-weighted quartile of books at home
                   (S_HOMLIT 0-4; same rule, ties included, so group shares differ from 25%)
      ses_paredu : lowest - highest within-system quartile of parental education (S_HISCED 0-4; same
                   rule) -- second alternative SES measure
      immig      : immigrant background (first or second generation, S_IMMBGR = 1; ICILS provides
                   only the dichotomous index) - no immigrant background (S_IMMBGR = 0)
      lang       : other language at home (S_TLANG = 0) - test language at home (S_TLANG = 1)
      hisei_miss : students with missing HISEI - students with observed HISEI (diagnostic for the
                   students that the HISEI-quartile gap cannot use)
    A grouping is evaluated in a system only if both groups contain >= MIN_N students. For each
    model: the gap and group-wise weighted RMSE (each group, and RMSE_A - RMSE_B). Paired model
    differences of the gap on the same students (RIDGE-XGB, RLMM-MERF, RIDGE-MERF; = gap of the
    prediction difference) with JRR x PV SEs. Students with a missing grouping variable are
    counted (N, % of the system and of the pooled 34 systems).
    Outputs: results/fairness_extended.csv, results/meta_fairness_extended.csv
Pooling: 05_meta.reml (REML tau^2, truncated HKSJ CI on t_{k-1}, PI on t_{k-2}, I^2) across the
34 systems (DNW excluded; 'excl_ROU' sensitivity). Counts: systems with est > 0 and with
z > 1.96 / z < -1.96.

Pure numpy/pandas; does not modify any other script or result file.
Run: python 12_performance_contrasts.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from config import PROC, RES, PVS, REPW, WGT
from survey import combine_pv

spec = importlib.util.spec_from_file_location("meta05", Path(__file__).parent / "05_meta.py")
meta05 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(meta05)
reml, EXCL = meta05.reml, meta05.EXCL

MODELS = ["RIDGE", "RLMM", "XGB", "MERF"]
R2_CONTRASTS = [("XGB", "RIDGE"), ("MERF", "XGB"), ("RLMM", "RIDGE")]
GAP_CONTRASTS = [("RIDGE", "XGB"), ("RLMM", "MERF"), ("RIDGE", "MERF")]
GROUPINGS = ["gender", "ses_hisei", "ses_books", "ses_paredu", "immig", "lang", "hisei_miss"]
MIN_N = 30
NP = len(PVS)


def wq(x, w, q):
    """Weighted quantile, identical to 07_fairness.wq."""
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / w.sum()
    return x[o][np.searchsorted(cw, q)]


def quartile_groups(x, w):
    ok = ~np.isnan(x)
    q1, q3 = wq(x[ok], w[ok], 0.25), wq(x[ok], w[ok], 0.75)
    return ok & (x <= q1), ok & (x >= q3), ok


def groups(g, w):
    """Return {grouping: (A, B, observed_mask)} with boolean arrays."""
    out = {}
    sex = g.S_SEX.to_numpy(float)
    out["gender"] = (sex == 1, sex == 0, ~np.isnan(sex))
    out["ses_hisei"] = quartile_groups(g.S_HISEI.to_numpy(float), w)
    out["ses_books"] = quartile_groups(g.S_HOMLIT.to_numpy(float), w)
    out["ses_paredu"] = quartile_groups(g.S_HISCED.to_numpy(float), w)
    im = g.S_IMMBGR.to_numpy(float)
    out["immig"] = (im == 1, im == 0, ~np.isnan(im))
    tl = g.S_TLANG.to_numpy(float)
    out["lang"] = (tl == 0, tl == 1, ~np.isnan(tl))
    h = g.S_HISEI.to_numpy(float)
    out["hisei_miss"] = (np.isnan(h), ~np.isnan(h), np.ones(len(h), bool))
    return out


def wmeans(W, x, m):
    """Weighted mean of x over mask m for every weight column of W -> (76,)."""
    Wm = W[m]
    return (Wm.T @ x[m]) / Wm.sum(axis=0)


def r2_all(W, y, p):
    sw = W.sum(axis=0)
    yc = y - np.average(y, weights=W[:, 0])
    sy, syy = W.T @ yc, W.T @ yc ** 2
    sst = syy - sy ** 2 / sw
    return 1 - (W.T @ (y - p) ** 2) / sst


def slope_all(W, y, p):
    sw = W.sum(axis=0)
    yc = y - np.average(y, weights=W[:, 0])
    pc = p - np.average(p, weights=W[:, 0])
    mp, my = (W.T @ pc) / sw, (W.T @ yc) / sw
    cov = (W.T @ (pc * yc)) / sw - mp * my
    var = (W.T @ pc ** 2) / sw - mp ** 2
    return cov / var


def combine(vals):
    """vals: (P, 76) array, column 0 = full-sample weight. Returns dict of Rubin combination."""
    vals = np.asarray(vals, float)
    full = vals[:, 0]
    sv = ((vals[:, 1:] - full[:, None]) ** 2).sum(axis=1)
    est, se, u, b = combine_pv(full, sv)
    return dict(est=float(est), se=float(se), samp_var=float(u), imp_var=float(b))


def run_system(c, g):
    g = g.reset_index(drop=True)
    w = g[WGT].to_numpy(float)
    W = np.column_stack([w] + [g[r].to_numpy(float) for r in REPW])
    Y = g[PVS].to_numpy(float)
    P = {m: g[[f"{m}_pv{j + 1}" for j in range(NP)]].to_numpy(float) for m in MODELS}
    assert not np.isnan(Y).any() and not any(np.isnan(v).any() for v in P.values()), c
    n = len(g)
    r2rows, slrows, frows = [], [], []

    # (1) R2 and paired differences
    R2 = {m: np.array([r2_all(W, Y[:, j], P[m][:, j]) for j in range(NP)]) for m in MODELS}
    for m in MODELS:
        r2rows.append(dict(CNTRY=c, quantity=f"R2_{m}", n=n, **combine(R2[m])))
    for a, b in R2_CONTRASTS:
        r2rows.append(dict(CNTRY=c, quantity=f"dR2_{a}-{b}", n=n, **combine(R2[a] - R2[b])))

    # (2) calibration slope
    for m in MODELS:
        s = combine(np.array([slope_all(W, Y[:, j], P[m][:, j]) for j in range(NP)]))
        slrows.append(dict(level="system", CNTRY=c, model=m, n=n, **s,
                           z_vs1=(s["est"] - 1) / s["se"]))

    # (3) gaps, group RMSE, paired gap differences
    E = {m: Y - P[m] for m in MODELS}
    for gname, (A, B, obs) in groups(g, w).items():
        nA, nB, nmiss = int(A.sum()), int(B.sum()), int((~obs).sum())
        base = dict(CNTRY=c, grouping=gname, n=n, n_A=nA, n_B=nB, n_missing=nmiss,
                    pct_missing=100 * nmiss / n,
                    wshare_A=float(w[A].sum() / w[obs].sum()), wshare_B=float(w[B].sum() / w[obs].sum()))
        if min(nA, nB) < MIN_N:
            frows.append(dict(base, model="ALL", stat="skipped_small_group", est=np.nan, se=np.nan))
            continue
        # replicate weights must leave positive weight in each group
        assert (W[A].sum(axis=0) > 0).all() and (W[B].sum(axis=0) > 0).all(), (c, gname)
        for m in MODELS:
            e = E[m]
            ga = np.array([wmeans(W, e[:, j], A) - wmeans(W, e[:, j], B) for j in range(NP)])
            frows.append(dict(base, model=m, stat="gap", **combine(ga)))
            ra = np.array([np.sqrt(wmeans(W, e[:, j] ** 2, A)) for j in range(NP)])
            rb = np.array([np.sqrt(wmeans(W, e[:, j] ** 2, B)) for j in range(NP)])
            frows.append(dict(base, model=m, stat="rmse_A", **combine(ra)))
            frows.append(dict(base, model=m, stat="rmse_B", **combine(rb)))
            frows.append(dict(base, model=m, stat="rmse_diff", **combine(ra - rb)))
        for a, b in GAP_CONTRASTS:
            d = E[a] - E[b]
            gd = np.array([wmeans(W, d[:, j], A) - wmeans(W, d[:, j], B) for j in range(NP)])
            frows.append(dict(base, model=f"{a}-{b}", stat="gap_diff", **combine(gd)))
    return r2rows, slrows, frows


def pool(df, keys, tag_excl):
    rows = []
    for tag, excl in tag_excl:
        x = df[~df.CNTRY.isin(excl)].dropna(subset=["est", "se"])
        x = x[x.se > 0]
        for k, g in x.groupby(keys):
            r = reml(g.est.to_numpy(), g.se.to_numpy() ** 2)
            z = g.est / g.se
            r.update(dict(zip(keys, k if isinstance(k, tuple) else (k,))))
            r.update(analysis=tag, n_pos=int((g.est > 0).sum()), n_neg=int((g.est < 0).sum()),
                     n_pos_sig=int((z > 1.96).sum()), n_neg_sig=int((z < -1.96).sum()))
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    cols = ["CNTRY", "IDSTUD", "S_SEX", "S_HISEI", "S_HOMLIT", "S_HISCED", "S_IMMBGR", "S_TLANG",
            WGT] + PVS + REPW
    df = pd.read_parquet(PROC / "analysis.parquet", columns=cols)
    oof = pd.read_parquet(RES / "oof_predictions.parquet")
    d = df.merge(oof, on=["CNTRY", "IDSTUD"], how="inner", validate="one_to_one")
    assert len(d) == len(df) == len(oof)
    R2, SL, FR = [], [], []
    for c, g in d.groupby("CNTRY"):
        a, b, f = run_system(c, g)
        R2 += a
        SL += b
        FR += f
    r2 = pd.DataFrame(R2)
    r2["z"] = r2.est / r2.se
    sl = pd.DataFrame(SL)
    fr = pd.DataFrame(FR)
    fr["z"] = fr.est / fr.se
    tags = [("main", EXCL), ("excl_ROU", EXCL + ["ROU"])]

    # sanity: full-sample R2 per PV must reproduce cv_metrics.csv
    cvm = pd.read_csv(RES / "cv_metrics.csv").groupby(["CNTRY", "model"]).r2.mean()
    chk = r2[r2.quantity.str.startswith("R2_")].assign(model=lambda x: x.quantity.str[3:])
    diff = (chk.set_index(["CNTRY", "model"]).est - cvm).abs().max()
    print(f"max |R2 - cv_metrics R2| = {diff:.2e}")
    assert diff < 1e-8

    r2.to_csv(RES / "delta_r2_design.csv", index=False)
    mr = pool(r2, ["quantity"], tags)
    mr.to_csv(RES / "meta_delta_r2.csv", index=False)

    ms = pool(sl.rename(columns={}), ["model"], tags)
    # count of systems whose slope differs from 1
    for i, r in ms.iterrows():
        x = sl[(sl.model == r.model) & ~sl.CNTRY.isin(EXCL + (["ROU"] if r.analysis == "excl_ROU" else []))]
        ms.loc[i, "n_below1_sig"] = int((x.z_vs1 < -1.96).sum())
        ms.loc[i, "n_above1_sig"] = int((x.z_vs1 > 1.96).sum())
    ms["level"] = "pooled"
    ms["CNTRY"] = "POOLED"
    pd.concat([sl, ms], ignore_index=True).to_csv(RES / "cal_slope_design.csv", index=False)

    fr.to_csv(RES / "fairness_extended.csv", index=False)
    mf = pool(fr[fr.stat != "skipped_small_group"], ["grouping", "model", "stat"], tags)
    # missingness of the grouping variable (pooled 34 systems, main analysis)
    miss = fr[~fr.CNTRY.isin(EXCL)].drop_duplicates(["CNTRY", "grouping"])
    mm = miss.groupby("grouping").agg(N_total=("n", "sum"), N_missing=("n_missing", "sum"),
                                      pct_missing_min=("pct_missing", "min"),
                                      pct_missing_max=("pct_missing", "max"),
                                      n_A_total=("n_A", "sum"), n_B_total=("n_B", "sum"))
    mm["pct_missing"] = 100 * mm.N_missing / mm.N_total
    skipped = fr[(fr.stat == "skipped_small_group") & ~fr.CNTRY.isin(EXCL)]
    mm["systems_skipped"] = skipped.groupby("grouping").CNTRY.apply(lambda s: " ".join(sorted(s)))
    mm["systems_skipped"] = mm.systems_skipped.fillna("")
    mf = mf.merge(mm.reset_index(), on="grouping", how="left")
    mf.to_csv(RES / "meta_fairness_extended.csv", index=False)

    # ------------------------------------------------------------------ console summary
    pd.set_option("display.width", 220)
    m = mr[mr.analysis == "main"]
    print(m[["quantity", "k", "est", "ci_lo", "ci_hi", "tau", "I2", "n_pos", "n_pos_sig", "n_neg_sig"]]
          .round(4).to_string(index=False))
    m = ms[ms.analysis == "main"]
    print(m[["model", "k", "est", "ci_lo", "ci_hi", "tau", "n_below1_sig", "n_above1_sig"]].round(3).to_string(index=False))
    m = mf[(mf.analysis == "main") & mf.stat.isin(["gap", "gap_diff"])]
    print(m[["grouping", "model", "k", "est", "ci_lo", "ci_hi", "tau", "n_pos_sig", "n_neg_sig"]]
          .round(2).to_string(index=False))
    m = mf[(mf.analysis == "main") & (mf.stat == "rmse_diff")]
    print(m[["grouping", "model", "k", "est", "ci_lo", "ci_hi"]].round(2).to_string(index=False))
    print(mm.round(2).to_string())


if __name__ == "__main__":
    main()
