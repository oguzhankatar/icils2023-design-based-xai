"""Robustness and 'naive-analysis' contrasts per education system (requires 04 results).

(a) Leakage: weighted out-of-sample R2 of the tuned XGBoost under student-level random 5-fold CV
    vs. school-grouped 5-fold CV (PV1).
(b) Naive explanation: unweighted XGBoost on PV1 only -> IMP/SLOPE (vs. design-based).
(c) Design effect of SHAP statistics: simple-random-sampling (student-level) bootstrap variance
    of IMP/SLOPE (PV1, 75 replicates, TOTWGTS) vs. JRR sampling variance of the same PV1 statistic.
(d) Model-class robustness: mixed-effects XGBoost (MERF, school random intercepts) SHAP
    importance of the fixed part, averaged over the five PVs.

Run: python runner.py 04b_robust.py 8   then   python 04b_robust.py --collect
"""
import json
import pickle
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold

import models as M
from config import PROC, RES, FEAT, PVS, REPW, SEED, WGT
from shapstats import stats
from survey import weighted_r2

CACHE = RES / "robust"
CACHE.mkdir(exist_ok=True)
N_BOOT = 75


def run_country(c, g, best):
    g = g.reset_index(drop=True)
    X = g[FEAT].to_numpy(float)
    y = g[PVS[0]].to_numpy(float)
    s = g.IDSCHOOL.to_numpy()
    wt = g[WGT].to_numpy(float)
    w = M.norm_w(wt)
    dX = xgb.DMatrix(X)
    F = len(FEAT)
    out = {"CNTRY": c}
    # (a) leakage: random vs grouped CV
    for name, folds in [("random", list(KFold(5, shuffle=True, random_state=SEED).split(X))),
                        ("grouped", M.group_folds(s, 5, SEED))]:
        pred = np.zeros(len(y))
        for tr, te in folds:
            pred[te] = M.fit_xgb(X[tr], y[tr], w[tr], best).predict(xgb.DMatrix(X[te]))
        out[f"r2_{name}"] = weighted_r2(y, pred, wt)
    # (b) naive: unweighted, PV1
    phi = M.fit_xgb(X, y, np.ones(len(y)), best).predict(dX, pred_contribs=True)[:, :F]
    out["naive_imp"], out["naive_slope"], _ = stats(phi, X, np.ones(len(y)))
    # design-based PV1 full-sample and JRR sampling variance (PV1 only)
    phi0 = M.fit_xgb(X, y, w, best).predict(dX, pred_contribs=True)[:, :F]
    imp0, sl0, _ = stats(phi0, X, wt)
    reps_imp, reps_sl = [], []
    for r in REPW:
        wr = g[r].to_numpy(float)
        ph = M.fit_xgb(X, y, M.norm_w(wr), best).predict(dX, pred_contribs=True)[:, :F]
        a, b_, _ = stats(ph, X, wr)
        reps_imp.append(a)
        reps_sl.append(b_)
    out["pv1_imp"], out["pv1_slope"] = imp0, sl0
    out["jrr_var_imp"] = ((np.array(reps_imp) - imp0) ** 2).sum(0)
    out["jrr_var_slope"] = ((np.array(reps_sl) - sl0) ** 2).sum(0)
    # (c) SRS bootstrap (students resampled with replacement, ignoring clustering/stratification)
    rng = np.random.default_rng(SEED)
    b_imp, b_sl = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        ph = M.fit_xgb(X[idx], y[idx], w[idx], best).predict(xgb.DMatrix(X[idx]), pred_contribs=True)[:, :F]
        a, b_, _ = stats(ph, X[idx], wt[idx])
        b_imp.append(a)
        b_sl.append(b_)
    out["srs_var_imp"] = np.nanvar(np.array(b_imp), axis=0, ddof=1)
    out["srs_var_slope"] = np.nanvar(np.array(b_sl), axis=0, ddof=1)
    # (d) mixed-effects XGBoost (MERF) SHAP importance of the fixed part, averaged over 5 PVs
    mimp = np.zeros(F)
    for pv in PVS:
        f, _ = M.fit_merf(X, g[pv].to_numpy(float), w, s, best)
        ph = f.predict(dX, pred_contribs=True)[:, :F]
        mimp += (wt[:, None] * np.abs(ph)).sum(0) / wt.sum() / len(PVS)
    out["merf_imp"] = mimp
    print(c, {k: round(v, 3) for k, v in out.items() if k.startswith("r2_")}, flush=True)
    return out


def collect():
    parts = [pickle.load(open(p, "rb")) for p in sorted(CACHE.glob("*.pkl"))]
    rows = []
    for o in parts:
        for j, f in enumerate(FEAT):
            rows.append({"CNTRY": o["CNTRY"], "feature": f,
                         **{k: o[k][j] for k in o if isinstance(o[k], np.ndarray)}})
    pd.DataFrame(rows).to_parquet(RES / "robust_feature_stats.parquet", index=False)
    pd.DataFrame([{k: o[k] for k in ["CNTRY", "r2_random", "r2_grouped"]} for o in parts]).to_csv(
        RES / "leakage_r2.csv", index=False)
    print("n systems:", len(parts))


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        c = sys.argv[2]
        df = pd.read_parquet(PROC / "analysis.parquet")
        best = json.load(open(RES / "final_hparams.json"))[c]
        best = {k: v for k, v in best.items() if k != "inner_r2"}
        best["n_rounds"] = int(best["n_rounds"])
        best["max_depth"] = int(best["max_depth"])
        g = df[df.CNTRY == c]
        if len(g) > 5000:  # large systems on the GPU (identical model, faster histogram build)
            M.XGB_FIXED["device"] = "cuda"
        res = run_country(c, g, best)
        with open(CACHE / f"{c}.pkl", "wb") as f:
            pickle.dump(res, f)
    elif sys.argv[1] == "--collect":
        collect()
