"""Nested, school-grouped cross-validation of four model classes per education system.

Models: RIDGE (weighted ridge), RLMM (penalised random-intercept model: ridge + school random
intercepts), XGB (XGBoost), MERF (mixed-effects XGBoost with school random intercepts, MERF algorithm).
Weights: all four models use the final student weight TOTWGTS, so that model classes differ only
in their structure. All performance metrics use TOTWGTS.
Outer loop: 5 school-grouped folds (test schools never seen in training).
Inner loop: 3 school-grouped folds for hyperparameter tuning (XGBoost: Optuna TPE, 25 trials;
MERF reuses the tuned XGBoost hyperparameters; ridge penalties: 11-value grid), tuned on PV1 within each outer
training set and reused for PV2-5. Every model is evaluated on all five plausible values.

Run:  python runner.py 03_cv.py 8      (one subprocess per country; cached in results/cv/)
      python 03_cv.py --collect        (assemble results)
Outputs: results/cv_metrics.csv, results/oof_predictions.parquet, results/cv_hparams.json
"""
import json
import pickle
import sys
import time

import numpy as np
import pandas as pd

import models as M
from config import PROC, RES, FEAT, PVS, SEED, WGT
from survey import weighted_r2

N_OUTER, N_TRIALS = 5, 25
CACHE = RES / "cv"
CACHE.mkdir(exist_ok=True)
MODELS = ["RIDGE", "RLMM", "XGB", "MERF"]


def run_country(c, g):
    t0 = time.time()
    g = g.reset_index(drop=True)
    X = g[FEAT].to_numpy(float)
    w = M.norm_w(g[WGT])
    ws = M.norm_w(g.WGTFAC1 * g.WGTADJ1S)
    s = g.IDSCHOOL.to_numpy()
    Y = g[PVS].to_numpy(float)
    oof = {m: np.full((len(g), len(PVS)), np.nan) for m in MODELS}
    hp = []
    for k, (tr, te) in enumerate(M.group_folds(s, N_OUTER, SEED)):
        xbest, xr2 = M.tune_xgb(X[tr], Y[tr, 0], w[tr], s[tr], N_TRIALS, seed=SEED + k)
        ra, rr2 = M.tune_ridge(X[tr], Y[tr, 0], w[tr], s[tr], False, seed=SEED + k)
        la, lr2 = M.tune_ridge(X[tr], Y[tr, 0], w[tr], s[tr], True, seed=SEED + k)
        hp.append({"fold": k, "xgb": xbest, "xgb_inner_r2": xr2,
                   "ridge_alpha": ra, "ridge_inner_r2": rr2, "rlmm_alpha": la, "rlmm_inner_r2": lr2})
        for j in range(len(PVS)):
            y = Y[:, j]
            oof["RIDGE"][te, j] = M.PenalizedLinear(ra).fit(X[tr], y[tr], w[tr]).predict(X[te])
            oof["RLMM"][te, j] = M.PenalizedLinear(la, True).fit(X[tr], y[tr], w[tr], s[tr]).predict(X[te])
            oof["XGB"][te, j] = M.fit_xgb(X[tr], y[tr], w[tr], xbest).predict(M.xgb.DMatrix(X[te]))
            oof["MERF"][te, j] = M.fit_predict_merf(X[tr], y[tr], w[tr], s[tr], X[te], xbest)[0]
    rows = []
    wt = g[WGT].to_numpy()
    for m, P in oof.items():
        for j, pv in enumerate(PVS):
            y = Y[:, j]
            rows.append({"CNTRY": c, "model": m, "pv": pv, "r2": weighted_r2(y, P[:, j], wt),
                         "rmse": np.sqrt(np.average((y - P[:, j]) ** 2, weights=wt))})
    pred = pd.DataFrame({"CNTRY": c, "IDSTUD": g.IDSTUD, "IDSCHOOL": s})
    for m, P in oof.items():
        for j in range(len(PVS)):
            pred[f"{m}_pv{j + 1}"] = P[:, j]
    with open(CACHE / f"{c}.pkl", "wb") as f:
        pickle.dump({"rows": rows, "pred": pred, "hp": hp}, f)
    print(f"{c}: {time.time() - t0:.0f}s", flush=True)


def refit_linear(c, g):
    """Re-estimate only RIDGE and RLMM (same outer folds) and update the cache. Used after the
    linear-model preprocessing was changed; tree models are untouched."""
    g = g.reset_index(drop=True)
    X = g[FEAT].to_numpy(float)
    w = M.norm_w(g[WGT])
    ws = M.norm_w(g.WGTFAC1 * g.WGTADJ1S)
    s = g.IDSCHOOL.to_numpy()
    Y = g[PVS].to_numpy(float)
    cache = pickle.load(open(CACHE / f"{c}.pkl", "rb"))
    pred = cache["pred"]
    assert (pred.IDSTUD.to_numpy() == g.IDSTUD.to_numpy()).all()
    oof = {m: np.full((len(g), len(PVS)), np.nan) for m in ["RIDGE", "RLMM"]}
    for k, (tr, te) in enumerate(M.group_folds(s, N_OUTER, SEED)):
        ra, rr2 = M.tune_ridge(X[tr], Y[tr, 0], w[tr], s[tr], False, seed=SEED + k)
        la, lr2 = M.tune_ridge(X[tr], Y[tr, 0], w[tr], s[tr], True, seed=SEED + k)
        cache["hp"][k].update(ridge_alpha=ra, ridge_inner_r2=rr2, rlmm_alpha=la, rlmm_inner_r2=lr2)
        for j in range(len(PVS)):
            y = Y[:, j]
            oof["RIDGE"][te, j] = M.PenalizedLinear(ra).fit(X[tr], y[tr], w[tr]).predict(X[te])
            oof["RLMM"][te, j] = M.PenalizedLinear(la, True).fit(X[tr], y[tr], w[tr], s[tr]).predict(X[te])
    wt = g[WGT].to_numpy()
    rows = [r for r in cache["rows"] if r["model"] not in oof]
    for m, P in oof.items():
        for j, pv in enumerate(PVS):
            pred[f"{m}_pv{j + 1}"] = P[:, j]
            y = Y[:, j]
            rows.append({"CNTRY": c, "model": m, "pv": pv, "r2": weighted_r2(y, P[:, j], wt),
                         "rmse": np.sqrt(np.average((y - P[:, j]) ** 2, weights=wt))})
    cache["rows"], cache["pred"] = rows, pred
    with open(CACHE / f"{c}.pkl", "wb") as f:
        pickle.dump(cache, f)
    print(c, "linear refit", flush=True)


def collect():
    parts = [pickle.load(open(p, "rb")) for p in sorted(CACHE.glob("*.pkl"))]
    metrics = pd.DataFrame([r for p in parts for r in p["rows"]])
    metrics.to_csv(RES / "cv_metrics.csv", index=False)
    pd.concat([p["pred"] for p in parts], ignore_index=True).to_parquet(
        RES / "oof_predictions.parquet", index=False)
    hp = {p["pred"].CNTRY.iloc[0]: p["hp"] for p in parts}
    with open(RES / "cv_hparams.json", "w") as f:
        json.dump(hp, f, indent=1, default=float)
    summ = metrics.groupby(["CNTRY", "model"]).r2.mean().unstack().round(3)
    print(summ.to_string())
    print(summ.describe().round(3).to_string())
    print("n systems:", len(parts))


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        df = pd.read_parquet(PROC / "analysis.parquet")
        run_country(sys.argv[2], df[df.CNTRY == sys.argv[2]])
    elif sys.argv[1] == "--linear":
        from joblib import Parallel, delayed
        df = pd.read_parquet(PROC / "analysis.parquet")
        Parallel(n_jobs=8)(delayed(refit_linear)(c, g) for c, g in df.groupby("CNTRY"))
    elif sys.argv[1] == "--collect":
        collect()
