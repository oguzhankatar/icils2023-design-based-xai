"""Sensitivity of the information-evaluation findings to correlated predictors (drop-out analysis).

Per system (PV1, final tuned XGBoost hyperparameters, school-grouped 5-fold CV, TOTWGTS):
  full model vs. models without (i) OTL_O_SCAM, (ii) the whole 'learned outside school' block,
  (iii) the whole 'learned at school' block, (iv) the four teacher information-evaluation variables.
Reports the loss in out-of-sample R2 (a model-agnostic measure of unique predictive value), the
SHAP importance the remaining outside-school items receive once OTL_O_SCAM is removed, and the
weighted Spearman correlations among the outside-school items.

Run: python runner.py 04c_dropone.py 8   then   python 04c_dropone.py --collect
"""
import json
import pickle
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

import models as M
from config import PROC, RES, FEAT, FEATURES, PVS, SEED, WGT
from survey import weighted_r2

CACHE = RES / "dropone"
CACHE.mkdir(exist_ok=True)
OUT_BLK = [f for f in FEAT if FEATURES[f][0] == "OTL info (outside)"]
SCH_BLK = [f for f in FEAT if FEATURES[f][0] == "OTL info (school)"]
T_INFO = ["T_INFOEMPH", "T_ITE_INFO", "T_PD_INFO", "P_DECEPT"]
VARIANTS = {"full": [], "minus_OTL_O_SCAM": ["OTL_O_SCAM"], "minus_outside_block": OUT_BLK,
            "minus_school_block": SCH_BLK, "minus_teacher_info": T_INFO}


def cv_r2(X, y, w, wt, s, best):
    pred = np.zeros(len(y))
    for tr, te in M.group_folds(s, 5, SEED):
        pred[te] = M.fit_xgb(X[tr], y[tr], w[tr], best).predict(xgb.DMatrix(X[te]))
    return weighted_r2(y, pred, wt)


def run_country(c, g, best):
    g = g.reset_index(drop=True)
    y = g[PVS[0]].to_numpy(float)
    s = g.IDSCHOOL.to_numpy()
    wt = g[WGT].to_numpy(float)
    w = M.norm_w(wt)
    out = {"CNTRY": c}
    for name, drop in VARIANTS.items():
        cols = [f for f in FEAT if f not in drop]
        out[f"r2_{name}"] = cv_r2(g[cols].to_numpy(float), y, w, wt, s, best)
    cols = [f for f in FEAT if f != "OTL_O_SCAM"]
    Xr = g[cols].to_numpy(float)
    phi = M.fit_xgb(Xr, y, w, best).predict(xgb.DMatrix(Xr), pred_contribs=True)[:, :len(cols)]
    imp = (wt[:, None] * np.abs(phi)).sum(0) / wt.sum()
    for f in OUT_BLK:
        if f != "OTL_O_SCAM":
            out[f"imp_noscam_{f}"] = imp[cols.index(f)]
    rk = g[OUT_BLK].rank()
    cm = rk.corr()
    out["rho_scam_eval"] = cm.loc["OTL_O_SCAM", "OTL_O_EVAL"]
    out["rho_scam_mean_others"] = cm.loc["OTL_O_SCAM", [f for f in OUT_BLK if f != "OTL_O_SCAM"]].mean()
    print(c, {k: round(v, 3) for k, v in out.items() if k.startswith("r2_")}, flush=True)
    return out


def collect():
    rows = [pickle.load(open(p, "rb")) for p in sorted(CACHE.glob("*.pkl"))]
    d = pd.DataFrame(rows)
    for v in VARIANTS:
        if v != "full":
            d[f"loss_{v}"] = d["r2_full"] - d[f"r2_{v}"]
    d.to_csv(RES / "dropone.csv", index=False)
    print(d.filter(like="loss_").describe().round(4).to_string())
    print(d[["rho_scam_eval", "rho_scam_mean_others"]].describe().round(2).to_string())
    print(d.filter(like="imp_noscam").median().round(2).to_string())


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        c = sys.argv[2]
        df = pd.read_parquet(PROC / "analysis.parquet")
        g = df[df.CNTRY == c]
        if len(g) > 5000:
            M.XGB_FIXED["device"] = "cuda"
        best = json.load(open(RES / "final_hparams.json"))[c]
        best = {k: v for k, v in best.items() if k != "inner_r2"}
        best["n_rounds"], best["max_depth"] = int(best["n_rounds"]), int(best["max_depth"])
        with open(CACHE / f"{c}.pkl", "wb") as f:
            pickle.dump(run_country(c, g, best), f)
    elif sys.argv[1] == "--collect":
        collect()
