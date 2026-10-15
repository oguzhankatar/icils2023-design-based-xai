"""Calibration check of jackknife (JRR) standard errors for refitted tree ensembles.

The delete-one-zone jackknife assumes a smooth estimator; refitted gradient-boosted trees are not
smooth, so JRR may overstate their sampling variance. As an independent check we use a school
(PSU) bootstrap (75 replicates, schools resampled with replacement, TOTWGTS), which is valid for
non-smooth statistics, and compare its variance with the JRR and the SRS-bootstrap variances of
the same PV1 statistics (from 04b). Also: targeted drop-out of the evaluation/scam items only.

Run: python runner.py 04d_cluster_boot.py 8   then   python 04d_cluster_boot.py --collect
"""
import json
import pickle
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

import models as M
from config import PROC, RES, FEAT, PVS, SEED, WGT
from shapstats import stats
from survey import weighted_r2

CACHE = RES / "clusterboot"
CACHE.mkdir(exist_ok=True)
N_BOOT = 75
DROPS = {"school_eval_scam": ["OTL_S_EVAL", "OTL_S_SCAM"],
         "outside_eval_scam": ["OTL_O_EVAL", "OTL_O_SCAM"]}


def cv_r2(X, y, w, wt, s, best):
    pred = np.zeros(len(y))
    for tr, te in M.group_folds(s, 5, SEED):
        pred[te] = M.fit_xgb(X[tr], y[tr], w[tr], best).predict(xgb.DMatrix(X[te]))
    return weighted_r2(y, pred, wt)


def run_country(c, g, best):
    g = g.reset_index(drop=True)
    X = g[FEAT].to_numpy(float)
    y = g[PVS[0]].to_numpy(float)
    s = g.IDSCHOOL.to_numpy()
    wt = g[WGT].to_numpy(float)
    w = M.norm_w(wt)
    F = len(FEAT)
    rng = np.random.default_rng(SEED + 7)
    schools = np.unique(s)
    rows_of = {sc: np.where(s == sc)[0] for sc in schools}
    b_imp, b_sl, b_blk = [], [], []
    for _ in range(N_BOOT):
        pick = rng.choice(schools, size=len(schools), replace=True)
        idx = np.concatenate([rows_of[sc] for sc in pick])
        Xb = X[idx]
        ph = M.fit_xgb(Xb, y[idx], w[idx], best).predict(xgb.DMatrix(Xb), pred_contribs=True)[:, :F]
        a, b_, bk = stats(ph, Xb, wt[idx])
        b_imp.append(a)
        b_sl.append(b_)
        b_blk.append(bk)
    out = {"CNTRY": c, "cb_var_imp": np.nanvar(np.array(b_imp), axis=0, ddof=1),
           "cb_var_slope": np.nanvar(np.array(b_sl), axis=0, ddof=1),
           "cb_var_block": np.nanvar(np.array(b_blk), axis=0, ddof=1)}
    saved = RES / "dropfocal_saved.csv"
    prev = pd.read_csv(saved).set_index("CNTRY") if saved.exists() else None
    if prev is not None and c in prev.index:  # drop-out losses do not depend on the bootstrap
        out.update({k: float(v) for k, v in prev.loc[c].items() if k.startswith("loss_")})
    else:
        full = cv_r2(X, y, w, wt, s, best)
        for name, drop in DROPS.items():
            cols = [i for i, f in enumerate(FEAT) if f not in drop]
            out[f"loss_{name}"] = full - cv_r2(X[:, cols], y, w, wt, s, best)
    print(c, {k: round(v, 4) for k, v in out.items() if k.startswith("loss_")}, flush=True)
    return out


def collect():
    parts = [pickle.load(open(p, "rb")) for p in sorted(CACHE.glob("*.pkl"))]
    rows = []
    for o in parts:
        for j, f in enumerate(FEAT):
            rows.append({"CNTRY": o["CNTRY"], "feature": f, "cb_var_imp": o["cb_var_imp"][j],
                         "cb_var_slope": o["cb_var_slope"][j]})
    pd.DataFrame(rows).to_parquet(RES / "clusterboot_stats.parquet", index=False)
    from shapstats import BLK_NAMES
    brows = [{"CNTRY": o["CNTRY"], "feature": b, "cb_var_block": o["cb_var_block"][j]}
             for o in parts for j, b in enumerate(BLK_NAMES)]
    pd.DataFrame(brows).to_parquet(RES / "clusterboot_blocks.parquet", index=False)
    pd.DataFrame([{k: v for k, v in o.items() if not isinstance(v, np.ndarray)} for o in parts]).to_csv(
        RES / "dropfocal.csv", index=False)
    print("n systems:", len(parts))


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
        res = run_country(c, g, best)   # compute first, then write the cache file
        with open(CACHE / f"{c}.pkl", "wb") as f:
            pickle.dump(res, f)
    elif sys.argv[1] == "--collect":
        collect()
