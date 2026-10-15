"""Design-based SHAP inference per education system.

For each system: XGBoost hyperparameters are tuned once on the full sample (PV1; 3-fold
school-grouped inner CV; 25 Optuna trials). The whole estimation pipeline (model fit + TreeSHAP +
summary statistic) is then repeated for every plausible value (5) x every weight set
(full-sample TOTWGTS + 75 JRR replicate weights) = 380 refits per system.

Summary statistics (computed with the same weight vector used for fitting):
  IMP_j   = sum_i w_i |phi_ij| / sum_i w_i                    (mean |SHAP|, score points)
  SLOPE_j = SD_w(x_j) * Cov_w(x_j, phi_j) / Var_w(x_j)        (SHAP-implied effect per SD)
  BLK_k   = sum_i w_i |sum_{j in k} phi_ij| / sum_i w_i        (block importance)
Combination (ICILS 2023 User Guide eq. 3.3): estimate = mean over PVs of full-sample values;
Var = mean_p sum_r (theta_pr - theta_p)^2 + (1 + 1/P) * Var_p(theta_p).

Run:  python runner.py 04_shap_jrr.py 8   then   python 04_shap_jrr.py --collect
Outputs: results/shap_country_stats.parquet (estimates, SEs), results/shap_values_full.parquet
(PV-averaged full-sample SHAP values), results/final_hparams.json
"""
import json
import pickle
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb

import models as M
from config import PROC, RES, FEAT, PVS, REPW, SEED, WGT
from shapstats import BLK_NAMES, stats
from survey import combine_pv

N_TRIALS = 25
def run_country(c, g):
    t0 = time.time()
    g = g.reset_index(drop=True)
    X = g[FEAT].to_numpy(float)
    s = g.IDSCHOOL.to_numpy()
    w0 = M.norm_w(g[WGT])
    best, r2 = M.tune_xgb(X, g[PVS[0]].to_numpy(float), w0, s, N_TRIALS, seed=SEED)
    dX = xgb.DMatrix(X)
    wsets = [g[WGT].to_numpy(float)] + [g[r].to_numpy(float) for r in REPW]
    P, R, F, B = len(PVS), len(wsets), len(FEAT), len(BLK_NAMES)
    IMP, SLO, BLK = np.zeros((P, R, F)), np.zeros((P, R, F)), np.zeros((P, R, B))
    phi_full = np.zeros((len(g), F))
    for p, pv in enumerate(PVS):
        y = g[pv].to_numpy(float)
        for r, wr in enumerate(wsets):
            wn = M.norm_w(wr)
            b = M.fit_xgb(X, y, wn, best)
            phi = b.predict(dX, pred_contribs=True)[:, :F]
            IMP[p, r], SLO[p, r], BLK[p, r] = stats(phi, X, wr)
            if r == 0:
                phi_full += phi / P
    out = []
    for name, A, labels in [("IMP", IMP, FEAT), ("SLOPE", SLO, FEAT), ("BLOCK", BLK, BLK_NAMES)]:
        full = A[:, 0, :]
        sv = ((A[:, 1:, :] - full[:, None, :]) ** 2).sum(1)
        est, se, u, bvar = combine_pv(full, sv)
        for k, lab in enumerate(labels):
            out.append({"CNTRY": c, "stat": name, "feature": lab, "est": est[k], "se": se[k],
                        "samp_var": u[k], "imp_var": bvar[k]})
    sh = pd.DataFrame(phi_full.astype("float32"), columns=FEAT)
    sh.insert(0, "IDSTUD", g.IDSTUD.to_numpy())
    sh.insert(0, "CNTRY", c)
    print(f"{c}: tuned R2={r2:.3f}, {time.time() - t0:.0f}s", flush=True)
    return out, sh, {c: {**best, "inner_r2": r2}}


CACHE = RES / "shap"
CACHE.mkdir(exist_ok=True)


def collect():
    parts = [pickle.load(open(p, "rb")) for p in sorted(CACHE.glob("*.pkl"))]
    pd.DataFrame([r for o in parts for r in o[0]]).to_parquet(RES / "shap_country_stats.parquet",
                                                            index=False)
    pd.concat([o[1] for o in parts], ignore_index=True).to_parquet(RES / "shap_values_full.parquet",
                                                                 index=False)
    hp = {}
    for o in parts:
        hp.update(o[2])
    with open(RES / "final_hparams.json", "w") as f:
        json.dump(hp, f, indent=1, default=float)
    print("n systems:", len(parts))


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        df = pd.read_parquet(PROC / "analysis.parquet")
        c = sys.argv[2]
        res = run_country(c, df[df.CNTRY == c])
        with open(CACHE / f"{c}.pkl", "wb") as f:
            pickle.dump(res, f)
    elif sys.argv[1] == "--collect":
        collect()
