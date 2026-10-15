"""Fold-assignment sensitivity of Delta R^2 (XGBoost - weighted ridge).

The outer 5-fold school-grouped CV of 03_cv.py is repeated with different fold seeds. To isolate
the effect of the fold assignment, hyperparameters are held FIXED at each system's full-sample
tuned values (NON-NESTED; tuning used all students of the system, so these R^2 values are not
honest out-of-sample estimates and serve only to measure the spread across fold seeds):
  XGBoost : results/final_hparams.json (configuration and number of boosting rounds tuned on the
            full sample by 04_shap_jrr.py), models.fit_xgb (CPU 'hist', seed SEED; nthread overridden to 1, see run_pv);
  ridge   : no full-sample ridge penalty is cached, so it is tuned here once on the full sample
            exactly like 03_cv.py tunes it within a training set: models.tune_ridge on PV1,
            3 school-grouped inner folds, seed SEED, grid models.RIDGE_ALPHAS.
Fold seeds: SEED (the original outer split of 03_cv.py, refitted with the fixed hyperparameters)
and SEED + 1001, SEED + 1002, SEED + 1003 (3 extra seeds).
One task = (system, fold seed, PV): fit on the training schools with norm_w(TOTWGTS), predict the
test schools, weighted R^2 (TOTWGTS) of the out-of-fold predictions. Tasks run PV by PV (PV1 for
all systems first), so the collection step uses the PVs 1..J that are complete for every system
and seed; Delta = mean over those PVs of R2_XGB - R2_RIDGE, per seed.
Per system: SD (ddof = 1) of Delta over the fixed-hyperparameter seeds ('sd_fixed'); secondary:
SD over {nested original Delta from cv_metrics.csv (same PVs), extra seeds} ('sd_mixed'). Compared
with the design-based (JRR x PV) SE of Delta from results/delta_r2_design.csv
(12_performance_contrasts.py).

Cache: results/foldseeds/<CNTRY>_<seed>_pv<j>.pkl. CPU only.
Run:  python 12b_delta_r2_foldseeds.py --run <pv> [n_jobs]   (one PV for all systems and seeds)
      python 12b_delta_r2_foldseeds.py --collect
Output: results/delta_r2_foldseeds.csv
"""
import json
import pickle
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import models as M
from config import PROC, RES, FEAT, PVS, SEED, WGT
from survey import weighted_r2

N_OUTER = 5
SEEDS = [SEED, SEED + 1001, SEED + 1002, SEED + 1003]
CACHE = RES / "foldseeds"
CACHE.mkdir(exist_ok=True)


def ridge_alpha(c, g):
    p = CACHE / f"{c}_ridge_alpha.json"
    if p.exists():
        return json.loads(p.read_text())["alpha"]
    X = g[FEAT].to_numpy(float)
    a, r2 = M.tune_ridge(X, g[PVS[0]].to_numpy(float), M.norm_w(g[WGT]), g.IDSCHOOL.to_numpy(),
                         False, seed=SEED)
    p.write_text(json.dumps({"alpha": a, "inner_r2": r2}))
    return a


def run(c, g, seed, j, alpha, best):
    out = CACHE / f"{c}_{seed}_pv{j + 1}.pkl"
    if out.exists():
        return
    t0 = time.time()
    g = g.reset_index(drop=True)
    X = g[FEAT].to_numpy(float)
    w = M.norm_w(g[WGT])
    s = g.IDSCHOOL.to_numpy()
    y = g[PVS[j]].to_numpy(float)
    P = {m: np.full(len(y), np.nan) for m in ["RIDGE", "XGB"]}
    for tr, te in M.group_folds(s, N_OUTER, seed):
        P["RIDGE"][te] = M.PenalizedLinear(alpha).fit(X[tr], y[tr], w[tr]).predict(X[te])
        P["XGB"][te] = M.fit_xgb(X[tr], y[tr], w[tr], best).predict(M.xgb.DMatrix(X[te]))
    wt = g[WGT].to_numpy(float)
    r = {"CNTRY": c, "seed": seed, "pv": j + 1, "ridge_alpha": alpha}
    for m in P:
        assert not np.isnan(P[m]).any()
        r[f"r2_{m}"] = weighted_r2(y, P[m], wt)
    r["dR2"] = r["r2_XGB"] - r["r2_RIDGE"]
    r["secs"] = time.time() - t0
    pickle.dump(r, open(out, "wb"))
    print(f"{c} seed {seed} pv{j + 1}: dR2 {r['dR2']:.4f} ({r['secs']:.0f}s)", flush=True)


def run_pv(j, n_jobs):
    df = pd.read_parquet(PROC / "analysis.parquet", columns=["CNTRY", "IDSCHOOL", WGT] + FEAT + PVS)
    hp = json.load(open(RES / "final_hparams.json"))
    groups = dict(tuple(df.groupby("CNTRY")))
    alphas = dict(zip(groups, Parallel(n_jobs=n_jobs)(delayed(ridge_alpha)(c, g) for c, g in groups.items())))
    # nthread = 1 (more single-threaded workers): OpenMP threads spin-wait badly on a shared,
    # oversubscribed CPU. 'hist' trees do not depend on the thread count up to floating-point order.
    best = {c: {**{k: v for k, v in hp[c].items() if k != "inner_r2"}, "nthread": 1} for c in groups}
    # expensive systems first (n x boosting rounds) for load balancing
    tasks = sorted(((c, s) for c in groups for s in SEEDS),
                   key=lambda t: -len(groups[t[0]]) * best[t[0]]["n_rounds"])
    Parallel(n_jobs=n_jobs)(delayed(run)(c, groups[c], s, j, alphas[c], best[c]) for c, s in tasks)


def collect():
    long = pd.DataFrame([pickle.load(open(p, "rb")) for p in CACHE.glob("*_pv*.pkl")])
    systems = pd.read_csv(RES / "cv_metrics.csv").CNTRY.unique()
    need = len(systems) * len(SEEDS)
    J = 0
    while J < len(PVS) and (long.pv == J + 1).sum() == need:
        J += 1
    assert J >= 1, "PV1 not complete for all systems and seeds"
    long = long[long.pv <= J]
    per_seed = long.groupby(["CNTRY", "seed"]).agg(dR2=("dR2", "mean"), ridge_alpha=("ridge_alpha", "first"),
                                                   r2_XGB=("r2_XGB", "mean"), r2_RIDGE=("r2_RIDGE", "mean"))
    per_seed = per_seed.reset_index()
    cvm = pd.read_csv(RES / "cv_metrics.csv")
    cvm = cvm[cvm.pv.isin(PVS[:J])].groupby(["CNTRY", "model"]).r2.mean().unstack()
    nested = (cvm.XGB - cvm.RIDGE).rename("dR2_nested_original")
    des = pd.read_csv(RES / "delta_r2_design.csv")
    des = des[des.quantity == "dR2_XGB-RIDGE"].set_index("CNTRY")
    gb = per_seed.groupby("CNTRY")
    out = pd.DataFrame({
        "ridge_alpha_full": gb.ridge_alpha.first(),
        "n_pvs": J,
        "n_seeds": gb.dR2.size(),
        "dR2_fixed_mean": gb.dR2.mean(),
        "dR2_fixed_min": gb.dR2.min(),
        "dR2_fixed_max": gb.dR2.max(),
        "sd_fixed": gb.dR2.std(ddof=1),
    }).join(nested)
    extra = per_seed[per_seed.seed != SEED]
    out["sd_mixed"] = [np.std(np.r_[out.loc[c, "dR2_nested_original"], extra[extra.CNTRY == c].dR2], ddof=1)
                       for c in out.index]
    out["se_design"] = des.se
    out["dR2_design_est"] = des.est
    out["ratio_sd_fixed_se"] = out.sd_fixed / out.se_design
    out["ratio_sd_mixed_se"] = out.sd_mixed / out.se_design
    wide = per_seed.pivot(index="CNTRY", columns="seed", values="dR2")
    wide.columns = [f"dR2_seed{s}" for s in wide.columns]
    out = out.join(wide)
    out.index.name = "CNTRY"
    out = out.reset_index()
    out.to_csv(RES / "delta_r2_foldseeds.csv", index=False)
    x = out[out.CNTRY != "DNW"]
    pd.set_option("display.width", 200)
    print(x[["CNTRY", "dR2_nested_original", "dR2_fixed_mean", "sd_fixed", "sd_mixed", "se_design",
             "ratio_sd_fixed_se"]].round(4).to_string(index=False))
    print(f"PVs used: 1..{J}; seeds: {SEEDS}")
    print("median (34 systems): sd_fixed %.4f  sd_mixed %.4f  se_design %.4f  ratio %.3f  ratio_mixed %.3f"
          % (x.sd_fixed.median(), x.sd_mixed.median(), x.se_design.median(), x.ratio_sd_fixed_se.median(),
             x.ratio_sd_mixed_se.median()))


if __name__ == "__main__":
    if sys.argv[1] == "--run":
        run_pv(int(sys.argv[2]) - 1, int(sys.argv[3]) if len(sys.argv) > 3 else 12)
    elif sys.argv[1] == "--collect":
        collect()
