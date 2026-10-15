"""Out-of-sample grouped drop-out importance with
design-based standard errors, all five plausible values and a noise reference distribution.

Per education system and plausible value p (PV1-PV5): school-grouped 5-fold CV with the folds
models.group_folds(IDSCHOOL, 5, SEED) (the folds of 03_cv.py, 04b_robust.py, 04c_dropone.py and
04d_cluster_boot.py), XGBoost with the full-sample tuned hyperparameters and rounds of
results/final_hparams.json (as in 04c_dropone.py), fitted with norm_w(TOTWGTS), XGBoost seed SEED,
'hist'; device exactly as in 04c_dropone.py: GPU (device='cuda') for systems with > 5,000 students
(ESP, CZE, OMN, TWN, DEU), CPU (4 threads) otherwise, so the PV1 values reproduce 04c. Setting the
environment variable DROPOUT_FORCE_CPU=1 forces the CPU (used for CZE, which was run after the GPU
had been reserved for another job; its PV1 values therefore differ slightly from 04c by device). The
device used is stored in each cache file ('device').
Out-of-fold predictions are kept (cache results/dropout_pv/<CNTRY>.pkl, float32).

Drop sets: the 9 predictor blocks, and the focal sets
  scam_out      OTL_O_SCAM                         (04c minus_OTL_O_SCAM)
  pair_out      OTL_O_EVAL + OTL_O_SCAM             (04d outside_eval_scam)
  outside_block the 'OTL info (outside)' block      (04c minus_outside_block; = block row)
  school_block  the 'OTL info (school)' block       (04c minus_school_block; = block row)
  pair_school   OTL_S_EVAL + OTL_S_SCAM             (04d school_eval_scam)
  teacher_info  T_INFOEMPH, T_ITE_INFO, T_PD_INFO, P_DECEPT (04c minus_teacher_info)
  eval_both     OTL_S_EVAL + OTL_O_EVAL             (credibility-evaluation construct)
  scam_both     OTL_S_SCAM + OTL_O_SCAM             (scam/safety construct)
  eval_school   OTL_S_EVAL alone
  scam_school   OTL_S_SCAM alone
dR2 = R2_w(full) - R2_w(without the set), weighted R2 with TOTWGTS.

Design-based SE: the out-of-fold predictions of both models are held FIXED; dR2 is then a smooth
function of them, re-evaluated with every replicate weight SRWGT01-75 (weighted R2 of both prediction
vectors, SST around the replicate-weighted mean): SV_p = sum_r (dR2_pr - dR2_p)^2 (no 1/2 factor);
Rubin across the 5 PVs: Var = mean_p SV_p + (1 + 1/5) Var_p(dR2_p) (survey.combine_pv). The SE is
conditional on the fitted models (refit variability is not included).

Noise reference distribution: per system, rng = numpy.random.default_rng(SEED + 2000) (fresh per
system) draws 3 distinct predictors (rng.choice(79, 3, replace=False)), then for each of them two
independent permutations across all students of the system (rng.permutation(n), in that order),
giving 6 noise columns (permuted copies keep the marginal distribution and missingness of the
source predictor, but no relation to CIL). The augmented model (79 + 6 columns) is cross-validated
like the others, and null dR2 = R2(augmented) - R2(augmented without the noise subset) for the 6
single columns, the 3 pairs of copies and all 6 columns (the last = the full model) -> 10 null values
per PV, 50 per system; median and 95th percentile per system. A set counts as 'above null' when its
PV-averaged dR2 exceeds the system's null 95th percentile (conservative: the null values are per-PV,
the comparison statistic is a PV average).

Outside-school item correlations: 04c_dropone.py computed UNWEIGHTED Spearman correlations
(pandas rank().corr()). Here also the TOTWGTS-weighted Spearman correlation (weighted Pearson
correlation of weighted mid-ranks, ranks = cumulative weight share, pairwise-complete students) of
OTL_O_SCAM with OTL_O_EVAL and the mean over the other five outside-school items.

Pooling across 34 systems (DNW excluded; sensitivity excl. ROU): 05_meta.reml (REML, truncated HKSJ).
Run:  python runner.py 11b_dropout_pv.py 6    then    python 11b_dropout_pv.py --collect
Outputs: results/dropout_blocks_pv.csv, results/meta_dropout.csv, results/dropout_null.csv,
         results/dropout_null_long.csv, results/outside_item_corr_weighted.csv
Does not modify any other script or result file.
"""
import importlib.util
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

import models as M
from config import PROC, RES, FEAT, FEATURES, PVS, REPW, SEED, WGT
from shapstats import BLOCKS
from survey import combine_pv

CACHE = RES / "dropout_pv"
CACHE.mkdir(exist_ok=True)
F = len(FEAT)
NULL_SEED = SEED + 2000
OUT_BLK = [f for f in FEAT if FEATURES[f][0] == "OTL info (outside)"]
SCH_BLK = [f for f in FEAT if FEATURES[f][0] == "OTL info (school)"]
FOCAL_SETS = {"scam_out": ["OTL_O_SCAM"], "pair_out": ["OTL_O_EVAL", "OTL_O_SCAM"],
              "outside_block": OUT_BLK, "school_block": SCH_BLK,
              "pair_school": ["OTL_S_EVAL", "OTL_S_SCAM"],
              "teacher_info": ["T_INFOEMPH", "T_ITE_INFO", "T_PD_INFO", "P_DECEPT"],
              "eval_both": ["OTL_S_EVAL", "OTL_O_EVAL"], "scam_both": ["OTL_S_SCAM", "OTL_O_SCAM"],
              "eval_school": ["OTL_S_EVAL"], "scam_school": ["OTL_S_SCAM"]}
ALIAS = {"outside_block": "OTL info (outside)", "school_block": "OTL info (school)"}
SETS = {**{b: [FEAT[j] for j in ix] for b, ix in BLOCKS.items()},
        **{k: v for k, v in FOCAL_SETS.items() if k not in ALIAS}}
NULL_VARIANTS = [f"single{q}" for q in range(6)] + [f"pair{a}" for a in range(3)] + ["all6"]


def load_best(c):
    best = json.load(open(RES / "final_hparams.json"))[c]
    best = {k: v for k, v in best.items() if k != "inner_r2"}
    best["n_rounds"], best["max_depth"] = int(best["n_rounds"]), int(best["max_depth"])
    return best


def oof(X, y, w, folds, best):
    p = np.zeros(len(y))
    for tr, te in folds:
        p[te] = M.fit_xgb(X[tr], y[tr], w[tr], best).predict(xgb.DMatrix(X[te]))
    return p


def r2_all(y, p, W):
    """Weighted R2 for every column of W (n x R)."""
    sw = W.sum(0)
    ybar = (W * y[:, None]).sum(0) / sw
    sst = (W * (y[:, None] - ybar) ** 2).sum(0)
    sse = W.T @ ((y - p) ** 2)
    return 1 - sse / sst


def wrank(x, w):
    u, inv = np.unique(x, return_inverse=True)
    ws = np.bincount(inv, weights=w)
    mid = np.cumsum(ws) - ws / 2
    return mid[inv] / ws.sum()


def wcorr(a, b, w):
    ma, mb = np.average(a, weights=w), np.average(b, weights=w)
    return np.average((a - ma) * (b - mb), weights=w) / np.sqrt(
        np.average((a - ma) ** 2, weights=w) * np.average((b - mb) ** 2, weights=w))


def spearman_pair(x, y, w):
    ok = np.isfinite(x) & np.isfinite(y)
    wu = np.ones(ok.sum())
    return (wcorr(wrank(x[ok], w[ok]), wrank(y[ok], w[ok]), w[ok]),
            wcorr(wrank(x[ok], wu), wrank(y[ok], wu), wu))


def item_corr(g, wt):
    sc = g["OTL_O_SCAM"].to_numpy(float)
    out = {}
    others = [f for f in OUT_BLK if f != "OTL_O_SCAM"]
    rw, ru = {}, {}
    for f in others:
        rw[f], ru[f] = spearman_pair(sc, g[f].to_numpy(float), wt)
        out[f"rho_w_scam_{f}"], out[f"rho_u_scam_{f}"] = rw[f], ru[f]
    out["rho_w_scam_eval"], out["rho_u_scam_eval"] = rw["OTL_O_EVAL"], ru["OTL_O_EVAL"]
    out["rho_w_scam_mean_others"] = np.mean(list(rw.values()))
    out["rho_u_scam_mean_others"] = np.mean(list(ru.values()))
    cm = g[OUT_BLK].rank().corr()  # exactly as 04c_dropone.py (unweighted)
    out["rho_04c_scam_eval"] = cm.loc["OTL_O_SCAM", "OTL_O_EVAL"]
    out["rho_04c_scam_mean_others"] = cm.loc["OTL_O_SCAM", others].mean()
    return out


def run_country(c, g):
    t0 = time.time()
    g = g.reset_index(drop=True)
    n = len(g)
    X = g[FEAT].to_numpy(float)
    s = g.IDSCHOOL.to_numpy()
    wt = g[WGT].to_numpy(float)
    w = M.norm_w(wt)
    W = np.column_stack([wt] + [g[r].to_numpy(float) for r in REPW])
    best = load_best(c)
    folds = M.group_folds(s, 5, SEED)
    rng = np.random.default_rng(NULL_SEED)
    picks = rng.choice(F, 3, replace=False)
    N = np.column_stack([X[rng.permutation(n), j] for j in picks for _ in range(2)])
    XA = np.hstack([X, N])
    models = {"full": list(range(F))}
    for name, drop in SETS.items():
        models[name] = [j for j, f in enumerate(FEAT) if f not in drop]
    models["aug"] = list(range(F + 6))
    for q in range(6):
        models[f"aug_minus_single{q}"] = [j for j in range(F + 6) if j != F + q]
    for a in range(3):
        models[f"aug_minus_pair{a}"] = [j for j in range(F + 6) if j not in (F + 2 * a, F + 2 * a + 1)]
    P = len(PVS)
    pred = {m: np.zeros((n, P), dtype="float32") for m in models}
    r2 = {m: np.zeros((P, W.shape[1])) for m in models}
    for p, pv in enumerate(PVS):
        y = g[pv].to_numpy(float)
        for m, cols in models.items():
            pr = oof(XA[:, cols], y, w, folds, best)
            pred[m][:, p] = pr
            r2[m][p] = r2_all(y, pr, W)
    out = {"CNTRY": c, "n": n, "noise_features": [FEAT[j] for j in picks], "r2": r2, "pred": pred,
           "IDSTUD": g.IDSTUD.to_numpy(), "folds_seed": SEED, "null_seed": NULL_SEED,
           "device": M.XGB_FIXED["device"]}
    out.update(item_corr(g, wt))
    out["sec"] = time.time() - t0
    d = {k: (r2["full"][:, 0] - r2[k][:, 0]).mean() for k in SETS}
    print(f"{c}: " + " ".join(f"{k}={1000 * v:.2f}" for k, v in d.items()) + f" | {out['sec']:.0f}s",
          flush=True)
    return out


# ------------------------------------------------------------------ collect
def collect():
    spec = importlib.util.spec_from_file_location("meta05", Path(__file__).parent / "05_meta.py")
    meta05 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(meta05)
    reml, EXCL = meta05.reml, meta05.EXCL
    parts = [pickle.load(open(p, "rb")) for p in sorted(CACHE.glob("*.pkl"))]
    rows, nrows, nlong, crow = [], [], [], []
    for o in parts:
        c, r2 = o["CNTRY"], o["r2"]
        nv = {}
        for v in NULL_VARIANTS:
            other = "full" if v == "all6" else f"aug_minus_{v}"
            nv[v] = r2["aug"][:, 0] - r2[other][:, 0]
            for p in range(len(PVS)):
                nlong.append({"CNTRY": c, "variant": v, "pv": PVS[p], "dr2": nv[v][p]})
        allnull = np.concatenate(list(nv.values()))
        q50, q95 = np.median(allnull), np.percentile(allnull, 95)
        nrows.append({"CNTRY": c, "noise_features": ",".join(o["noise_features"]), "n_null": len(allnull),
                      "null_median": q50, "null_p95": q95, "null_mean": allnull.mean(),
                      "null_min": allnull.min(), "null_max": allnull.max(),
                      "null_sd": allnull.std(ddof=1), "null_seed": o["null_seed"],
                      "r2_full_mean_pv": r2["full"][:, 0].mean(), "r2_aug_mean_pv": r2["aug"][:, 0].mean()})
        for name in list(SETS) + list(ALIAS):
            key = ALIAS.get(name, name)
            D = r2["full"] - r2[key]                      # P x (1 + 75)
            full = D[:, 0]
            sv = ((D[:, 1:] - full[:, None]) ** 2).sum(1)
            est, se, u, b = combine_pv(full, sv)
            drop = FOCAL_SETS.get(name, SETS.get(name))
            rows.append({"CNTRY": c, "set": name,
                         "set_type": "block" if name in BLOCKS else "focal",
                         "same_as_block": ALIAS.get(name, ""), "k_vars": len(drop),
                         "variables": ",".join(drop), "dr2": float(est), "se": float(se),
                         "samp_var": float(u), "imp_var": float(b), "var": float(se ** 2),
                         "z": float(est / se) if se > 0 else np.nan,
                         **{f"dr2_pv{p + 1}": full[p] for p in range(len(PVS))},
                         "r2_full_mean_pv": r2["full"][:, 0].mean(),
                         "null_median": q50, "null_p95": q95, "above_null": bool(est > q95)})
        crow.append({"CNTRY": c, **{k: v for k, v in o.items() if k.startswith("rho_")}})
    d = pd.DataFrame(rows)
    d.to_csv(RES / "dropout_blocks_pv.csv", index=False)
    pd.DataFrame(nrows).to_csv(RES / "dropout_null.csv", index=False)
    pd.DataFrame(nlong).to_csv(RES / "dropout_null_long.csv", index=False)
    cr = pd.DataFrame(crow)
    cr.to_csv(RES / "outside_item_corr_weighted.csv", index=False)
    out = []
    for excl, tag in [(EXCL, "main"), (EXCL + ["ROU"], "excl_ROU")]:
        x = d[~d.CNTRY.isin(excl)]
        for name, gs in x.groupby("set", sort=False):
            gs = gs[gs["var"] > 0]
            r = reml(gs.dr2, gs["var"])
            r.update(set=name, set_type=gs.set_type.iloc[0], same_as_block=gs.same_as_block.iloc[0],
                     k_vars=int(gs.k_vars.iloc[0]), variables=gs.variables.iloc[0], analysis=tag,
                     n_above_null=int(gs.above_null.sum()), n_pos=int((gs.dr2 > 0).sum()),
                     n_pos_sig=int((gs.z > 1.96).sum()), n_neg_sig=int((gs.z < -1.96).sum()),
                     median_dr2=gs.dr2.median(), median_null_p95=gs.null_p95.median())
            out.append(r)
    m = pd.DataFrame(out)
    m.to_csv(RES / "meta_dropout.csv", index=False)
    mm = m[m.analysis == "main"].copy()
    for col in ["est", "ci_lo", "ci_hi", "median_dr2", "median_null_p95"]:
        mm[col] *= 1000
    print(mm[["set", "k", "est", "ci_lo", "ci_hi", "I2", "n_pos", "n_pos_sig", "n_above_null",
              "median_null_p95"]].round(3).to_string(index=False))
    # sanity: PV1 vs 04c / 04d
    old = pd.read_csv(RES / "dropone.csv").set_index("CNTRY")
    for name, col in [("scam_out", "loss_minus_OTL_O_SCAM"), ("outside_block", "loss_minus_outside_block"),
                      ("school_block", "loss_minus_school_block"), ("teacher_info", "loss_minus_teacher_info")]:
        x = d[d.set == name].set_index("CNTRY").dr2_pv1
        diff = (x - old.loc[x.index, col]).abs()
        print(f"PV1 {name} vs 04c: max |diff| = {diff.max():.2e} (systems with diff > 1e-6: "
              f"{sorted(diff[diff > 1e-6].index)})")
    ck = cr[~cr.CNTRY.isin(EXCL)]
    print(ck.filter(regex="rho_(w|u|04c)_scam_(eval|mean_others)").median().round(3).to_string())
    print("n systems:", len(parts))


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        c = sys.argv[2]
        df = pd.read_parquet(PROC / "analysis.parquet")
        g = df[df.CNTRY == c]
        if len(g) > 5000 and not os.environ.get("DROPOUT_FORCE_CPU"):   # as 04c_dropone.py
            M.XGB_FIXED["device"] = "cuda"
        res = run_country(c, g)   # compute first, then write the cache file
        with open(CACHE / f"{c}.pkl", "wb") as f:
            pickle.dump(res, f)
    elif sys.argv[1] == "--collect":
        collect()
