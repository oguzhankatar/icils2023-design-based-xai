"""Sanity checks of the explanation model and local (student-level) explanations.

All analyses use PV1, the final student weight TOTWGTS and, per education system, the full-sample
tuned XGBoost hyperparameters of 04_shap_jrr.py (results/final_hparams.json). Models are fitted
exactly as the main pipeline fits the full-sample explanation model:
    models.fit_xgb(X, PV1, norm_w(TOTWGTS), best)   (CPU 'hist', seed = config.SEED)
and SHAP statistics are computed with shapstats.stats(phi, X, TOTWGTS) (IMP = weighted mean |phi|,
SLOPE = weighted Cov(x, phi) / SD(x)).

(1) Permuted-label check: PV1 is permuted across students within each system with
    numpy.random.default_rng(SEED + 1).permutation (SEED = 20231101; the generator is created afresh
    for every system, so the permutation depends only on the system's row order in
    data/processed/analysis.parquet). Weights and predictors stay attached to the students. The model
    is refitted with identical hyperparameters, weights and seed; TreeSHAP IMP is compared with the
    real model's IMP. Secondary variant (1b): the number of boosting rounds is re-selected on the
    permuted labels with the main pipeline's own early-stopping rule (models.xgb_cv_score: 3-fold
    school-grouped inner CV, folds seeded with SEED), other hyperparameters unchanged.
(2) Seed stability: the full-sample model is refitted with XGBoost seeds SEED + 0, ..., SEED + 4
    (seed SEED + 0 is the main-pipeline model); the seed only drives the row subsampling and column
    subsampling of each tree. IMP and SLOPE are computed per seed.
(3) Local explanations (one system, chosen in --local): see run_local().

Run:  python runner.py 04e_sanity_local.py 8      (one subprocess per system; cache results/sanity/)
      python 04e_sanity_local.py --collect        (results/sanity_permuted.csv, sanity_seeds.csv,
                                                   sanity_seeds_long.parquet)
      python 04e_sanity_local.py --local          (results/local_explanations.parquet + _meta.json)
"""
import itertools
import json
import pickle
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

import models as M
from config import PROC, RES, FEAT, FEATURES, FOCAL, PVS, SEED, WGT
from shapstats import stats

CACHE = RES / "sanity"
CACHE.mkdir(exist_ok=True)
PERM_SEED = SEED + 1
SEEDS = [SEED + k for k in range(5)]
F = len(FEAT)


def load_best(c):
    best = json.load(open(RES / "final_hparams.json"))[c]
    best = {k: v for k, v in best.items() if k != "inner_r2"}
    best["n_rounds"] = int(best["n_rounds"])
    best["max_depth"] = int(best["max_depth"])
    return best


def fit_phi(X, y, w, best, seed=SEED):
    """Fit as the main pipeline does (seed only changes the subsampling RNG); return booster, phi
    (n x F) and bias column."""
    old = M.XGB_FIXED["seed"]
    M.XGB_FIXED["seed"] = seed
    try:
        b = M.fit_xgb(X, y, w, best)
    finally:
        M.XGB_FIXED["seed"] = old
    contrib = b.predict(xgb.DMatrix(X), pred_contribs=True)
    return b, contrib[:, :F], contrib[:, F]


def run_country(c, g):
    t0 = time.time()
    g = g.reset_index(drop=True)
    X = g[FEAT].to_numpy(float)
    y = g[PVS[0]].to_numpy(float)
    wt = g[WGT].to_numpy(float)
    w = M.norm_w(wt)
    best = load_best(c)
    out = {"CNTRY": c, "n": len(g), "best": best}
    # seeds (SEEDS[0] == SEED is the real, main-pipeline model)
    imp_s, slo_s = [], []
    for sd in SEEDS:
        b, phi, _ = fit_phi(X, y, w, best, sd)
        a, s_, _ = stats(phi, X, wt)
        imp_s.append(a)
        slo_s.append(s_)
        if sd == SEED:
            pred = b.predict(xgb.DMatrix(X))
            out["r2_insample_real"] = 1 - np.sum(wt * (y - pred) ** 2) / np.sum(
                wt * (y - np.average(y, weights=wt)) ** 2)
    out["imp_seeds"], out["slope_seeds"] = np.array(imp_s), np.array(slo_s)
    # permuted labels
    yp = np.random.default_rng(PERM_SEED).permutation(y)
    b, phi, _ = fit_phi(X, yp, w, best, SEED)
    out["imp_perm"], out["slope_perm"], _ = stats(phi, X, wt)
    pred = b.predict(xgb.DMatrix(X))
    out["r2_insample_perm"] = 1 - np.sum(wt * (yp - pred) ** 2) / np.sum(
        wt * (yp - np.average(yp, weights=wt)) ** 2)
    # (1b) secondary variant: same hyperparameters, but the number of boosting rounds re-selected on
    # the permuted labels by the main pipeline's own rule (mean early-stopping iteration over the
    # 3-fold school-grouped inner CV of models.xgb_cv_score, folds seeded with SEED).
    s = g.IDSCHOOL.to_numpy()
    params = {k: v for k, v in best.items() if k != "n_rounds"}
    out["cv_r2_perm"], nr = M.xgb_cv_score(X, yp, w, s, params, k=3, seed=SEED)
    out["n_rounds_perm_es"] = int(nr)
    _, phi, _ = fit_phi(X, yp, w, {**params, "n_rounds": int(nr)}, SEED)
    out["imp_perm_es"], out["slope_perm_es"], _ = stats(phi, X, wt)
    out["sec"] = time.time() - t0
    print(f"{c}: total IMP real={out['imp_seeds'][0].sum():.2f} perm={out['imp_perm'].sum():.2f} "
          f"{out['sec']:.0f}s", flush=True)
    return out


def pairwise_spearman(A):
    """A: seeds x features; median and min pairwise Spearman over features finite in both seeds."""
    rs = []
    for i, j in itertools.combinations(range(len(A)), 2):
        ok = np.isfinite(A[i]) & np.isfinite(A[j])
        rs.append(spearmanr(A[i, ok], A[j, ok])[0])
    return np.median(rs), np.min(rs)


def slope_compare(real, perm, sfx):
    """SLOPE of the permuted vs the real model: ratio of sum |SLOPE| (predictors finite in both),
    max |SLOPE| among the focal predictors (permuted model), Spearman across predictors."""
    ok = np.isfinite(real) & np.isfinite(perm)
    fi = [FEAT.index(f) for f in FOCAL]
    return {f"sum_abs_slope_real": np.abs(real[ok]).sum(),
            f"sum_abs_slope_perm{sfx}": np.abs(perm[ok]).sum(),
            f"slope_ratio{sfx}": np.abs(perm[ok]).sum() / np.abs(real[ok]).sum(),
            f"max_abs_slope_focal_perm{sfx}": np.nanmax(np.abs(perm[fi])),
            f"max_abs_slope_focal_real": np.nanmax(np.abs(real[fi])),
            f"rho_slope_perm_real{sfx}": spearmanr(real[ok], perm[ok])[0],
            f"n_slope{sfx}": int(ok.sum())}


def collect():
    parts = [pickle.load(open(p, "rb")) for p in sorted(CACHE.glob("*.pkl"))]
    perm, seeds, long = [], [], []
    for o in parts:
        c = o["CNTRY"]
        real = o["imp_seeds"][0]
        perm.append({"CNTRY": c, "n": o["n"], "n_rounds": o["best"]["n_rounds"],
                     "max_depth": o["best"]["max_depth"],
                     "total_imp_real": real.sum(), "total_imp_perm": o["imp_perm"].sum(),
                     "ratio": o["imp_perm"].sum() / real.sum(),
                     "max_imp_real": real.max(), "max_imp_perm": o["imp_perm"].max(),
                     "top_feature_real": FEAT[int(np.argmax(real))],
                     "focal_imp_real": real[[FEAT.index(f) for f in FOCAL]].sum(),
                     "focal_imp_perm": o["imp_perm"][[FEAT.index(f) for f in FOCAL]].sum(),
                     "r2_insample_real": o["r2_insample_real"],
                     "r2_insample_perm": o["r2_insample_perm"],
                     "n_rounds_perm_es": o["n_rounds_perm_es"],
                     "cv_r2_perm": o["cv_r2_perm"],
                     "total_imp_perm_es": o["imp_perm_es"].sum(),
                     "ratio_es": o["imp_perm_es"].sum() / real.sum(),
                     **slope_compare(o["slope_seeds"][0], o["slope_perm"], ""),
                     **slope_compare(o["slope_seeds"][0], o["slope_perm_es"], "_es"),
                     "perm_seed": PERM_SEED})
        I, S = o["imp_seeds"], o["slope_seeds"]
        ri_med, ri_min = pairwise_spearman(I)
        rs_med, rs_min = pairwise_spearman(S)
        sd_slope = np.nanstd(S, axis=0, ddof=1)
        sd_imp = np.nanstd(I, axis=0, ddof=1)
        fi = [FEAT.index(f) for f in FOCAL]
        top10 = np.argsort(-I.mean(0))[:10]
        seeds.append({"CNTRY": c, "rho_imp_median": ri_med, "rho_imp_min": ri_min,
                      "rho_slope_median": rs_med, "rho_slope_min": rs_min,
                      "max_sd_slope_focal": np.nanmax(sd_slope[fi]),
                      "max_sd_slope_focal_feature": FEAT[fi[int(np.nanargmax(sd_slope[fi]))]],
                      "max_sd_slope_top10": np.nanmax(sd_slope[top10]),
                      "max_sd_slope_all": np.nanmax(sd_slope),
                      "max_sd_imp_top10": np.nanmax(sd_imp[top10]),
                      "max_sd_imp_all": np.nanmax(sd_imp),
                      "total_imp_sd": I.sum(1).std(ddof=1),
                      "seeds": ",".join(map(str, SEEDS))})
        for k, sd in enumerate(SEEDS):
            for j, f in enumerate(FEAT):
                long.append({"CNTRY": c, "seed": sd, "feature": f, "IMP": I[k, j],
                             "SLOPE": S[k, j], "IMP_perm": o["imp_perm"][j] if k == 0 else np.nan})
    perm = pd.DataFrame(perm)
    seeds = pd.DataFrame(seeds)
    perm.to_csv(RES / "sanity_permuted.csv", index=False)
    seeds.to_csv(RES / "sanity_seeds.csv", index=False)
    pd.DataFrame(long).to_parquet(RES / "sanity_seeds_long.parquet", index=False)
    print(perm[["CNTRY", "total_imp_real", "total_imp_perm", "ratio", "r2_insample_real",
                "r2_insample_perm", "n_rounds", "n_rounds_perm_es", "ratio_es", "cv_r2_perm",
                "slope_ratio", "slope_ratio_es", "max_abs_slope_focal_perm",
                "max_abs_slope_focal_perm_es", "rho_slope_perm_real", "rho_slope_perm_real_es"]]
          .round(3).to_string())
    print(seeds.drop(columns="seeds").round(3).to_string())
    print("n systems:", len(parts))


# ------------------------------------------------------------------ local explanations
def wmedian(x, w):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return x[o][np.searchsorted(cw, 0.5 * cw[-1])]


def run_local():
    cv = pd.read_csv(RES / "cv_metrics.csv")
    r2 = cv[cv.model == "XGB"].groupby("CNTRY").r2.mean().sort_values()
    assert len(r2) % 2 == 1
    c = r2.index[len(r2) // 2]  # the median of an odd number of systems (rank 18 of 35)
    df = pd.read_parquet(PROC / "analysis.parquet")
    g = df[df.CNTRY == c].reset_index(drop=True)
    oof = pd.read_parquet(RES / "oof_predictions.parquet")
    oof = oof[oof.CNTRY == c][["IDSTUD", "XGB_pv1"]]
    g = g.merge(oof, on="IDSTUD", how="left", validate="1:1")
    assert g.XGB_pv1.notna().all()
    X = g[FEAT].to_numpy(float)
    y = g[PVS[0]].to_numpy(float)
    wt = g[WGT].to_numpy(float)
    best = load_best(c)
    b, phi, base = fit_phi(X, y, M.norm_w(wt), best, SEED)
    full_pred = b.predict(xgb.DMatrix(X))
    assert np.allclose(phi.sum(1) + base, full_pred, atol=1e-2)
    imp, _, _ = stats(phi, X, wt)
    cached = pickle.load(open(CACHE / f"{c}.pkl", "rb"))["imp_seeds"][0] if (
        CACHE / f"{c}.pkl").exists() else None
    if cached is not None:
        assert np.allclose(imp, cached), "explanation model not reproduced"
    res = y - g.XGB_pv1.to_numpy()
    med = wmedian(y, wt)
    typ_pool = np.where(np.abs(y - med) <= 10)[0]
    picks = {"typical": int(typ_pool[np.argmin(np.abs(res[typ_pool]))]),
             "under_predicted": int(np.argmax(res)),
             "over_predicted": int(np.argmin(res))}
    rows, meta_st = [], []
    for role, i in picks.items():
        for j, f in enumerate(FEAT):
            rows.append({"CNTRY": c, "student_id": int(g.IDSTUD[i]), "IDSCHOOL": int(g.IDSCHOOL[i]),
                         "role": role, "feature": f, "value": X[i, j], "phi": phi[i, j],
                         "block": FEATURES[f][0], "label": FEATURES[f][1],
                         "base_value": float(base[i]), "pred_full": float(full_pred[i]),
                         "pred_oof": float(g.XGB_pv1[i]), "observed": float(y[i]),
                         "weight": float(wt[i])})
        top = np.argsort(-np.abs(phi[i]))[:3]
        meta_st.append({"role": role, "student_id": int(g.IDSTUD[i]), "IDSCHOOL": int(g.IDSCHOOL[i]),
                        "observed_pv1": float(y[i]), "pred_oof": float(g.XGB_pv1[i]),
                        "residual_oof": float(res[i]), "pred_full": float(full_pred[i]),
                        "base_value": float(base[i]),
                        "top3": [{"feature": FEAT[j], "value": float(X[i, j]), "phi": float(phi[i, j])}
                                 for j in top]})
    pd.DataFrame(rows).to_parquet(RES / "local_explanations.parquet", index=False)
    meta = {
        "system": c,
        "selection_rule_system": "XGBoost out-of-sample weighted R2 (mean over the five PVs, "
                                 "results/cv_metrics.csv) equal to the median of the 35 systems "
                                 "(rank 18 of 35 in ascending order)",
        "xgb_r2_mean_pv": float(r2[c]), "xgb_r2_rank_ascending": int(len(r2) // 2 + 1),
        "xgb_r2_pv1": float(cv[(cv.model == "XGB") & (cv.CNTRY == c) & (cv.pv == PVS[0])].r2.iloc[0]),
        "n_students": int(len(g)),
        "residuals": "observed PV1 minus out-of-fold XGB PV1 prediction of the nested 5-fold "
                     "school-grouped CV (03_cv.py; results/oof_predictions.parquet, column XGB_pv1)",
        "weighted_median_pv1": float(med), "n_typical_pool": int(len(typ_pool)),
        "selection_rule_students": {
            "typical": "smallest |OOF residual| among students with |PV1 - weighted median PV1| <= 10",
            "under_predicted": "largest positive OOF residual (observed > predicted)",
            "over_predicted": "largest negative OOF residual (observed < predicted)"},
        "explanation_model": "full-sample XGBoost explanation model (PV1, TOTWGTS normalised to mean 1, "
                             "final_hparams.json, seed %d, CPU hist), path-dependent TreeSHAP "
                             "contributions via XGBoost pred_contribs; phi sums to "
                             "pred_full - base_value" % SEED,
        "students": meta_st,
    }
    json.dump(meta, open(RES / "local_explanations_meta.json", "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        c = sys.argv[2]
        df = pd.read_parquet(PROC / "analysis.parquet")
        res = run_country(c, df[df.CNTRY == c])
        with open(CACHE / f"{c}.pkl", "wb") as f:
            pickle.dump(res, f)
    elif sys.argv[1] == "--collect":
        collect()
    elif sys.argv[1] == "--local":
        run_local()
