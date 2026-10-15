"""Noise floor of the in-sample
SHAP importance statistics, missing-branch attribution and cross-fitted (out-of-fold) SHAP.

All model fits use, per education system, the full-sample tuned XGBoost hyperparameters and number
of boosting rounds of 04_shap_jrr.py (results/final_hparams.json) and are fitted exactly as the
full-sample explanation model:  models.fit_xgb(X, y, norm_w(TOTWGTS), best)  (CPU 'hist', 4 threads,
XGBoost seed = config.SEED). Statistics: shapstats.stats(phi, X, TOTWGTS)
(IMP = weighted mean |phi_j|, SLOPE, BLOCK = weighted mean |sum_{j in block} phi_j|).

(1) Clustering-preserving permutation null (PV1, TOTWGTS). PV1 is decomposed into TOTWGTS-weighted
    school means m_j and within-school deviations d_ij = y_ij - m_j. The school means are permuted
    across the schools of the system and the deviations are permuted among the students of each
    school: y*_ij = m_{pi(j)} + d_{sigma_j(i)}. The ICILS 2023 public-use files contain no explicit
    stratum identifier (only the JRR variance zones JKZONES, which pair two schools and are too
    small to permute within), so school means are permuted within the system. Predictors and weights
    stay attached to the students; the between-school variance share of y is therefore preserved,
    unlike in the student-level permutation of 04e_sanity_local.py. Replicates r = 0..4 use
    numpy.random.default_rng(SEED + 1000 + r) (SEED = 20231101, i.e. seeds 20232101..20232105),
    created afresh for every system; school order = numpy.unique(IDSCHOOL), rows in
    data/processed/analysis.parquet order. Null IMP/BLOCK = mean over the 5 replicates.
    Null-calibrated statistic = real - null, with the real statistic = the PV-averaged
    full-sample estimate (results/shap_country_stats.parquet; reproduced here and checked).
    Meta-analysis of calibrated BLOCK across 34 systems (DNW excluded): 05_meta.reml with variance
    var_main(real BLOCK) + across-replicate variance of the null (ddof = 1) / 5, where var_main is
    the primary variance of 05_meta.load_stats (JRR sampling variance + 1.2 x between-PV variance);
    comparison version with the PSU-bootstrap variance var_boot (suffix _boot). Size-adjusted
    BLOCK = BLOCK / k (k = number of predictors in the block).
    Outputs: results/noise_floor_block.csv, results/noise_floor_imp.csv,
             results/meta_BLOCK_calibrated.csv, results/meta_BLOCK_calibrated_boot.csv
(4) Missing-branch attribution (real full-sample models, all 5 PVs, TOTWGTS). For the two OTL blocks
    and the 8 focal predictors, IMP_j = sum_i w_i |phi_ij| / sum_i w_i is split into the part from
    students with the item missing and the part from students with the item observed (the two parts
    sum to IMP); the same for BLOCK with 'missing' = at least one item of the block missing ('any')
    and = all items missing ('all'). Averaged over the 5 PVs (so the total equals the main-analysis IMP /
    BLOCK). Output: results/otl_missing_attribution.csv
(6) Cross-fitted SHAP (PV1): school-grouped 5-fold split models.group_folds(IDSCHOOL, 5, SEED) (the
    outer folds of 03_cv.py, i.e. of results/oof_predictions.parquet); a model with the full-sample
    hyperparameters is fitted on 4 folds and TreeSHAP is computed for the held-out fold, giving an
    out-of-fold phi (and base value) for every student. IMP, SLOPE, BLOCK from the out-of-fold phi
    vs the in-sample PV1 statistics. Pooled out-of-fold BLOCK: 05_meta.reml with the in-sample
    var_main (JRR) as an approximate sampling variance (_boot: in-sample var_boot).
    Outputs: results/crossfit_shap_stats.csv, results/meta_BLOCK_crossfit.csv,
             results/meta_BLOCK_crossfit_boot.csv
    --local-oof: for the system and the three students of results/local_explanations_meta.json,
    out-of-fold TreeSHAP (a) of the exact 03_cv.py outer-fold XGBoost model (fold-specific tuned
    hyperparameters from results/cv_hparams.json; its prediction reproduces the stored out-of-fold
    prediction XGB_pv1, checked) and (b) of the cross-fitted model of (6).
    Output: results/local_explanations_oof.parquet

Run:  python runner.py 11_noise_floor.py 6      (one subprocess per system; cache results/noise_floor/)
      python 11_noise_floor.py --collect
      python 11_noise_floor.py --local-oof
Does not modify any other script or result file.
"""
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

import models as M
from config import PROC, RES, FEAT, FEATURES, FOCAL, PVS, SEED, WGT
from shapstats import BLK_NAMES, BLOCKS, stats

CACHE = RES / "noise_floor"
CACHE.mkdir(exist_ok=True)
F = len(FEAT)
N_NULL = 5
NULL_SEEDS = [SEED + 1000 + r for r in range(N_NULL)]
OTL_BLOCKS = ["OTL info (school)", "OTL info (outside)"]


def load_best(c):
    best = json.load(open(RES / "final_hparams.json"))[c]
    best = {k: v for k, v in best.items() if k != "inner_r2"}
    best["n_rounds"], best["max_depth"] = int(best["n_rounds"]), int(best["max_depth"])
    return best


def contribs(b, X):
    c = b.predict(xgb.DMatrix(X), pred_contribs=True)
    return c[:, :F], c[:, F]


def perm_clustered(y, s, wt, rng):
    """Permute weighted school means across schools and within-school deviations within schools."""
    sch, idx = np.unique(s, return_inverse=True)
    m = np.bincount(idx, weights=wt * y) / np.bincount(idx, weights=wt)
    d = y - m[idx]
    m_new = m[rng.permutation(len(sch))]
    d_new = np.empty_like(d)
    order = np.argsort(idx, kind="stable")
    ends = np.cumsum(np.bincount(idx))
    start = 0
    for end in ends:
        rows = order[start:end]
        d_new[rows] = d[rng.permutation(rows)]
        start = end
    return m_new[idx] + d_new


def between_share(y, s, wt):
    sch, idx = np.unique(s, return_inverse=True)
    m = np.bincount(idx, weights=wt * y) / np.bincount(idx, weights=wt)
    mu = np.average(y, weights=wt)
    return np.average((m[idx] - mu) ** 2, weights=wt) / np.average((y - mu) ** 2, weights=wt)


def wr2(y, p, w):
    return 1 - np.sum(w * (y - p) ** 2) / np.sum(w * (y - np.average(y, weights=w)) ** 2)


def attribution(phi, X, wt):
    """Missing/observed split of IMP (focal predictors) and BLOCK (two OTL blocks)."""
    sw = wt.sum()
    out = {}
    for f in FOCAL:
        j = FEAT.index(f)
        a = np.abs(phi[:, j])
        miss = np.isnan(X[:, j])
        out[("IMP", f, "item")] = dict(
            total=(wt * a).sum() / sw, from_missing=(wt * a)[miss].sum() / sw,
            from_observed=(wt * a)[~miss].sum() / sw, pct_missing=100 * wt[miss].sum() / sw,
            mean_phi_missing=np.average(phi[miss, j], weights=wt[miss]) if miss.any() else np.nan)
    for b in OTL_BLOCKS:
        idx = BLOCKS[b]
        a = np.abs(phi[:, idx].sum(1))
        nm = np.isnan(X[:, idx]).sum(1)
        for defn, miss in [("any", nm > 0), ("all", nm == len(idx))]:
            out[("BLOCK", b, defn)] = dict(
                total=(wt * a).sum() / sw, from_missing=(wt * a)[miss].sum() / sw,
                from_observed=(wt * a)[~miss].sum() / sw, pct_missing=100 * wt[miss].sum() / sw,
                mean_phi_missing=np.average(phi[miss][:, idx].sum(1), weights=wt[miss])
                if miss.any() else np.nan)
    return out


def run_country(c, g):
    t0 = time.time()
    g = g.reset_index(drop=True)
    X = g[FEAT].to_numpy(float)
    s = g.IDSCHOOL.to_numpy()
    wt = g[WGT].to_numpy(float)
    w = M.norm_w(wt)
    best = load_best(c)
    out = {"CNTRY": c, "n": len(g), "n_schools": int(len(np.unique(s))), "best": best}
    # real full-sample models, all PVs
    imp_r, slo_r, blk_r, att = [], [], [], []
    for p, pv in enumerate(PVS):
        y = g[pv].to_numpy(float)
        b = M.fit_xgb(X, y, w, best)
        phi, _ = contribs(b, X)
        a, sl, bk = stats(phi, X, wt)
        imp_r.append(a), slo_r.append(sl), blk_r.append(bk)
        att.append(attribution(phi, X, wt))
        if p == 0:
            out["r2_insample_real_pv1"] = wr2(y, b.predict(xgb.DMatrix(X)), wt)
    out["imp_real"], out["slope_real"], out["blk_real"] = map(np.array, (imp_r, slo_r, blk_r))
    out["attribution"] = {k: {m: np.mean([a[k][m] for a in att]) for m in att[0][k]} for k in att[0]}
    # clustering-preserving permutation null, PV1
    y1 = g[PVS[0]].to_numpy(float)
    out["between_share_real"] = between_share(y1, s, wt)
    imp_n, blk_n, slo_n, bs, r2n = [], [], [], [], []
    for sd in NULL_SEEDS:
        yp = perm_clustered(y1, s, wt, np.random.default_rng(sd))
        b = M.fit_xgb(X, yp, w, best)
        phi, _ = contribs(b, X)
        a, sl, bk = stats(phi, X, wt)
        imp_n.append(a), slo_n.append(sl), blk_n.append(bk)
        bs.append(between_share(yp, s, wt))
        r2n.append(wr2(yp, b.predict(xgb.DMatrix(X)), wt))
    out["imp_null"], out["slope_null"], out["blk_null"] = map(np.array, (imp_n, slo_n, blk_n))
    out["between_share_null"], out["r2_insample_null"] = np.array(bs), np.array(r2n)
    # cross-fitted SHAP, PV1
    phi_oof = np.full((len(g), F), np.nan)
    base_oof = np.full(len(g), np.nan)
    pred_oof = np.full(len(g), np.nan)
    fold = np.full(len(g), -1)
    for k, (tr, te) in enumerate(M.group_folds(s, 5, SEED)):
        b = M.fit_xgb(X[tr], y1[tr], w[tr], best)
        phi_oof[te], base_oof[te] = contribs(b, X[te])
        pred_oof[te] = b.predict(xgb.DMatrix(X[te]))
        fold[te] = k
    assert np.allclose(phi_oof.sum(1) + base_oof, pred_oof, atol=1e-2)
    out["imp_oof"], out["slope_oof"], out["blk_oof"] = stats(phi_oof, X, wt)
    out["r2_oof_pv1"] = wr2(y1, pred_oof, wt)
    out["oof"] = pd.DataFrame({"IDSTUD": g.IDSTUD.to_numpy(), "fold": fold, "base": base_oof,
                               "pred": pred_oof}).assign(**{f: phi_oof[:, j].astype("float32")
                                                             for j, f in enumerate(FEAT)})
    out["sec"] = time.time() - t0
    print(f"{c}: BLOCK real(PV1) {np.round(out['blk_real'][0], 2)} null {np.round(out['blk_null'].mean(0), 2)} "
          f"R2 oof={out['r2_oof_pv1']:.3f} {out['sec']:.0f}s", flush=True)
    return out


# ------------------------------------------------------------------ collect
def load_meta05():
    spec = importlib.util.spec_from_file_location("meta05", Path(__file__).parent / "05_meta.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def collect():
    meta05 = load_meta05()
    reml, EXCL = meta05.reml, meta05.EXCL
    parts = [pickle.load(open(p, "rb")) for p in sorted(CACHE.glob("*.pkl"))]
    d = meta05.load_stats()
    real = d.set_index(["CNTRY", "stat", "feature"])
    K = {b: len(ix) for b, ix in BLOCKS.items()}
    brows, irows, arows, crows = [], [], [], []
    maxdiff = 0.0
    for o in parts:
        c = o["CNTRY"]
        bn, ir = o["blk_null"], o["imp_null"]
        for j, b in enumerate(BLK_NAMES):
            r = real.loc[(c, "BLOCK", b)]
            maxdiff = max(maxdiff, abs(o["blk_real"][:, j].mean() - r.est))
            nm, nv = bn[:, j].mean(), bn[:, j].var(ddof=1)
            brows.append({"CNTRY": c, "block": b, "k": K[b], "block_real": r.est,
                          "block_real_pv1": o["blk_real"][0, j], "block_null": nm,
                          "block_null_sd": np.sqrt(nv), "null_over_real": nm / r.est,
                          "null_over_real_pv1": nm / o["blk_real"][0, j],
                          "block_cal": r.est - nm, "var_main_real": r.var_main,
                          "var_boot_real": r.var_boot, "var_null_rep": nv,
                          "var_cal": r.var_main + nv / N_NULL, "var_cal_boot": r.var_boot + nv / N_NULL,
                          "block_real_per_k": r.est / K[b], "block_null_per_k": nm / K[b],
                          "block_cal_per_k": (r.est - nm) / K[b],
                          **{f"block_null_rep{q + 1}": bn[q, j] for q in range(N_NULL)},
                          "between_share_real": o["between_share_real"],
                          "between_share_null_mean": o["between_share_null"].mean(),
                          "null_seeds": ",".join(map(str, NULL_SEEDS))})
        for j, f in enumerate(FEAT):
            r = real.loc[(c, "IMP", f)]
            maxdiff = max(maxdiff, abs(o["imp_real"][:, j].mean() - r.est))
            nm, nv = ir[:, j].mean(), ir[:, j].var(ddof=1)
            irows.append({"CNTRY": c, "feature": f, "block": FEATURES[f][0], "imp_real": r.est,
                          "imp_real_pv1": o["imp_real"][0, j], "imp_null": nm,
                          "imp_null_sd": np.sqrt(nv),
                          "null_over_real": nm / r.est if r.est > 0 else np.nan,
                          "imp_cal": r.est - nm, "var_main_real": r.var_main,
                          "var_boot_real": r.var_boot, "var_cal": r.var_main + nv / N_NULL,
                          "var_cal_boot": r.var_boot + nv / N_NULL,
                          **{f"imp_null_rep{q + 1}": ir[q, j] for q in range(N_NULL)}})
        for (stat, unit, defn), v in o["attribution"].items():
            arows.append({"CNTRY": c, "stat": stat, "unit": unit, "missing_def": defn, **v,
                          "share_from_missing": v["from_missing"] / v["total"] if v["total"] > 0 else np.nan,
                          "intensity_ratio_missing_vs_observed":
                              (v["from_missing"] / v["pct_missing"]) / (v["from_observed"] / (100 - v["pct_missing"]))
                              if 0 < v["pct_missing"] < 100 and v["from_observed"] > 0 else np.nan})
        for stat, A, B, labs in [("IMP", o["imp_real"][0], o["imp_oof"], FEAT),
                                 ("SLOPE", o["slope_real"][0], o["slope_oof"], FEAT),
                                 ("BLOCK", o["blk_real"][0], o["blk_oof"], BLK_NAMES)]:
            for j, lab in enumerate(labs):
                crows.append({"CNTRY": c, "stat": stat, "feature": lab,
                              "block": FEATURES[lab][0] if lab in FEATURES else lab,
                              "insample_pv1": A[j], "oof_pv1": B[j],
                              "ratio_oof_over_in": B[j] / A[j] if stat != "SLOPE" and A[j] > 0 else np.nan,
                              "r2_insample_pv1": o["r2_insample_real_pv1"], "r2_oof_pv1": o["r2_oof_pv1"]})
    print(f"max |reproduced PV-avg real - shap_country_stats est| = {maxdiff:.2e}")
    nb, ni, na, cs = map(pd.DataFrame, (brows, irows, arows, crows))
    nb.to_csv(RES / "noise_floor_block.csv", index=False)
    ni.to_csv(RES / "noise_floor_imp.csv", index=False)
    na.to_csv(RES / "otl_missing_attribution.csv", index=False)
    # SLOPE agreement across predictors per system
    sl = cs[cs.stat == "SLOPE"].dropna(subset=["insample_pv1", "oof_pv1"])
    cor = sl.groupby("CNTRY").apply(lambda x: pd.Series({
        "slope_pearson_oof_in": np.corrcoef(x.insample_pv1, x.oof_pv1)[0, 1],
        "slope_spearman_oof_in": x[["insample_pv1", "oof_pv1"]].rank().corr().iloc[0, 1]}),
        include_groups=False).reset_index()
    cs = cs.merge(cor, on="CNTRY", how="left")
    cs.to_csv(RES / "crossfit_shap_stats.csv", index=False)
    # meta-analyses: primary (05_meta var_main = JRR) and PSU-bootstrap comparison (_boot)
    x = nb[~nb.CNTRY.isin(EXCL)]
    mall = pd.read_csv(RES / "meta_BLOCK.csv")
    vb_all = d[d.stat == "BLOCK"].set_index(["CNTRY", "feature"])
    for sfx, vcal, vreal, ana in [("", "var_cal", "var_main", "main"), ("_boot", "var_cal_boot", "var_boot", "boot")]:
        mreal = mall[mall.analysis == ana].set_index("feature")
        rows = []
        for b, gb in x.groupby("block"):
            r = reml(gb.block_cal, gb[vcal])
            z = gb.block_cal / np.sqrt(gb[vcal])
            k = K[b]
            r.update(block=b, variance=vreal, k_pred=k, est_per_k=r["est"] / k, ci_lo_per_k=r["ci_lo"] / k,
                     ci_hi_per_k=r["ci_hi"] / k, pooled_real=mreal.loc[b].est,
                     pooled_real_per_k=mreal.loc[b].est / k, mean_null=gb.block_null.mean(),
                     median_null=gb.block_null.median(), median_null_over_real=gb.null_over_real.median(),
                     q1_null_over_real=gb.null_over_real.quantile(.25),
                     q3_null_over_real=gb.null_over_real.quantile(.75),
                     n_cal_pos=int((gb.block_cal > 0).sum()), n_cal_sig=int((z > 1.96).sum()),
                     above_floor=bool(r["ci_lo"] > 0))
            rows.append(r)
        mc = pd.DataFrame(rows).sort_values("pooled_real", ascending=False)
        mc.to_csv(RES / f"meta_BLOCK_calibrated{sfx}.csv", index=False)
        print(f"calibrated BLOCK, variance {vreal}")
        print(mc[["block", "k", "pooled_real", "mean_null", "median_null_over_real", "est", "ci_lo", "ci_hi",
                  "est_per_k", "n_cal_sig", "above_floor"]].round(3).to_string(index=False))
        # pooled out-of-fold BLOCK (approximate variance: in-sample variance of the same version)
        cb = cs[(cs.stat == "BLOCK") & ~cs.CNTRY.isin(EXCL)].copy()
        cb["var_approx"] = [vb_all.loc[(c, b), vreal] for c, b in zip(cb.CNTRY, cb.feature)]
        rows = []
        for b, gb in cb.groupby("feature"):
            r = reml(gb.oof_pv1, gb.var_approx)
            r.update(block=b, variance=vreal, median_ratio_oof_over_in=gb.ratio_oof_over_in.median(),
                     pooled_real=mreal.loc[b].est)
            rows.append(r)
        mo = pd.DataFrame(rows).sort_values("pooled_real", ascending=False)
        mo.to_csv(RES / f"meta_BLOCK_crossfit{sfx}.csv", index=False)
        print(mo[["block", "pooled_real", "est", "ci_lo", "ci_hi", "median_ratio_oof_over_in"]].round(3)
              .to_string(index=False))
    fo = cs[(cs.stat == "IMP") & cs.feature.isin(FOCAL) & ~cs.CNTRY.isin(EXCL)]
    print(fo.groupby("feature").ratio_oof_over_in.median().round(3).to_string())
    print("SLOPE corr oof vs in (median):", cor[~cor.CNTRY.isin(EXCL)].median(numeric_only=True).round(3).to_dict())
    print(na[~na.CNTRY.isin(EXCL)].groupby(["stat", "unit", "missing_def"])[
        ["pct_missing", "share_from_missing", "intensity_ratio_missing_vs_observed"]].median().round(3).to_string())
    print("n systems:", len(parts))


# ------------------------------------------------------------------ local explanations, out-of-fold
def local_oof():
    meta = json.load(open(RES / "local_explanations_meta.json"))
    c = meta["system"]
    df = pd.read_parquet(PROC / "analysis.parquet")
    g = df[df.CNTRY == c].reset_index(drop=True)
    X = g[FEAT].to_numpy(float)
    y = g[PVS[0]].to_numpy(float)
    s = g.IDSCHOOL.to_numpy()
    w = M.norm_w(g[WGT])
    oofp = pd.read_parquet(RES / "oof_predictions.parquet")
    oofp = oofp[oofp.CNTRY == c].set_index("IDSTUD").XGB_pv1
    cvhp = json.load(open(RES / "cv_hparams.json"))[c]
    best = load_best(c)
    folds = M.group_folds(s, 5, SEED)
    fold_of = np.empty(len(g), int)
    for k, (tr, te) in enumerate(folds):
        fold_of[te] = k
    cf = pickle.load(open(CACHE / f"{c}.pkl", "rb"))["oof"].set_index("IDSTUD")
    rows = []
    for st in meta["students"]:
        i = int(np.where(g.IDSTUD.to_numpy() == st["student_id"])[0][0])
        k = fold_of[i]
        tr = folds[k][0]
        assert i not in set(tr)
        hp = {**cvhp[k]["xgb"]}
        hp["n_rounds"], hp["max_depth"] = int(hp["n_rounds"]), int(hp["max_depth"])
        b = M.fit_xgb(X[tr], y[tr], w[tr], hp)
        phi, base = contribs(b, X[i:i + 1])
        pred = float(b.predict(xgb.DMatrix(X[i:i + 1]))[0])
        oof_cv = float(oofp.loc[st["student_id"]])
        assert abs(pred - oof_cv) < 1e-2, (pred, oof_cv)
        assert abs(phi.sum() + base[0] - pred) < 1e-2
        variants = {"cv_model": (phi[0], float(base[0]), pred,
                                       "03_cv.py outer-fold XGBoost model (fold-specific tuned "
                                       "hyperparameters, results/cv_hparams.json)"),
                    "crossfit_fullsample_hp": (cf.loc[st["student_id"], FEAT].to_numpy(float),
                                               float(cf.loc[st["student_id"], "base"]),
                                               float(cf.loc[st["student_id"], "pred"]),
                                               "cross-fitted model with the full-sample tuned "
                                               "hyperparameters (11_noise_floor.py, part 6)")}
        for vname, (ph, bs, pr, desc) in variants.items():
            for j, f in enumerate(FEAT):
                rows.append({"CNTRY": c, "student_id": st["student_id"], "IDSCHOOL": st["IDSCHOOL"],
                             "role": st["role"], "variant": vname, "variant_desc": desc,
                             "fold": int(k), "feature": f, "value": X[i, j], "phi": float(ph[j]),
                             "block": FEATURES[f][0], "label": FEATURES[f][1], "base_value": bs,
                             "pred_oof_model": pr, "pred_oof_cv": oof_cv, "observed": float(y[i]),
                             "weight": float(g[WGT][i])})
            print(f"{st['role']:16s} {vname:24s} base={bs:7.2f} sum(phi)={sum(ph):7.2f} "
                  f"pred={pr:7.2f} cv_oof={oof_cv:7.2f} top3="
                  f"{[FEAT[j] for j in np.argsort(-np.abs(ph))[:3]]}")
    pd.DataFrame(rows).to_parquet(RES / "local_explanations_oof.parquet", index=False)


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        c = sys.argv[2]
        df = pd.read_parquet(PROC / "analysis.parquet")
        res = run_country(c, df[df.CNTRY == c])   # compute first, then write the cache file
        with open(CACHE / f"{c}.pkl", "wb") as f:
            pickle.dump(res, f)
    elif sys.argv[1] == "--collect":
        collect()
    elif sys.argv[1] == "--local-oof":
        local_oof()
