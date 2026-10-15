"""Group calibration analysis of out-of-fold predictions (school-grouped CV, 03_cv.py).

For each system, model and PV: residual e = y - yhat. Calibration gap = weighted mean residual in
group A minus group B (positive = model under-predicts group A relative to B):
  gender: girls - boys;  SES: lowest - highest within-system HISEI quartile.
Also group-specific R2. Design-based SEs: JRR (75 replicate weights) conditional on the
out-of-fold predictions, combined over PVs (User Guide eq. 3.3).

Outputs: results/fairness_gaps.csv
"""
import numpy as np
import pandas as pd

from config import PROC, RES, PVS, REPW, WGT
from survey import combine_pv, jrr_var, weighted_r2

MODELS = ["RIDGE", "RLMM", "XGB", "MERF"]


def wq(x, w, q):
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / w.sum()
    return x[o][np.searchsorted(cw, q)]


def gap(e, a, b, w):
    return np.average(e[a], weights=w[a]) - np.average(e[b], weights=w[b])


def main():
    df = pd.read_parquet(PROC / "analysis.parquet",
                         columns=["CNTRY", "IDSTUD", "S_SEX", "S_HISEI", WGT] + PVS + REPW)
    oof = pd.read_parquet(RES / "oof_predictions.parquet")
    d = df.merge(oof, on=["CNTRY", "IDSTUD"], how="inner", validate="one_to_one")
    rows = []
    for c, g in d.groupby("CNTRY"):
        w = g[WGT].to_numpy(float)
        reps = [g[r].to_numpy(float) for r in REPW]
        girl, boy = (g.S_SEX == 1).to_numpy(), (g.S_SEX == 0).to_numpy()
        h = g.S_HISEI.to_numpy(float)
        ok = ~np.isnan(h)
        q1, q3 = wq(h[ok], w[ok], 0.25), wq(h[ok], w[ok], 0.75)
        low, high = ok & (h <= q1), ok & (h >= q3)
        for m in MODELS:
            res = {}
            for name, a, b in [("gender", girl, boy), ("ses", low, high)]:
                f, sv = [], []
                for j, pv in enumerate(PVS):
                    e = g[pv].to_numpy(float) - g[f"{m}_pv{j + 1}"].to_numpy(float)
                    full = gap(e, a, b, w)
                    f.append(full)
                    sv.append(jrr_var(full, [gap(e, a, b, wr) for wr in reps]))
                est, se, _, _ = combine_pv(f, sv)
                res[f"{name}_gap"], res[f"{name}_se"] = est, se
            slopes = []
            for j, pv in enumerate(PVS):
                yy, pp = g[pv].to_numpy(float), g[f"{m}_pv{j + 1}"].to_numpy(float)
                mp, my = np.average(pp, weights=w), np.average(yy, weights=w)
                slopes.append(np.average((pp - mp) * (yy - my), weights=w) / np.average((pp - mp) ** 2, weights=w))
            res["cal_slope"] = float(np.mean(slopes))
            for name, msk in [("girls", girl), ("boys", boy), ("lowSES", low), ("highSES", high)]:
                res[f"r2_{name}"] = np.mean([weighted_r2(g.loc[msk, pv].to_numpy(float),
                                                         g.loc[msk, f"{m}_pv{j + 1}"].to_numpy(float),
                                                         w[msk]) for j, pv in enumerate(PVS)])
            rows.append({"CNTRY": c, "model": m, **res})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "fairness_gaps.csv", index=False)
    s = out.groupby("model")[["gender_gap", "ses_gap", "r2_girls", "r2_boys", "r2_lowSES", "r2_highSES"]]
    print(s.median().round(2).to_string())
    out["gender_sig"] = (out.gender_gap / out.gender_se).abs() > 1.96
    out["ses_sig"] = (out.ses_gap / out.ses_se).abs() > 1.96
    print(out.groupby("model")[["gender_sig", "ses_sig"]].sum())


if __name__ == "__main__":
    main()
