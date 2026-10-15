"""Reproduce published ICILS 2023 country CIL means/SEs and gender gaps to validate
weights, replicate weights and plausible-value handling."""
import numpy as np
import pandas as pd

from config import PROC, RES, PVS, REPW, WGT
from survey import jrr_var, combine_pv, wmean

df = pd.read_parquet(PROC / "analysis.parquet")
rows = []
for c, g in df.groupby("CNTRY"):
    g = g[g[WGT] > 0]
    full = np.array([wmean(g[p].values, g[WGT].values) for p in PVS])
    sv = []
    for p in PVS:
        reps = [wmean(g[p].values, g[r].values) for r in REPW]
        sv.append(jrr_var(wmean(g[p].values, g[WGT].values), reps))
    est, se, u, b = combine_pv(full, sv)
    # gender gap (girls - boys), dependent samples via replicate weights
    girl = (g.S_SEX == 1).values
    boy = (g.S_SEX == 0).values
    gf, gsv = [], []
    for p in PVS:
        y = g[p].values
        d = lambda w: wmean(y[girl], w[girl]) - wmean(y[boy], w[boy])
        f = d(g[WGT].values)
        gf.append(f)
        gsv.append(jrr_var(f, [d(g[r].values) for r in REPW]))
    gest, gse, _, _ = combine_pv(gf, gsv)
    rows.append({"CNTRY": c, "n": len(g), "n_schools": g.IDSCHOOL.nunique(), "CIL_mean": est, "CIL_se": se,
                 "gap_girls_minus_boys": gest, "gap_se": gse, "pct_girls": 100 * wmean(girl, g[WGT].values)})
out = pd.DataFrame(rows).round(2)
out.to_csv(RES / "validation_country_means.csv", index=False)
print(out.to_string(index=False))
print("PV missing:", df[PVS].isna().sum().sum(), " zero/NA weights:", int((df[WGT].fillna(0) <= 0).sum()))
