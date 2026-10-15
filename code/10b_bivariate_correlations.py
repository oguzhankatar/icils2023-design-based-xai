"""Bivariate correlations of specialist-application use and specialist ICT self-efficacy.

Per system: weighted (TOTWGTS) Pearson correlations on pairwise-complete cases, with CIL averaged
over the five correlations of the plausible values. 34 systems (North Rhine-Westphalia excluded);
the medians across systems are printed.

Input:  data/processed/analysis.parquet
Output: results/bivariate_specialist_corr.csv
Run: python 10b_bivariate_correlations.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
d = pd.read_parquet(ROOT / "data" / "processed" / "analysis.parquet")
PVS = [f"PV{i}CIL" for i in range(1, 6)]


def wcorr(x, y, w):
    m = x.notna() & y.notna() & w.notna()
    x, y, w = x[m].to_numpy(float), y[m].to_numpy(float), w[m].to_numpy(float)
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    return cov / np.sqrt(np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w))


rows = []
for c, g in d.groupby("CNTRY"):
    w = g.TOTWGTS
    rows.append(dict(
        CNTRY=c,
        spec_cil=np.mean([wcorr(g.S_SPECLASS, g[p], w) for p in PVS]),
        spec_gen=wcorr(g.S_SPECLASS, g.S_GENCLASS, w),
        speceff_cil=np.mean([wcorr(g.S_SPECEFF, g[p], w) for p in PVS]),
        speceff_geneff=wcorr(g.S_SPECEFF, g.S_GENEFF, w)))
r = pd.DataFrame(rows)
r = r[r.CNTRY != "DNW"]   # 34 systems, as in the pooled analyses (benchmarking participant excluded)
r.to_csv(ROOT / "results" / "bivariate_specialist_corr.csv", index=False)
print("medians across systems:", r.drop(columns="CNTRY").median().round(2).to_dict(),
      "| systems with a negative spec_cil:", int((r.spec_cil < 0).sum()))
