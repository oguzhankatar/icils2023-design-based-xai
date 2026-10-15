"""Design-based root mean squared error of the out-of-fold predictions of XGBoost and weighted ridge.

Per system and model: weighted sqrt(mean squared out-of-fold residual) for each plausible value and
each of the 76 weight vectors (TOTWGTS + 75 JRR replicate weights), JRR x PV variance
(survey.combine_pv). The full-sample RMSE averaged over the PVs reproduces results/cv_metrics.csv
(checked). Pooled by 05_meta.reml (REML tau^2, HKSJ CI) over the 34 systems (DNW excluded).

Inputs:  data/processed/analysis.parquet, results/oof_predictions.parquet, results/cv_metrics.csv
Outputs: results/rmse_design.csv (system level), results/meta_rmse_design.csv (pooled)
Run: python 12d_rmse_design.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from config import PROC, PVS, REPW, RES, WGT
from survey import combine_pv

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("meta05", HERE / "05_meta.py")
meta05 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(meta05)
reml, EXCL = meta05.reml, meta05.EXCL

cols = ["CNTRY", "IDSTUD", WGT] + PVS + REPW
df = pd.read_parquet(PROC / "analysis.parquet", columns=cols)
oof = pd.read_parquet(RES / "oof_predictions.parquet")
d = df.merge(oof, on=["CNTRY", "IDSTUD"], how="inner", validate="one_to_one")
assert len(d) == len(df)
rm = []
for c, g in d.groupby("CNTRY"):
    W = np.column_stack([g[WGT].to_numpy(float)] + [g[r_].to_numpy(float) for r_ in REPW])
    Y = g[PVS].to_numpy(float)
    for m in ["XGB", "RIDGE"]:
        P = g[[f"{m}_pv{j + 1}" for j in range(len(PVS))]].to_numpy(float)
        vals = np.array([np.sqrt((W.T @ (Y[:, j] - P[:, j]) ** 2) / W.sum(axis=0)) for j in range(len(PVS))])
        full = vals[:, 0]
        sv = ((vals[:, 1:] - full[:, None]) ** 2).sum(axis=1)
        est, se, _, _ = combine_pv(full, sv)
        rm.append(dict(CNTRY=c, model=m, est=float(est), se=float(se)))
rm = pd.DataFrame(rm)
cvm = pd.read_csv(RES / "cv_metrics.csv")
pooled = []
for m in ["XGB", "RIDGE"]:
    x = rm[(rm.model == m) & ~rm.CNTRY.isin(EXCL)]
    # full-sample RMSE averaged over PVs reproduces cv_metrics.csv
    chk = cvm[cvm.model == m].groupby("CNTRY").rmse.mean()
    assert np.allclose(x.set_index("CNTRY").est.sort_index(), chk.loc[x.CNTRY].sort_index(), atol=1e-8), m
    r = reml(x.est, x.se ** 2)
    assert r["k"] == 34
    r.update(model=m, sys_median=float(x.est.median()), sys_min=float(x.est.min()), sys_max=float(x.est.max()))
    pooled.append(r)
rm.to_csv(RES / "rmse_design.csv", index=False)
pooled = pd.DataFrame(pooled)
pooled.to_csv(RES / "meta_rmse_design.csv", index=False)
print(pooled.round(3).to_string())
