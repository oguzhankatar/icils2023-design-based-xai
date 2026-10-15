"""Paired differences of the group calibration gap between each model and XGBoost.

gap_model - gap_XGB on the same students, per system with JRR x PV SEs (out-of-fold predictions held
fixed) and pooled by REML/HKSJ over the 34 systems (DNW excluded), for the groupings gender and
ses_hisei. Computed with the functions of 12_performance_contrasts.py (groups, wmeans, combine) and
05_meta.reml, i.e. exactly as the RIDGE-XGB rows of results/meta_fairness_extended.csv, which are
reproduced as a check. Gap sign as in 12_performance_contrasts.py: residual (PV - prediction) of group
A minus group B, so positive = the model under-predicts A (girls; lowest HISEI quartile) relative to B.

Inputs:  data/processed/analysis.parquet, results/oof_predictions.parquet,
         results/meta_fairness_extended.csv
Outputs: results/gap_contrasts_system.csv, results/gap_contrasts_pooled.csv
Run: python 12c_gap_contrasts.py      (after 12_performance_contrasts.py)
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from config import PROC, RES, PVS, REPW, WGT

HERE = Path(__file__).resolve().parent
GAP_VS = "XGB"          # reference model of the paired calibration-gap differences
GAP_GROUPINGS = ("gender", "ses_hisei")


def gap_contrasts():
    """Returns (per_system DataFrame, pooled DataFrame) and writes both to results/."""
    ps_, pp_ = RES / "gap_contrasts_system.csv", RES / "gap_contrasts_pooled.csv"

    def load(name, fn):
        spec = importlib.util.spec_from_file_location(name, HERE / fn)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    rp = load("perf12", "12_performance_contrasts.py")
    cols = ["CNTRY", "IDSTUD", "S_SEX", "S_HISEI", "S_HOMLIT", "S_HISCED", "S_IMMBGR", "S_TLANG",
            WGT] + PVS + REPW
    df = pd.read_parquet(PROC / "analysis.parquet", columns=cols)
    oof = pd.read_parquet(RES / "oof_predictions.parquet")
    d = df.merge(oof, on=["CNTRY", "IDSTUD"], how="inner", validate="one_to_one")
    others = [m for m in rp.MODELS if m != GAP_VS]
    rows = []
    for c, g in d.groupby("CNTRY"):
        g = g.reset_index(drop=True)
        w = g[WGT].to_numpy(float)
        W = np.column_stack([w] + [g[r].to_numpy(float) for r in REPW])
        Y = g[PVS].to_numpy(float)
        P = {m: g[[f"{m}_pv{j + 1}" for j in range(rp.NP)]].to_numpy(float) for m in rp.MODELS}
        grp = rp.groups(g, w)
        for gname in GAP_GROUPINGS:
            A, B, _ = grp[gname]
            if min(A.sum(), B.sum()) < rp.MIN_N:
                continue
            for m in others:
                dd = (Y - P[m]) - (Y - P[GAP_VS])
                gd = np.array([rp.wmeans(W, dd[:, j], A) - rp.wmeans(W, dd[:, j], B)
                               for j in range(rp.NP)])
                rows.append(dict(CNTRY=c, grouping=gname, model=m, ref=GAP_VS, **rp.combine(gd)))
    sysd = pd.DataFrame(rows)
    sysd["z"] = sysd.est / sysd.se
    pooled = []
    x = sysd[~sysd.CNTRY.isin(rp.EXCL)]
    for (gname, m), gg in x.groupby(["grouping", "model"]):
        r = rp.reml(gg.est.to_numpy(), gg.se.to_numpy() ** 2)
        r.update(grouping=gname, model=m, ref=GAP_VS, n_pos=int((gg.est > 0).sum()),
                 n_neg=int((gg.est < 0).sum()))
        pooled.append(r)
    pooled = pd.DataFrame(pooled)
    # check against the RIDGE-XGB rows of meta_fairness_extended.csv
    mf = pd.read_csv(RES / "meta_fairness_extended.csv")
    mf = mf[(mf.analysis == "main") & (mf.model == "RIDGE-" + GAP_VS)].set_index("grouping")
    for gname in GAP_GROUPINGS:
        mine = pooled[(pooled.grouping == gname) & (pooled.model == "RIDGE")].iloc[0]
        assert abs(mine.est - mf.loc[gname, "est"]) < 1e-8 and abs(mine.ci_lo - mf.loc[gname, "ci_lo"]) < 1e-8, gname
    sysd.to_csv(ps_, index=False)
    pooled.to_csv(pp_, index=False)
    return sysd, pooled


if __name__ == "__main__":
    s, p = gap_contrasts()
    print(p.to_string())
