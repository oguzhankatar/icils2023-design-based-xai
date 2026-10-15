"""Random-effects meta-analysis of country-level SHAP statistics.

For each feature (SLOPE, IMP) and block (BLOCK): REML estimate of tau^2 (Fisher scoring,
Viechtbauer 2005), pooled mean with Hartung-Knapp-Sidik-Jonkman (HKSJ) 95% CI, 95% prediction
interval (t_{k-2}), I^2 (Higgins & Thompson 2002). The North Rhine-Westphalia benchmark (DNW)
is excluded because its population is part of Germany (DEU). Sensitivity: excluding Romania (ROU),
which is excluded from the ICILS 2023 international average.
Sampling variance (main): the conventional JRR of the ICILS design (75 zones x 5 PVs
plus imputation variance, Eq. 4). A coverage simulation (code/13_coverage_sim.py) showed that for SHAP
statistics of refitted XGBoost the JRR overstates the true standard error (conservative) whereas the
with-replacement PSU (school) bootstrap understates it (liberal); the JRR is therefore primary and the
bootstrap is reported as the liberal bound ('boot'). 'jrr' is an alias of 'main'.
Country-level tests: z = est/se, Benjamini-Hochberg FDR across all feature x country tests.

Outputs: results/meta_{SLOPE,IMP,BLOCK}.csv, results/country_slope_tests.csv
"""
import numpy as np
import pandas as pd
from scipy import stats as st

from config import RES, FEATURES

EXCL = ["DNW"]


def reml(y, v, tol=1e-10, it=500):
    y, v = np.asarray(y, float), np.asarray(v, float)
    k = len(y)
    w = 1 / v
    mu = np.sum(w * y) / w.sum()
    tau2 = max(0.0, (np.sum(w * (y - mu) ** 2) - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    for _ in range(it):
        w = 1 / (v + tau2)
        mu = np.sum(w * y) / w.sum()
        new = max(0.0, np.sum(w ** 2 * ((y - mu) ** 2 - v)) / np.sum(w ** 2) + 1 / w.sum())
        if abs(new - tau2) < tol:
            tau2 = new
            break
        tau2 = new
    w = 1 / (v + tau2)
    mu = np.sum(w * y) / w.sum()
    # HKSJ standard error
    q = np.sum(w * (y - mu) ** 2) / (k - 1)
    wf_ = 1 / v
    mu_f = np.sum(wf_ * y) / wf_.sum()
    q_df = np.sum(wf_ * (y - mu_f) ** 2) / (k - 1)   # Cochran's Q / (k-1)
    se_hk = np.sqrt(q / w.sum())
    se_hk = max(se_hk, np.sqrt(1 / w.sum()))  # truncated HKSJ (never narrower than standard)
    tcrit = st.t.ppf(0.975, k - 1)
    wf = 1 / v
    s2 = (k - 1) * wf.sum() / (wf.sum() ** 2 - (wf ** 2).sum())
    i2 = tau2 / (tau2 + s2)
    tp = st.t.ppf(0.975, k - 2)
    pi = tp * np.sqrt(tau2 + se_hk ** 2)
    return dict(k=k, est=mu, se=se_hk, ci_lo=mu - tcrit * se_hk, ci_hi=mu + tcrit * se_hk,
                p=2 * st.t.sf(abs(mu / se_hk), k - 1), tau=np.sqrt(tau2), I2=100 * i2,
                pi_lo=mu - pi, pi_hi=mu + pi, Q_df=q_df)


def bh(p):
    p = np.asarray(p)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def load_stats():
    """Country-level estimates with two variance versions:
    var_jrr  = JRR sampling variance (mean over PVs) + (1 + 1/5) * between-PV variance
    var_boot = PSU-bootstrap sampling variance (PV1) + (1 + 1/5) * between-PV variance (liberal bound)
    var_main = var_jrr, the primary (main) variance (see module docstring)."""
    d = pd.read_parquet(RES / "shap_country_stats.parquet")
    cb = pd.read_parquet(RES / "clusterboot_stats.parquet")
    cbb = pd.read_parquet(RES / "clusterboot_blocks.parquet")
    cbl = pd.concat([
        cb[["CNTRY", "feature", "cb_var_imp"]].rename(columns={"cb_var_imp": "cb_var"}).assign(stat="IMP"),
        cb[["CNTRY", "feature", "cb_var_slope"]].rename(columns={"cb_var_slope": "cb_var"}).assign(stat="SLOPE"),
        cbb.rename(columns={"cb_var_block": "cb_var"}).assign(stat="BLOCK")])
    d = d.merge(cbl, on=["CNTRY", "feature", "stat"], how="left", validate="one_to_one")
    d["var_boot"] = d.cb_var + 1.2 * d.imp_var
    d["var_jrr"] = d.samp_var + 1.2 * d.imp_var
    d["var_main"] = d.var_jrr                      # primary: conservative JRR (see module docstring)
    d["se_main"] = np.sqrt(d.var_main)
    d["se_boot"] = np.sqrt(d.var_boot)
    return d


def main():
    d = load_stats()
    for stat in ["SLOPE", "IMP", "BLOCK"]:
        rows = []
        for excl, tag, vcol in [(EXCL, "main", "var_main"), (EXCL + ["ROU"], "excl_ROU", "var_main"),
                                (EXCL, "jrr", "var_jrr"), (EXCL, "boot", "var_boot")]:
            x = d[(d.stat == stat) & ~d.CNTRY.isin(excl)].dropna(subset=["est", vcol])
            x = x[x[vcol] > 0]
            for f, g in x.groupby("feature"):
                r = reml(g.est, g[vcol])
                z = g.est / np.sqrt(g[vcol])
                r.update(feature=f, analysis=tag, n_pos_sig=int((z > 1.96).sum()),
                         n_neg_sig=int((z < -1.96).sum()))
                if f in FEATURES:
                    r.update(block=FEATURES[f][0], label=FEATURES[f][1])
                rows.append(r)
        out = pd.DataFrame(rows)
        out.to_csv(RES / f"meta_{stat}.csv", index=False)
        m = out[out.analysis == "main"].sort_values("est", key=np.abs, ascending=False)
        print(stat)
        print(m[["feature", "k", "est", "ci_lo", "ci_hi", "tau", "I2", "Q_df", "n_pos_sig", "n_neg_sig"]]
              .head(15).round(2).to_string(index=False))
    # country-level slope tests (main variance) with Benjamini-Hochberg across all tests
    s = d[(d.stat == "SLOPE")].dropna(subset=["est", "se_main"]).copy()
    s = s[s.se_main > 0]
    s["z"] = s.est / s.se_main
    s["p"] = 2 * st.norm.sf(np.abs(s.z))
    s["q_bh"] = bh(s.p.values)
    sb = s.se_boot > 0
    s["p_boot"] = np.where(sb, 2 * st.norm.sf(np.abs(s.est / s.se_boot.where(sb))), np.nan)
    s.loc[sb, "q_bh_boot"] = bh(s.loc[sb, "p_boot"].values)
    s.to_csv(RES / "country_slope_tests.csv", index=False)


if __name__ == "__main__":
    main()
