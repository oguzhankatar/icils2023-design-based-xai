"""Build the analysis file: students + school-aggregated teacher data + school data.

Output: data/processed/analysis.parquet, results/prep_summary.csv
"""
import numpy as np
import pandas as pd
import pyreadstat

from config import RAW, PROC, RES, FEAT, PVS, REPW

CNTRYS = sorted(p.name[3:6] for p in RAW.glob("BSG*I3.sav"))


def read(prefix, c, cols=None):
    df, _ = pyreadstat.read_sav(str(RAW / f"{prefix}{c}I3.sav"), user_missing=False,
                                usecols=cols)
    return df


def rev(x, k):
    """Reverse an ordinal item coded 1..k (1 = most) into 0..k-1 (higher = more)."""
    return k - x


def shift(x):
    """Shift an ordinal item coded 1..k (1 = least) to 0..k-1."""
    return x - 1


def rowmean(df, cols, min_valid):
    m = df[cols].mean(axis=1)
    m[df[cols].notna().sum(axis=1) < min_valid] = np.nan
    return m


def wmean(x, w):
    ok = x.notna() & w.notna() & (w > 0)
    return np.nan if ok.sum() == 0 else np.average(x[ok], weights=w[ok])


def loo_wmean(d, var, w):
    """Weighted leave-one-out school mean of `var` (excludes the focal student)."""
    ok = d[var].notna() & (d[w] > 0)
    xw = (d[var] * d[w]).where(ok, 0.0)
    ww = d[w].where(ok, 0.0)
    sx = xw.groupby(d["IDSCHOOL"]).transform("sum")
    sw = ww.groupby(d["IDSCHOOL"]).transform("sum")
    return (sx - xw) / (sw - ww).replace(0, np.nan)


def student(c):
    s = read("BSG", c)
    o = pd.DataFrame({"CNTRY": c, "IDCNTRY": s.IDCNTRY, "IDSCHOOL": s.IDSCHOOL,
                      "IDCLASS": s.IDCLASS, "IDSTUD": s.IDSTUD})
    for v in ["TOTWGTS", "SENWGTS", "WGTFAC1", "WGTADJ1S", "WGTADJ3S", "JKZONES", "JKREPS"] + REPW + PVS:
        o[v] = s[v]
    for v in ["S_SEX", "S_AGE", "S_IMMBGR", "S_TLANG", "S_HISEI", "S_HISCED", "S_HOMLIT", "S_ISCED",
              "S_INTNET", "S_EXCOMP", "S_LRNSAFE", "S_CTCLS", "S_GENCLASS", "S_SPECLASS", "S_ICTSTUD",
              "S_GENEFF", "S_SPECEFF", "S_ACMULT", "S_ICTPOSG", "S_ICTNEG", "S_ICTFUT", "S_ICTPOSS"]:
        o[v] = s[v] if v in s else np.nan
    conn = pd.DataFrame({k: s[k].where(s[k] >= 2) - 2 for k in ["IS3G15A", "IS3G15B"]})
    o["H_CONNPROB"] = rowmean(conn, list(conn), 1)
    o["H_NCOMP"] = shift(s.IS3G16AA)
    o["H_NTAB"] = shift(s.IS3G16AB)
    for k, n in zip("ABCDEF", ["U_SCH_WORK", "U_SCH_OTH", "U_OUT_WORK", "U_OUT_OTH", "U_WE_WORK", "U_WE_OTH"]):
        o[n] = s[f"IS3G18{k}"]
    o["U_LIMIT_SD"] = (s.IS3G19A < 3).astype(float).where(s.IS3G19A.notna())
    o["U_LIMIT_NSD"] = (s.IS3G19B < 3).astype(float).where(s.IS3G19B.notna())
    for k, n in zip("ABCDEF", ["A_CHAT", "A_SMPOST", "A_SMCHECK", "A_INFO", "A_VIDEO", "A_MUSIC"]):
        o[n] = shift(s[f"IS3G21{k}"])
    for k, n in zip("ABCDEF", ["FIND", "REFINE", "EVAL", "REF", "SCAM", "PRIV"]):
        o[f"OTL_S_{n}"] = rev(s[f"IS3G20A{k}1"], 4)
        o[f"OTL_O_{n}"] = rev(s[f"IS3G20A{k}2"], 4)
    o["SCH_HISEI"] = loo_wmean(o.assign(WGTADJ3S=s.WGTADJ3S), "S_HISEI", "WGTADJ3S")
    o["SCH_HOMLIT"] = loo_wmean(o.assign(WGTADJ3S=s.WGTADJ3S), "S_HOMLIT", "WGTADJ3S")
    return o


def teacher(c):
    t = read("BTG", c)
    e = pd.DataFrame({k: rev(t[f"IT3G19{k}"], 4) for k in "ABCDEFGHIJKLMN"})
    d = pd.DataFrame({"IDSCHOOL": t.IDSCHOOL, "w": t.WGTFAC3T})
    d["T_INFOEMPH"] = rowmean(e, ["C", "M", "N"], 2)
    d["T_OTHEREMPH"] = rowmean(e, list("ABDEFGHIJKL"), 6)
    d["T_ITE_INFO"] = (t.IT3G08F == 1).astype(float).where(t.IT3G08F.notna())
    pd_ = pd.DataFrame({k: shift(t[k]) for k in ["IT3G09AF", "IT3G09AG"]})
    d["T_PD_INFO"] = rowmean(pd_, list(pd_), 1)
    for v in ["T_ICTEFF", "T_VWPOS", "T_VWNEG", "T_COLICT", "T_RESRC", "T_ICTCLASB", "T_ICTCLASA",
              "T_USETOOL", "T_PROFLRN", "T_EMPITE", "T_TEAPIN", "T_CODEMP", "T_EXLES"]:
        d[v] = t[v] if v in t else np.nan
    d["w"] = d["w"].fillna(1.0)
    tv = [v for v in d if v.startswith("T_")]
    agg = d.groupby("IDSCHOOL").apply(lambda g: pd.Series({v: wmean(g[v], g["w"]) for v in tv}),
                                      include_groups=False)
    agg["T_N"] = d.groupby("IDSCHOOL").size()
    return agg.reset_index()


def school(c):
    s = read("BCG", c)
    o = pd.DataFrame({"IDSCHOOL": s.IDSCHOOL})
    o["P_DECEPT"] = rev(s.IP3G11I, 4)
    for v in ["P_VWICT", "P_EXPLRN", "P_PRIORS", "C_HINRES", "C_HINPED", "C_ICTRES", "C_RATSTD",
              "P_NUMSTD_CAT"]:
        o[v] = s[v] if v in s else np.nan
    # students per student-available ICT device is extremely right-skewed (max > 400):
    # log scale (monotone, so tree models are unaffected)
    o["C_RATSTD"] = np.log(o["C_RATSTD"].where(o["C_RATSTD"] > 0))
    return o


def main():
    parts, summ = [], []
    for c in CNTRYS:
        st = student(c)
        n0 = len(st)
        df = st.merge(teacher(c), on="IDSCHOOL", how="left").merge(school(c), on="IDSCHOOL", how="left")
        assert len(df) == n0, c
        parts.append(df)
        summ.append({"CNTRY": c, "n_students": n0, "n_schools": df.IDSCHOOL.nunique(),
                     "pct_students_with_teacher_data": 100 * df.T_N.notna().mean(),
                     "pct_students_with_school_data": 100 * df.P_VWICT.notna().mean(),
                     "mean_feature_missing_pct": 100 * df[FEAT].isna().mean().mean()})
        print(c, n0, flush=True)
    all_ = pd.concat(parts, ignore_index=True)
    missing = [f for f in FEAT if f not in all_]
    assert not missing, missing
    all_.to_parquet(PROC / "analysis.parquet", index=False)
    pd.DataFrame(summ).to_csv(RES / "prep_summary.csv", index=False)
    fm = all_.groupby("CNTRY")[FEAT].apply(lambda g: g.isna().mean() * 100).T.round(1)
    fm.to_csv(RES / "feature_missing_by_country.csv")
    print(all_.shape)


if __name__ == "__main__":
    main()
