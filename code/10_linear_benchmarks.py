"""Design-based linear benchmarks, equivalence tests, school-mean
split, per-unit slopes, teacher-coverage sensitivity, sequential adjustment, OTL composites,
teacher-item split, descriptives and missingness. Does not modify any other script or result file.

Common estimation (as in 09_additional_analyses.py): per system, plausible value (5) and weight set
(TOTWGTS + 75 JRR replicate weights SRWGT01-75), weighted least squares; JRR sampling variance
SV_p = sum_r (theta_pr - theta_p)^2 (no 1/2 factor); Rubin: Var = mean_p SV_p + (1 + 1/5) B
(survey.combine_pv). Covariates with missing values: TOTWGTS-weighted system mean imputation
(computed once on all students of the system, fixed across replicate weights) plus a 0/1 missing
indicator per covariate with any missing value in the system; columns constant in the analytic sample
and identical (duplicate) columns are dropped (e.g. the shared missing indicator of all teacher
variables), as are exactly collinear columns (greedy, in column order: predictors before indicators). Predictors entirely missing in a system are left out of that system's model.
Coefficients are in CIL points per weighted SD of the predictor (SD of the observed values computed
with the same weight vector as the fit), unless stated otherwise (per-unit slopes). Pooling across the
34 systems (DNW excluded) with 05_meta.reml (REML tau^2, truncated HKSJ CI on t_{k-1}, PI on
t_{k-2}, I^2) and counts of systems with z > 1.96 / z < -1.96.

(1) Fully adjusted OLS: CIL on all 79 predictors (all students; the predictors themselves mean-imputed
    with missing indicators). -> results/ols_full_by_system.csv, results/meta_ols_full.csv
(2) Equivalence (TOST, alpha = .05 per side = 90% HKSJ CI on t_{k-1} inside +-Delta, Delta = 3 and 5)
    of the pooled focal estimates: bivariate, background-adjusted (results/meta_focal_unadjusted.csv),
    fully adjusted OLS (1), SHAP SLOPE (results/meta_SLOPE.csv, main). -> results/equivalence_focal.csv
(3) School-mean split of the 4 student focal items: leave-one-out school mean (WGTADJ3S-weighted, as
    for SCH_HISEI/SCH_HOMLIT in 01_prepare.loo_wmean) and within-school deviation (item - LOO mean),
    entered together with the step-(b) adjustment set of (7) (background except expected education
    plus peers' LOO HISEI and books, i.e. step (b) of the sequential adjustment (7)); both coefficients times the
    weighted SD of the item (points per student-level SD of the item); contextual = between - within;
    'between_sdmean' = between coefficient per weighted SD of the LOO school mean itself. Weighted ICC
    of each item (TOTWGTS, moment estimator, see icc()). -> results/meta_focal_schoolsplit.csv
(4) Per-unit slopes (points per scale category, 0-3) of the 4 student items, step-(b) adjustment
    set (as (3); 'bg_perSD' then equals step b of (7), checked). -> results/meta_focal_perunit.csv
(5) Teacher-coverage sensitivity for the 4 school-level focal predictors: SHAP SLOPE (system estimates
    and main variances from 05_meta.load_stats), fully adjusted OLS (1) and background-adjusted (09),
    pooled in systems with >= 85% of students with teacher data (results/prep_summary.csv) and a
    mixed-effects meta-regression on coverage (REML, Knapp-Hartung, truncated at q >= 1, t_{k-2}).
    -> results/teacher_coverage_sensitivity.csv
(6) Descriptives / missingness: focal by system (TOTWGTS-weighted mean, SD, % top two categories,
    % missing); all 79 predictors (weighted % missing: median, max across the 34 systems, number of
    systems with the predictor entirely missing); weighted mean CIL of students with missing vs
    observed answers to the 4 student OTL items (design-based difference missing - observed, systems
    with >= 30 missing students, pooled). -> results/focal_descriptives.csv,
    results/missingness_all.csv, results/otl_missing_cil.csv
(7) Sequential adjustment of the 8 focal predictors (students with the focal predictor observed):
    a = bivariate; b = background except expected education + peers' LOO HISEI and books
    (SCH_HISEI, SCH_HOMLIT); c = b + ICT-use block; d = c + dispositions block + expected education;
    e = fully adjusted model (1). -> results/meta_focal_sequential.csv
(8) OTL composites: mean of the six at-school items (OTLS6) and of the six outside-school items
    (OTLO6), at least 4 of 6 valid, per weighted SD of the composite, under steps a, b, c, d and e (in
    e the 12 item columns are replaced by the 2 composites). Also the six at-school items under (1).
    -> results/meta_otl_composites.csv
(9) Teacher emphasis items: T_INFOEMPH = school mean (WGTFAC3T) of the teacher mean of IT3G19C/M/N
    (reversed to 0-3, >= 2 valid). Item-level school means rebuilt from the raw teacher files with the
    same weighting (checked against T_INFOEMPH); each item under steps a, b and e (in e the item
    replaces T_INFOEMPH). Cronbach's alpha of the three items at teacher level (unweighted and
    TOTWGTT-weighted, complete cases) by system. -> results/teacher_items_split.csv,
    results/teacher_items_alpha.csv
(10) Wording of the focal items from the IDB SPSS metadata. -> results/focal_item_wording.csv
(11) Participation sensitivity: re-pooling without systems that missed the student (or, for the 4
    school-level predictors, the teacher) participation requirements / were not reported, flags of
    results/published_cil_means.csv (ICILS 2023 technical report). -> results/participation_sensitivity.csv
(12) Response-style robustness of the 4 student items (background model = step-(b) adjustment set of
    (7)): (a) background model plus the student's mean of
    all 12 OTL items (>= 8 valid; in the fully adjusted model the 12 items are already controlled, so
    adding their mean would be (near-)collinear and is not estimated); (b) within-student contrast
    outside minus at school (EVAL, SCAM) as predictor per SD of the contrast: background model, background
    model + pair mean (level), fully adjusted model with the two items replaced by contrast and pair mean.
    -> results/otl_response_style.csv
(13) Reporting extras for the 8 focal predictors: exact HKSJ p (t_{k-1}) of the pooled SHAP SLOPE and
    fully adjusted OLS; minimum detectable effect of a system-level test at 80% power, alpha = .05
    (2.8 x median system SE); Shapiro-Wilk test and skewness of the standardised system deviations
    (est - mu) / sqrt(v + tau^2). -> results/reporting_extras.csv
(14) Subject-restricted teacher emphasis: T_INFOEMPH rebuilt from teachers whose main subjects
    (IT3G03A-I, 'Checked') include language arts in the test language (A), human sciences/humanities (E)
    or information technology (G); same item coding and WGTFAC3T school weighting; steps a, b, e.
    Share of such teachers among teachers answering the subject question, by system.
    -> results/teacher_subject_emphasis.csv, results/teacher_subject_share.csv
Per-system estimates of all linear models: results/linear_by_system.csv

Run: python 10_linear_benchmarks.py   (pure numpy/scipy, CPU; 4 worker processes, 1 BLAS thread each)
"""
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
import scipy.linalg as sla
from scipy import stats as st

from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from config import PROC, RES, RAW, FEAT, FEATURES, FOCAL, PVS, REPW, WGT
from survey import combine_pv

spec = importlib.util.spec_from_file_location("meta05", Path(__file__).parent / "05_meta.py")
meta05 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(meta05)
reml, EXCL, load_stats = meta05.reml, meta05.EXCL, meta05.load_stats

BG = [f for f in FEAT if FEATURES[f][0] == "Background" and f != "S_ISCED"]
assert BG == ["S_SEX", "S_AGE", "S_IMMBGR", "S_TLANG", "S_HISEI", "S_HISCED", "S_HOMLIT"], BG
COMP = ["SCH_HISEI", "SCH_HOMLIT"]
ICTUSE = [f for f in FEAT if FEATURES[f][0] == "ICT use"]
DISP = [f for f in FEAT if FEATURES[f][0] == "Dispositions"] + ["S_ISCED"]
STEPS = {"b": BG + COMP, "c": BG + COMP + ICTUSE, "d": BG + COMP + ICTUSE + DISP}
BGB = BG + COMP   # step-(b) adjustment set, the 'background model' of (3), (4) and (12)
OTL_S6 = [f for f in FEAT if FEATURES[f][0] == "OTL info (school)"]
OTL_O6 = [f for f in FEAT if FEATURES[f][0] == "OTL info (outside)"]
ST4 = ["OTL_S_EVAL", "OTL_S_SCAM", "OTL_O_EVAL", "OTL_O_SCAM"]
SCH4 = ["T_INFOEMPH", "T_ITE_INFO", "T_PD_INFO", "P_DECEPT"]
assert ST4 + SCH4 == FOCAL
TITEMS = {"T_EMPH_C": "IT3G19C", "T_EMPH_M": "IT3G19M", "T_EMPH_N": "IT3G19N"}
SUBJ_ALL = [f"IT3G03{k}" for k in "ABCDEFGHI"]
SUBJ_KEEP = ["IT3G03A", "IT3G03E", "IT3G03G"]   # test language arts, human sciences/humanities, IT
MIN_N = 30
N_JOBS = 4
COV_CUT = 85.0
DELTAS = [3.0, 5.0]


# ------------------------------------------------------------------ helpers
def wsd(x, w):
    sw = w.sum()
    m = (w * x).sum() / sw
    return np.sqrt((w * (x - m) ** 2).sum() / sw)


def wslope(x, Y, w):
    """Weighted Cov(x, Y) / SD(x) for each column of Y (as 09_additional_analyses.wslope)."""
    sw = w.sum()
    mx = (w * x).sum() / sw
    dx = x - mx
    vx = (w * dx ** 2).sum() / sw
    if not vx > 0:
        return np.full(Y.shape[1], np.nan)
    my = (w[:, None] * Y).sum(0) / sw
    return (w[:, None] * dx[:, None] * (Y - my)).sum(0) / sw / np.sqrt(vx)


def summarize(A):
    """A: (76, P) estimates per weight set (row 0 = TOTWGTS) and PV -> est, se, U, B."""
    full = A[0]
    sv = ((A[1:] - full) ** 2).sum(0)
    est, se, u, b = combine_pv(full, sv)
    return dict(est=float(est), se=float(se), samp_var=float(u), imp_var=float(b))


def loo_school_mean(x, w, school):
    """Weighted leave-one-out school mean of x (01_prepare.loo_wmean, weight WGTADJ3S)."""
    ok = ~np.isnan(x) & (w > 0)
    xw = np.where(ok, x * w, 0.0)
    ww = np.where(ok, w, 0.0)
    sx = pd.Series(xw).groupby(school).transform("sum").to_numpy()
    sw = pd.Series(ww).groupby(school).transform("sum").to_numpy()
    den = sw - ww
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (sx - xw) / np.where(den > 0, den, 1.0), np.nan)


def icc(x, w, school):
    """Weighted one-way moment ICC: schools j with total weight W_j, weighted mean m_j, weighted
    within variance s2_j and effective size n_j = W_j^2 / sum w^2 (n_j > 1).
    sigma_w^2 = sum_j W_j s2_j n_j/(n_j-1) / sum W_j;
    sigma_b^2 = max(0, sum_j W_j (m_j - m)^2 / sum W_j - sum_j W_j [s2_j/(n_j-1)] / sum W_j)."""
    ok = ~np.isnan(x) & (w > 0)
    d = pd.DataFrame({"s": school[ok], "w": w[ok], "x": x[ok]})
    d["wx"] = d.w * d.x
    d["wx2"] = d.w * d.x ** 2
    d["w2"] = d.w ** 2
    a = d.groupby("s")[["w", "wx", "wx2", "w2"]].sum()
    a["m"] = a.wx / a.w
    a["s2"] = a.wx2 / a.w - a.m ** 2
    a["ne"] = a.w ** 2 / a.w2
    a = a[a["ne"] > 1.0001]
    if len(a) < 2:
        return np.nan
    W = a.w.sum()
    m = (a.w * a.m).sum() / W
    s2u = a.s2 * a["ne"] / (a["ne"] - 1)
    sw2 = (a.w * s2u).sum() / W
    sb2 = max(0.0, (a.w * (a.m - m) ** 2).sum() / W - (a.w * s2u / a["ne"]).sum() / W)
    return sb2 / (sb2 + sw2) if sb2 + sw2 > 0 else np.nan


def cronbach(X, w=None):
    X = X[~np.isnan(X).any(1)]
    if w is not None:
        w = w[: len(w)]
    k = X.shape[1]
    if len(X) < 10:
        return np.nan, len(X)
    if w is None:
        w = np.ones(len(X))
    sw = w.sum()
    m = (w[:, None] * X).sum(0) / sw
    iv = (w[:, None] * (X - m) ** 2).sum(0) / sw
    t = X.sum(1)
    tv = (w * (t - (w * t).sum() / sw) ** 2).sum() / sw
    return k / (k - 1) * (1 - iv.sum() / tv), len(X)


class System:
    def __init__(self, g):
        self.g = g
        self.Y = g[PVS].to_numpy(float)
        self.wt = g[WGT].to_numpy(float)
        self.W = np.column_stack([self.wt] + [g[r].to_numpy(float) for r in REPW])
        self.school = g.IDSCHOOL.to_numpy()
        self._c = {}

    def raw(self, f):
        return self.g[f].to_numpy(float)

    def filled(self, f):
        if f not in self._c:
            v = self.raw(f)
            miss = np.isnan(v)
            if miss.all():
                self._c[f] = (None, None)
            else:
                m = np.average(v[~miss], weights=self.wt[~miss])
                self._c[f] = (np.where(miss, m, v), miss if miss.any() else None)
        return self._c[f]


def design(S, names, rows, lead=()):
    """Intercept, lead columns (no missing), filled predictors, then missing indicators; drops
    columns constant in `rows` and duplicate columns (first occurrence kept)."""
    n = int(rows.sum())
    cols, labs, inds = [np.ones(n)], ["_const"], []
    for lab, a in lead:
        cols.append(np.asarray(a, float))
        labs.append(lab)
    for f in names:
        fv, miss = S.filled(f)
        if fv is None:
            continue
        cols.append(fv[rows])
        labs.append(f)
        if miss is not None:
            inds.append((f + "_miss", miss[rows].astype(float)))
    for lab, a in inds:
        cols.append(a)
        labs.append(lab)
    keep, seen = [0], set()
    for j in range(1, len(cols)):
        a = cols[j]
        if np.ptp(a) == 0:
            continue
        key = np.ascontiguousarray(a).tobytes()
        if key in seen:
            continue
        seen.add(key)
        keep.append(j)
    X = np.column_stack([cols[j] for j in keep])
    labs = [labs[j] for j in keep]
    # drop exactly collinear columns (greedy in column order, so later missing indicators are dropped
    # before predictors; e.g. 'no internet at home' implies the connection-problem item is missing)
    Xs = X / np.linalg.norm(X, axis=0)
    G = Xs.T @ Xs
    ind = []
    for j in range(X.shape[1]):
        if ind:
            Gk = G[np.ix_(ind, ind)]
            g = G[ind, j]
            res = G[j, j] - g @ np.linalg.lstsq(Gk, g, rcond=None)[0]
        else:
            res = G[j, j]
        if res > 1e-9:
            ind.append(j)
    return X[:, ind], [labs[j] for j in ind]


def wls(D, Y, W, report, sdsrc):
    """report: column indices of D (None = not estimable); sdsrc: (x, mask) per reported column
    for the SD scaling (None = per unit). Returns (n_w, P, n_report)."""
    out = np.full((W.shape[1], Y.shape[1], len(report)), np.nan)
    for r in range(W.shape[1]):
        w = W[:, r]
        sw = np.sqrt(w)
        coef = sla.lstsq(D * sw[:, None], Y * sw[:, None], lapack_driver="gelsy",
                         check_finite=False)[0]
        for j, (ci, sd) in enumerate(zip(report, sdsrc)):
            if ci is None:
                continue
            s = 1.0 if sd is None else wsd(sd[0][sd[1]], w[sd[1]])
            out[r, :, j] = coef[ci] * s
    return out


# ------------------------------------------------------------------ per-system models
def run_system(c, g, rows_out, extra_out):
    S = System(g.reset_index(drop=True))
    n = len(S.Y)
    allrows = np.ones(n, bool)

    def add(analysis, feature, A, n_obs, component="", **kw):
        r = dict(CNTRY=c, analysis=analysis, feature=feature, component=component, n_obs=int(n_obs))
        if A is not None and np.isfinite(A[0]).all():
            r.update(summarize(A))
        r.update(kw)
        rows_out.append(r)

    # ---- full models (1), (8e), (9e)
    def full_model(names, tag, report_names):
        D, labs = design(S, names, allrows)
        rank = np.linalg.matrix_rank(D * np.sqrt(S.wt)[:, None])
        rep, sds = [], []
        for f in report_names:
            ci = labs.index(f) if f in labs else None
            rep.append(ci)
            x = S.raw(f)
            sds.append((x, ~np.isnan(x)))
        A = wls(D, S.Y, S.W, rep, sds)
        for j, f in enumerate(report_names):
            add(tag, f, A[:, :, j] if rep[j] is not None else None, (~np.isnan(S.raw(f))).sum(),
                n_cols=D.shape[1], rank=int(rank))
        return D.shape[1], rank

    p, rk = full_model(FEAT, "full", FEAT)
    print(f"  {c}: full model n={n} cols={p} rank={rk}", flush=True)
    comp_names = [f for f in FEAT if f not in OTL_S6 + OTL_O6] + ["OTLS6", "OTLO6"]
    full_model(comp_names, "comp_e", ["OTLS6", "OTLO6"])
    for ti in TITEMS:
        full_model([ti if f == "T_INFOEMPH" else f for f in FEAT], "titem_e", [ti])

    # ---- bivariate and stepwise models for a predictor x (students with x observed)
    def steps(f, tagpre, which):
        x = S.raw(f)
        ok = ~np.isnan(x)
        if ok.sum() < MIN_N or np.nanstd(x) == 0:
            for s_ in which:
                add(f"{tagpre}_{s_}", f, None, ok.sum())
            return
        if "a" in which:
            A = np.array([wslope(x[ok], S.Y[ok], S.W[ok, r]) for r in range(S.W.shape[1])])
            add(f"{tagpre}_a", f, A, ok.sum())
        for s_ in [s for s in which if s in STEPS]:
            D, labs = design(S, [v for v in STEPS[s_] if v != f], ok, lead=[(f, x[ok])])
            A = wls(D, S.Y[ok], S.W[ok], [1], [(x[ok], np.ones(ok.sum(), bool))])
            add(f"{tagpre}_{s_}", f, A[:, :, 0], ok.sum(), n_cols=D.shape[1])

    for f in FOCAL:
        steps(f, "seq", ["a", "b", "c", "d"])
    for f in ["OTLS6", "OTLO6"]:
        steps(f, "comp", ["a", "b", "c", "d"])
    for f in TITEMS:
        steps(f, "titem", ["a", "b"])
    # (14) subject-restricted teacher emphasis
    steps("T_INFOEMPH_SUBJ", "tsubj", ["a", "b"])
    full_model(["T_INFOEMPH_SUBJ" if f == "T_INFOEMPH" else f for f in FEAT], "tsubj_e", ["T_INFOEMPH_SUBJ"])
    # (12) response style
    for f in ST4:
        x = S.raw(f)
        ok = ~np.isnan(x)
        D, labs = design(S, BGB + ["OTL12"], ok, lead=[(f, x[ok])])
        A = wls(D, S.Y[ok], S.W[ok], [1], [(x[ok], np.ones(ok.sum(), bool))])
        add("rs_bg_ctrl12", f, A[:, :, 0], ok.sum())
    for k in ["EVAL", "SCAM"]:
        dv = S.raw(f"DIFF_{k}")
        pv = S.raw(f"PAIR_{k}")
        ok = ~np.isnan(dv)
        sd = (dv[ok], np.ones(ok.sum(), bool))
        D, labs = design(S, BGB, ok, lead=[(f"DIFF_{k}", dv[ok])])
        add("rs_bg_contrast", f"DIFF_{k}", wls(D, S.Y[ok], S.W[ok], [1], [sd])[:, :, 0], ok.sum())
        D, labs = design(S, BGB, ok, lead=[(f"DIFF_{k}", dv[ok]), (f"PAIR_{k}", pv[ok])])
        add("rs_bg_contrast_level", f"DIFF_{k}", wls(D, S.Y[ok], S.W[ok], [1], [sd])[:, :, 0], ok.sum())
        full_model([f for f in FEAT if f not in (f"OTL_S_{k}", f"OTL_O_{k}")] + [f"DIFF_{k}", f"PAIR_{k}"],
                   "rs_full_contrast", [f"DIFF_{k}"])

    # ---- background model = step-(b) set (BG + peers): per SD (= step b), per unit (4), school-mean
    #      split (3); the 09 background block without peers ('bg09_perSD') is kept only as a check
    for f in ST4:
        x = S.raw(f)
        ok = ~np.isnan(x)
        D, labs = design(S, BG, ok, lead=[(f, x[ok])])
        A = wls(D, S.Y[ok], S.W[ok], [1], [(x[ok], np.ones(ok.sum(), bool))])
        add("bg09_perSD", f, A[:, :, 0], ok.sum())
        D, labs = design(S, BGB, ok, lead=[(f, x[ok])])
        A = wls(D, S.Y[ok], S.W[ok], [1, 1], [(x[ok], np.ones(ok.sum(), bool)), None])
        add("bg_perSD", f, A[:, :, 0], ok.sum())
        add("bg_perunit", f, A[:, :, 1], ok.sum())
        m = loo_school_mean(x, S.g.WGTADJ3S.to_numpy(float), S.school)
        ok2 = ok & ~np.isnan(m)
        dev = x - m
        D, labs = design(S, BGB, ok2, lead=[("dev", dev[ok2]), ("mean", m[ok2])])
        assert labs[1:3] == ["dev", "mean"], labs[:3]
        sd = (x[ok2], np.ones(ok2.sum(), bool))
        sdm = (m[ok2], np.ones(ok2.sum(), bool))
        A = wls(D, S.Y[ok2], S.W[ok2], [1, 2, 2], [sd, sd, sdm])
        add("split", f, A[:, :, 0], ok2.sum(), component="within")
        add("split", f, A[:, :, 1], ok2.sum(), component="between")
        add("split", f, A[:, :, 1] - A[:, :, 0], ok2.sum(), component="contextual")
        add("split", f, A[:, :, 2], ok2.sum(), component="between_sdmean")
        w0 = S.wt[ok2]
        extra_out["sdratio"].append(dict(CNTRY=c, feature=f, sd_item=wsd(x[ok2], w0),
                                         sd_schoolmean=wsd(m[ok2], w0)))
        extra_out["icc"].append(dict(CNTRY=c, feature=f, icc=icc(x, S.wt, S.school)))

    # ---- (6c) CIL of students with missing vs observed answers (4 OTL items; also the 4 school-level)
    for f in FOCAL:
        miss = np.isnan(S.raw(f))
        r = dict(CNTRY=c, feature=f, n_missing=int(miss.sum()), n_observed=int((~miss).sum()),
                 pct_missing_w=100 * S.wt[miss].sum() / S.wt.sum())
        if miss.sum() >= MIN_N and (~miss).sum() >= MIN_N:
            def mdiff(w):
                return ((w[miss, None] * S.Y[miss]).sum(0) / w[miss].sum()
                        - (w[~miss, None] * S.Y[~miss]).sum(0) / w[~miss].sum())
            A = np.array([mdiff(S.W[:, k]) for k in range(S.W.shape[1])])
            r.update(summarize(A))
            r["cil_missing"] = float(np.mean((S.wt[miss, None] * S.Y[miss]).sum(0) / S.wt[miss].sum()))
            r["cil_observed"] = float(np.mean((S.wt[~miss, None] * S.Y[~miss]).sum(0) / S.wt[~miss].sum()))
        extra_out["miss"].append(r)


def _one(c, g):
    """Worker: one system (single-threaded BLAS)."""
    with threadpool_limits(1):
        rows, extra = [], {"icc": [], "miss": [], "sdratio": []}
        run_system(c, g, rows, extra)
    return rows, extra


# ------------------------------------------------------------------ meta helpers
def pool(d, keys):
    x = d[~d.CNTRY.isin(EXCL)].dropna(subset=["est", "se"])
    x = x[x.se > 0]
    out = []
    for k, gg in x.groupby(keys, sort=False):
        r = reml(gg.est, gg.se ** 2)
        z = gg.est / gg.se
        row = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        row.update(est=r["est"], se=r["se"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], pi_lo=r["pi_lo"],
                   pi_hi=r["pi_hi"], tau=r["tau"], I2=r["I2"], n_pos_sig=int((z > 1.96).sum()),
                   n_neg_sig=int((z < -1.96).sum()), k=r["k"], p=r["p"])
        out.append(row)
    return pd.DataFrame(out)


def metareg(y, v, x):
    """Mixed-effects meta-regression y ~ 1 + x, REML tau^2 (Fisher scoring), Knapp-Hartung SE with
    q truncated at 1, t_{k-2}."""
    y, v, x = (np.asarray(a, float) for a in (y, v, x))
    X = np.column_stack([np.ones(len(y)), x])
    k, p = X.shape
    tau2 = 0.0
    for _ in range(500):
        w = 1 / (v + tau2)
        XtW = X.T * w
        Pm = np.diag(w) - XtW.T @ np.linalg.solve(XtW @ X, XtW)
        Py = Pm @ y
        new = max(0.0, tau2 + (Py @ Py - np.trace(Pm)) / np.trace(Pm @ Pm))
        if abs(new - tau2) < 1e-10:
            tau2 = new
            break
        tau2 = new
    w = 1 / (v + tau2)
    XtW = X.T * w
    V = np.linalg.inv(XtW @ X)
    b = V @ XtW @ y
    q = np.sum(w * (y - X @ b) ** 2) / (k - p)
    se = np.sqrt(np.diag(V) * max(q, 1.0))
    t = b[1] / se[1]
    return dict(mr_slope=b[1], mr_se=se[1], mr_p=2 * st.t.sf(abs(t), k - p), mr_tau=np.sqrt(tau2), mr_k=k)


# ------------------------------------------------------------------ teacher items (9)
def teacher_items(cntrys):
    sch, alph, shares = [], [], []
    for c in cntrys:
        t, _ = pyreadstat.read_sav(str(RAW / f"BTG{c}I3.sav"), user_missing=False,
                                   usecols=["IDSCHOOL", "IT3G19C", "IT3G19M", "IT3G19N", "WGTFAC3T", "TOTWGTT"]
                                   + SUBJ_ALL)
        e = pd.DataFrame({k: 4 - t[v] for k, v in TITEMS.items()})
        w = t.WGTFAC3T.fillna(1.0)
        teach_mean = e.mean(axis=1).where(e.notna().sum(axis=1) >= 2)
        answered = t[SUBJ_ALL].notna().any(axis=1)
        subj = (t[SUBJ_KEEP] == 1).any(axis=1) & answered
        d = e.assign(T_INFOEMPH_chk=teach_mean, T_INFOEMPH_SUBJ=teach_mean.where(subj),
                     IDSCHOOL=t.IDSCHOOL, w=w)
        tw = t.TOTWGTT.fillna(0).to_numpy(float)
        ans = answered.to_numpy()
        shares.append(dict(CNTRY=c, n_teachers=len(t), n_answered_subject=int(ans.sum()),
                           n_subject_teachers=int(subj.sum()),
                           share_subject=float(subj[ans].mean()) if ans.any() else np.nan,
                           share_subject_TOTWGTT=float(tw[subj.to_numpy()].sum() / tw[ans].sum()) if tw[ans].sum() > 0 else np.nan,
                           n_schools=int(t.IDSCHOOL.nunique()),
                           n_schools_with_subject_emphasis=int(d.dropna(subset=["T_INFOEMPH_SUBJ"]).IDSCHOOL.nunique())))
        agg = {}
        for v in list(TITEMS) + ["T_INFOEMPH_chk", "T_INFOEMPH_SUBJ"]:
            ok = d[v].notna()
            agg[v] = (d[v].where(ok, 0) * d.w.where(ok, 0)).groupby(d.IDSCHOOL).sum() / \
                d.w.where(ok, 0).groupby(d.IDSCHOOL).sum().replace(0, np.nan)
        a = pd.DataFrame(agg).reset_index().assign(CNTRY=c)
        sch.append(a)
        X = e.to_numpy(float)
        au, nu = cronbach(X)
        ok = ~np.isnan(X).any(1) & t.TOTWGTT.notna().to_numpy()
        aw, _ = cronbach(X[ok], t.TOTWGTT.to_numpy(float)[ok])
        r = e.corr().to_numpy()
        alph.append(dict(CNTRY=c, alpha_unweighted=au, alpha_TOTWGTT=aw, n_teachers_complete=nu,
                         r_CM=r[0, 1], r_CN=r[0, 2], r_MN=r[1, 2]))
    return pd.concat(sch, ignore_index=True), pd.DataFrame(alph), pd.DataFrame(shares)


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    df = pd.read_parquet(PROC / "analysis.parquet")
    cntrys = sorted(df.CNTRY.unique())
    # composites (8)
    for nm, items in [("OTLS6", OTL_S6), ("OTLO6", OTL_O6)]:
        df[nm] = df[items].mean(axis=1).where(df[items].notna().sum(axis=1) >= 4)
    # response-style variables (12)
    o12 = OTL_S6 + OTL_O6
    df["OTL12"] = df[o12].mean(axis=1).where(df[o12].notna().sum(axis=1) >= 8)
    for k in ["EVAL", "SCAM"]:
        df[f"DIFF_{k}"] = df[f"OTL_O_{k}"] - df[f"OTL_S_{k}"]
        df[f"PAIR_{k}"] = (df[f"OTL_O_{k}"] + df[f"OTL_S_{k}"]) / 2
    # teacher item school means (9)
    tsch, alpha, tshare = teacher_items(cntrys)
    n0 = len(df)
    df = df.merge(tsch, on=["CNTRY", "IDSCHOOL"], how="left", validate="many_to_one")
    assert len(df) == n0
    dchk = (df.T_INFOEMPH_chk - df.T_INFOEMPH).abs()
    print("T_INFOEMPH rebuild: max abs diff", np.nanmax(dchk), "| NaN pattern equal:",
          bool((df.T_INFOEMPH_chk.isna() == df.T_INFOEMPH.isna()).all()), flush=True)
    assert np.nanmax(dchk) < 1e-9
    print(f"teacher items done {time.time() - t0:.0f}s", flush=True)

    rows, extra = [], {"icc": [], "miss": [], "sdratio": []}
    res = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(_one)(c, df[df.CNTRY == c]) for c in cntrys)
    for r_, e_ in res:
        rows += r_
        extra["icc"] += e_["icc"]
        extra["miss"] += e_["miss"]
        extra["sdratio"] += e_["sdratio"]
    print(f"systems done {time.time() - t0:.0f}s", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(RES / "linear_by_system.csv", index=False)

    # ---- (1) fully adjusted OLS
    full = d[d.analysis == "full"].drop(columns=["analysis", "component"])
    full.to_csv(RES / "ols_full_by_system.csv", index=False)
    mfull = pool(full, ["feature"])
    mfull["block"] = mfull.feature.map(lambda f: FEATURES[f][0])
    mfull["label"] = mfull.feature.map(lambda f: FEATURES[f][1])
    mfull["focal"] = mfull.feature.isin(FOCAL)
    mfull.to_csv(RES / "meta_ols_full.csv", index=False)
    print(mfull[mfull.focal].round(3).to_string())
    cst = pd.read_csv(RES / "country_slope_tests.csv")
    cst = cst[(cst.stat == "SLOPE") & cst.feature.isin(FOCAL) & ~cst.CNTRY.isin(EXCL)]
    ser = full.merge(cst[["CNTRY", "feature", "se_main"]], on=["CNTRY", "feature"])
    ser["ratio"] = ser.se / ser.se_main
    print("SE ratio OLS/SHAP (focal): median %.3f" % ser.ratio.median())
    print(ser.groupby("feature").ratio.median().round(3).to_string())

    # ---- (7) sequential
    seq = d[d.analysis.str.startswith("seq_")].copy()
    seq["step"] = seq.analysis.str[-1]
    fe = full[full.feature.isin(FOCAL)].assign(step="e")
    seq = pd.concat([seq, fe], ignore_index=True)
    mseq = pool(seq, ["feature", "step"])
    mseq["feature"] = pd.Categorical(mseq.feature, FOCAL, ordered=True)
    mseq = mseq.sort_values(["feature", "step"]).reset_index(drop=True)
    mseq["step_def"] = mseq.step.map({
        "a": "bivariate", "b": "background (excl. expected education) + peers' LOO HISEI and books",
        "c": "b + ICT use block", "d": "c + dispositions block + expected education",
        "e": "fully adjusted (all 79 predictors)"})
    mseq.to_csv(RES / "meta_focal_sequential.csv", index=False)
    print(mseq.pivot(index="feature", columns="step", values="est").round(2).to_string())

    # check: step a reproduces 09 bivariate; bg_perSD reproduces 09 background
    old = pd.read_csv(RES / "focal_unadjusted.csv")
    for mod, tag in [("bivariate", "seq_a"), ("background", "bg09_perSD")]:
        chk = old[old.model == mod].merge(d[d.analysis == tag], on=["CNTRY", "feature"], suffixes=("_09", ""))
        print(f"check vs 09 {mod}: max |est diff| {np.nanmax(np.abs(chk.est - chk.est_09)):.2e}, "
              f"max |se diff| {np.nanmax(np.abs(chk.se - chk.se_09)):.2e}, n={len(chk)}")

    # check: the step-(b) background model of (3)/(4)/(12) equals step b of (7) for the 4 student items
    chk = d[d.analysis == "bg_perSD"].merge(seq[seq.step == "b"], on=["CNTRY", "feature"], suffixes=("", "_b"))
    print(f"check bg_perSD = seq_b: max |est diff| {np.nanmax(np.abs(chk.est - chk.est_b)):.2e}, n={len(chk)}")
    assert np.nanmax(np.abs(chk.est - chk.est_b)) < 1e-8 and len(chk) == 4 * 35

    # ---- (2) equivalence
    mf = pd.read_csv(RES / "meta_focal_unadjusted.csv")
    ms = pd.read_csv(RES / "meta_SLOPE.csv")
    ms = ms[(ms.analysis == "main") & ms.feature.isin(FOCAL)]
    src = [("bivariate", mf[mf.model == "bivariate"]), ("background", mf[mf.model == "background"]),
           ("ols_full", mfull[mfull.focal]), ("shap_slope", ms)]
    eq = []
    for name, t in src:
        for _, r in t.iterrows():
            df_ = int(r.k) - 1
            t90 = st.t.ppf(0.95, df_)
            lo90, hi90 = r.est - t90 * r.se, r.est + t90 * r.se
            for D_ in DELTAS:
                p_lo = st.t.sf((r.est + D_) / r.se, df_)     # H0: theta <= -Delta
                p_hi = st.t.cdf((r.est - D_) / r.se, df_)    # H0: theta >= +Delta
                equiv = (lo90 > -D_) and (hi90 < D_)
                nonzero = (r.ci_lo > 0) or (r.ci_hi < 0)
                concl = ("equivalent, nonzero" if equiv and nonzero else "equivalent" if equiv else
                         "non-equivalent, nonzero" if nonzero else "inconclusive")
                if not equiv and ((lo90 >= D_) or (hi90 <= -D_)):
                    concl = "exceeds Delta"
                eq.append(dict(estimator=name, feature=r.feature, delta=D_, k=int(r.k), est=r.est,
                               se=r.se, ci90_lo=lo90, ci90_hi=hi90, ci95_lo=r.ci_lo, ci95_hi=r.ci_hi,
                               p_tost=max(p_lo, p_hi), equivalent=bool(equiv), nonzero_95=bool(nonzero),
                               conclusion=concl))
    eq = pd.DataFrame(eq)
    eq["feature"] = pd.Categorical(eq.feature, FOCAL, ordered=True)
    eq = eq.sort_values(["delta", "estimator", "feature"]).reset_index(drop=True)
    eq.to_csv(RES / "equivalence_focal.csv", index=False)
    print(eq.groupby(["delta", "estimator"]).equivalent.sum().to_string())
    print(eq[eq.delta == 3][["estimator", "feature", "est", "ci90_lo", "ci90_hi", "conclusion"]].round(2).to_string())

    # ---- (3) school split + ICC
    sp = d[d.analysis == "split"]
    msp = pool(sp, ["feature", "component"])
    ic = pd.DataFrame(extra["icc"])
    ic.to_csv(RES / "icc_otl_by_system.csv", index=False)
    icm = ic[~ic.CNTRY.isin(EXCL)].groupby("feature").icc.agg(icc_median="median", icc_q1=lambda s: s.quantile(.25),
                                                              icc_q3=lambda s: s.quantile(.75))
    msp = msp.merge(icm, left_on="feature", right_index=True)
    sr = pd.DataFrame(extra["sdratio"])
    sr = sr[~sr.CNTRY.isin(EXCL)].assign(ratio=lambda z: z.sd_schoolmean / z.sd_item)
    msp = msp.merge(sr.groupby("feature").ratio.median().rename("sd_schoolmean_over_item_median"),
                    left_on="feature", right_index=True)
    msp.to_csv(RES / "meta_focal_schoolsplit.csv", index=False)
    print(msp.round(3).to_string())

    # ---- (4) per unit
    pu = pool(d[d.analysis.isin(["bg_perunit", "bg_perSD"])], ["feature", "analysis"])
    pu.to_csv(RES / "meta_focal_perunit.csv", index=False)
    print(pu.round(3).to_string())

    # ---- (5) teacher coverage
    prep = pd.read_csv(RES / "prep_summary.csv").set_index("CNTRY")
    cov = prep.pct_students_with_teacher_data
    sh = load_stats()
    sh = sh[(sh.stat == "SLOPE") & sh.feature.isin(SCH4)][["CNTRY", "feature", "est", "var_main"]]
    sh = sh.assign(se=np.sqrt(sh.var_main))[["CNTRY", "feature", "est", "se"]]
    bgm = old[(old.model == "background") & old.feature.isin(SCH4)][["CNTRY", "feature", "est", "se"]]
    olsm = full[full.feature.isin(SCH4)][["CNTRY", "feature", "est", "se"]]
    tc = []
    for name, t in [("shap_slope", sh), ("ols_full", olsm), ("background", bgm)]:
        t = t[~t.CNTRY.isin(EXCL)].dropna(subset=["est", "se"])
        t = t[t.se > 0].assign(coverage=lambda z: z.CNTRY.map(cov))
        for f in SCH4:
            g = t[t.feature == f]
            for subset, gg in [("all", g), (f"coverage>={COV_CUT:.0f}", g[g.coverage >= COV_CUT])]:
                r = reml(gg.est, gg.se ** 2)
                row = dict(estimator=name, feature=f, subset=subset, k=r["k"], est=r["est"], se=r["se"],
                           ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], tau=r["tau"], I2=r["I2"], p=r["p"],
                           coverage_min=gg.coverage.min())
                if subset == "all":
                    row.update(metareg(g.est, g.se ** 2, g.coverage))
                tc.append(row)
    tc = pd.DataFrame(tc)
    tc.to_csv(RES / "teacher_coverage_sensitivity.csv", index=False)
    print(tc.round(3).to_string())

    # ---- (11) participation sensitivity (flags of results/published_cil_means.csv, ICILS 2023 TR)
    pf = pd.read_csv(RES / "published_cil_means.csv", encoding="utf-8-sig").set_index("CNTRY")
    bad = lambda s: s.fillna("").str.match(r"\s*(Category 3|Not reported)")  # noqa: E731
    ex_stu = sorted(pf.index[bad(pf.student_participation_flag)])
    ex_tea = sorted(pf.index[bad(pf.teacher_participation_flag)])
    assert set(ex_stu) == {"NLD", "USA", "CHL"}, ex_stu
    print("participation exclusions: student", ex_stu, "| teacher", ex_tea, len(ex_tea))
    shall = load_stats()
    shall = shall[(shall.stat == "SLOPE") & shall.feature.isin(FOCAL)]
    shall = shall.assign(se=np.sqrt(shall.var_main))[["CNTRY", "feature", "est", "se"]]
    ps = []
    for name, t in [("shap_slope", shall), ("ols_full", full[full.feature.isin(FOCAL)][["CNTRY", "feature", "est", "se"]])]:
        t = t[~t.CNTRY.isin(EXCL)].dropna(subset=["est", "se"])
        t = t[t.se > 0]
        for f in FOCAL:
            g = t[t.feature == f]
            subs = [("all", g, ""), ("student_requirements_met", g[~g.CNTRY.isin(ex_stu)], " ".join(ex_stu))]
            if f in SCH4:
                subs.append(("teacher_requirements_met", g[~g.CNTRY.isin(ex_tea)], " ".join(ex_tea)))
            for subset, gg, exl in subs:
                r = reml(gg.est, gg.se ** 2)
                z = gg.est / gg.se
                ps.append(dict(estimator=name, feature=f, subset=subset, k=r["k"], est=r["est"], se=r["se"],
                               ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], pi_lo=r["pi_lo"], pi_hi=r["pi_hi"],
                               tau=r["tau"], I2=r["I2"], p=r["p"], n_pos_sig=int((z > 1.96).sum()),
                               n_neg_sig=int((z < -1.96).sum()), excluded=exl))
    ps = pd.DataFrame(ps)
    ps.to_csv(RES / "participation_sensitivity.csv", index=False)
    print(ps.drop(columns="excluded").round(3).to_string())

    # ---- (6) descriptives and missingness
    desc = []
    for c, g in df.groupby("CNTRY"):
        w = g[WGT].to_numpy(float)
        for f in FOCAL:
            x = g[f].to_numpy(float)
            ok = ~np.isnan(x)
            r = dict(CNTRY=c, feature=f, level=FEATURES[f][2], n=len(x), n_observed=int(ok.sum()),
                     pct_missing_w=100 * w[~ok].sum() / w.sum(), excluded_from_pooling=c in EXCL)
            if ok.any():
                r.update(mean_w=np.average(x[ok], weights=w[ok]), sd_w=wsd(x[ok], w[ok]))
                if f in ST4 or f == "P_DECEPT":
                    r["pct_top2_w"] = 100 * w[ok & (x >= 2)].sum() / w[ok].sum()
            desc.append(r)
    desc = pd.DataFrame(desc)
    desc.to_csv(RES / "focal_descriptives.csv", index=False)
    mi = []
    for c, g in df.groupby("CNTRY"):
        w = g[WGT].to_numpy(float)
        for f in FEAT:
            miss = g[f].isna().to_numpy()
            mi.append(dict(CNTRY=c, feature=f, pct_missing_w=100 * w[miss].sum() / w.sum(),
                           pct_missing=100 * miss.mean()))
    mi = pd.DataFrame(mi)
    m34 = mi[~mi.CNTRY.isin(EXCL)]
    ma = m34.groupby("feature", sort=False).agg(
        median_pct_missing=("pct_missing_w", "median"), max_pct_missing=("pct_missing_w", "max"),
        min_pct_missing=("pct_missing_w", "min"),
        k_entirely_missing=("pct_missing", lambda s: int((s >= 100).sum())),
        median_pct_missing_unweighted=("pct_missing", "median")).reset_index()
    ma["system_max"] = m34.loc[m34.groupby("feature", sort=False).pct_missing_w.idxmax()].CNTRY.to_numpy()
    ma["systems_entirely_missing"] = ma.feature.map(
        m34[m34.pct_missing >= 100].groupby("feature").CNTRY.apply(lambda s: " ".join(sorted(s)))).fillna("")
    ma["block"] = ma.feature.map(lambda f: FEATURES[f][0])
    ma.to_csv(RES / "missingness_all.csv", index=False)
    print(ma.sort_values("median_pct_missing", ascending=False).head(12).round(1).to_string())
    om = pd.DataFrame(extra["miss"])
    pm = pool(om, ["feature"]).rename(columns=lambda s: s if s == "feature" else "pooled_" + s)
    om = om.merge(pm, on="feature", how="left")
    om.to_csv(RES / "otl_missing_cil.csv", index=False)
    print(pm.round(2).to_string())

    # ---- (8) composites
    cp = d[d.analysis.str.startswith("comp_")].copy()
    cp["step"] = cp.analysis.str[-1]
    s6 = full[full.feature.isin(OTL_S6 + OTL_O6)].assign(step="e_item")
    mcp = pool(pd.concat([cp, s6], ignore_index=True), ["feature", "step"])
    mcp.to_csv(RES / "meta_otl_composites.csv", index=False)
    print(mcp.round(2).to_string())

    # ---- (9) teacher items
    ti = d[d.analysis.str.startswith("titem_")].copy()
    ti["step"] = ti.analysis.str[-1]
    ref = pd.concat([seq[(seq.feature == "T_INFOEMPH") & seq.step.isin(["a", "b"])],
                     full[full.feature == "T_INFOEMPH"].assign(step="e")], ignore_index=True)
    mti = pool(pd.concat([ti, ref], ignore_index=True), ["feature", "step"])
    mti["source"] = mti.feature.map({**TITEMS, "T_INFOEMPH": "mean of IT3G19C/M/N"})
    a34 = alpha[~alpha.CNTRY.isin(EXCL)]
    for col in ["alpha_unweighted", "alpha_TOTWGTT", "r_CM", "r_CN", "r_MN"]:
        mti[f"{col}_median"] = a34[col].median()
    mti["alpha_unweighted_min"] = a34.alpha_unweighted.min()
    mti["alpha_unweighted_max"] = a34.alpha_unweighted.max()
    mti.to_csv(RES / "teacher_items_split.csv", index=False)
    alpha.to_csv(RES / "teacher_items_alpha.csv", index=False)
    print(mti.round(3).to_string())
    print(a34.describe().round(3).to_string())

    # ---- (12) response style
    rs = d[d.analysis.str.startswith("rs_")]
    ref = pd.concat([d[(d.analysis == "bg_perSD")].assign(analysis="bg_reference"),
                     full[full.feature.isin(ST4)].assign(analysis="full_reference")], ignore_index=True)
    mrs = pool(pd.concat([rs, ref], ignore_index=True), ["feature", "analysis"])
    mrs["definition"] = mrs.analysis.map({
        "bg_reference": "background model (step-(b) set: background excl. expected education + peers), per SD of the item",
        "full_reference": "fully adjusted model (1), per SD of the item",
        "rs_bg_ctrl12": "background model + student mean of the 12 OTL items (>= 8 valid), per SD of the item",
        "rs_bg_contrast": "background model, predictor = outside minus at-school item, per SD of the contrast",
        "rs_bg_contrast_level": "background model, contrast + pair mean (level) entered together; contrast per SD",
        "rs_full_contrast": "fully adjusted model with the two items replaced by contrast and pair mean; contrast per SD"})
    mrs.to_csv(RES / "otl_response_style.csv", index=False)
    print(mrs.drop(columns="definition").round(3).to_string())

    # ---- (13) reporting extras
    shs = load_stats()
    shs = shs[(shs.stat == "SLOPE") & shs.feature.isin(FOCAL) & ~shs.CNTRY.isin(EXCL)]
    shs = shs.assign(se=np.sqrt(shs.var_main))[["CNTRY", "feature", "est", "se"]]
    msl = pd.read_csv(RES / "meta_SLOPE.csv")
    msl = msl[msl.analysis == "main"].set_index("feature")
    rx = []
    for name, t, mt in [("shap_slope", shs, msl), ("ols_full", full[~full.CNTRY.isin(EXCL)], mfull.set_index("feature"))]:
        for f in FOCAL:
            g = t[(t.feature == f)].dropna(subset=["est", "se"])
            g = g[g.se > 0]
            r = reml(g.est, g.se ** 2)
            zt = r["est"] / r["se"]
            sdev = (g.est - r["est"]) / np.sqrt(g.se ** 2 + r["tau"] ** 2)
            m_ = mt.loc[f]
            assert abs(m_.est - r["est"]) < 1e-8, (name, f)
            rx.append(dict(estimator=name, feature=f, k=r["k"], est=r["est"], se_hksj=r["se"], t=zt,
                           df=r["k"] - 1, p_exact=2 * st.t.sf(abs(zt), r["k"] - 1),
                           median_system_se=g.se.median(), mde80_system=2.8 * g.se.median(),
                           shapiro_W=st.shapiro(sdev).statistic, shapiro_p=st.shapiro(sdev).pvalue,
                           skew_std_dev=st.skew(sdev), skew_raw_est=st.skew(g.est)))
    rx = pd.DataFrame(rx)
    rx.to_csv(RES / "reporting_extras.csv", index=False)
    print(rx.to_string(float_format=lambda v: f"{v:.4g}"))

    # ---- (14) subject-restricted teacher emphasis
    tsj = d[d.analysis.str.startswith("tsubj_")].copy()
    tsj["step"] = tsj.analysis.str[-1]
    ref = pd.concat([seq[(seq.feature == "T_INFOEMPH") & seq.step.isin(["a", "b"])],
                     full[full.feature == "T_INFOEMPH"].assign(step="e")], ignore_index=True)
    mts = pool(pd.concat([tsj, ref], ignore_index=True), ["feature", "step"])
    t34 = tshare[~tshare.CNTRY.isin(EXCL)]
    mts["share_subject_median"] = t34.share_subject.median()
    mts["share_subject_min"] = t34.share_subject.min()
    mts["share_subject_max"] = t34.share_subject.max()
    mts["share_subject_TOTWGTT_median"] = t34.share_subject_TOTWGTT.median()
    stud_cov = df[~df.CNTRY.isin(EXCL)].groupby("CNTRY").apply(
        lambda g: 100 * g[WGT][g.T_INFOEMPH_SUBJ.notna()].sum() / g[WGT].sum(), include_groups=False)
    mts["pct_students_with_subject_emphasis_median"] = stud_cov.median()
    mts.to_csv(RES / "teacher_subject_emphasis.csv", index=False)
    tshare.merge(stud_cov.rename("pct_students_with_subject_emphasis_w").reset_index(), on="CNTRY",
                 how="left").to_csv(RES / "teacher_subject_share.csv", index=False)
    print(mts.round(3).to_string())
    print(t34.describe().round(3).to_string())

    # ---- (10) wording
    meta = {}
    for pre, vs in [("BSG", ["IS3G20AC1", "IS3G20AE1", "IS3G20AC2", "IS3G20AE2"]),
                    ("BTG", ["IT3G19C", "IT3G19M", "IT3G19N", "IT3G08F", "IT3G09AF", "IT3G09AG"]),
                    ("BCG", ["IP3G11I"])]:
        _, m = pyreadstat.read_sav(str(RAW / f"{pre}AUTI3.sav"), metadataonly=True)
        for v in vs:
            vl = {k: s for k, s in m.variable_value_labels.get(v, {}).items() if k < 8}
            meta[v] = (m.column_names_to_labels.get(v, ""), "; ".join(f"{int(k)} = {s}" for k, s in vl.items()))
    src_ = {"OTL_S_EVAL": ["IS3G20AC1"], "OTL_S_SCAM": ["IS3G20AE1"], "OTL_O_EVAL": ["IS3G20AC2"],
            "OTL_O_SCAM": ["IS3G20AE2"], "T_INFOEMPH": ["IT3G19C", "IT3G19M", "IT3G19N"],
            "T_ITE_INFO": ["IT3G08F"], "T_PD_INFO": ["IT3G09AF", "IT3G09AG"], "P_DECEPT": ["IP3G11I"]}
    coding = {
        "OTL_S_EVAL": "reversed: 0 = not at all ... 3 = to a large extent",
        "OTL_S_SCAM": "reversed: 0 = not at all ... 3 = to a large extent",
        "OTL_O_EVAL": "reversed: 0 = not at all ... 3 = to a large extent",
        "OTL_O_SCAM": "reversed: 0 = not at all ... 3 = to a large extent",
        "T_INFOEMPH": "each item reversed to 0 = no emphasis ... 3 = strong emphasis; teacher mean (>= 2 of 3 valid); school mean weighted by WGTFAC3T",
        "T_ITE_INFO": "1 = Yes, 0 = No or I cannot remember (8/9 = missing); school share weighted by WGTFAC3T",
        "T_PD_INFO": "each item shifted to 0 = did not participate, 1 = once only, 2 = more than once; teacher mean (>= 1 of 2 valid); school mean weighted by WGTFAC3T",
        "P_DECEPT": "reversed: 0 = not important ... 3 = very important (principal questionnaire)"}
    sl = short_labels()
    wd = [dict(code=f, short_label=sl.get(f, ""), label=FEATURES[f][1], level=FEATURES[f][2],
               source_variables=" + ".join(src_[f]),
               idb_variable_labels=" | ".join(meta[v][0] for v in src_[f]),
               response_categories=" | ".join(sorted({meta[v][1] for v in src_[f]})),
               coding_in_analysis=coding[f],
               note="Wording = IDB variable label (SPSS metadata and codebook ICILS2023MS_Codebook_PUF); "
                    "full questionnaire stems are not part of the IDB files") for f in FOCAL]
    pd.DataFrame(wd).to_csv(RES / "focal_item_wording.csv", index=False)
    print(f"total {time.time() - t0:.0f}s")


def short_labels():
    from labels import SHORTLAB
    return dict(SHORTLAB)


if __name__ == "__main__":
    main()
