"""Nested leave-one-country-out (LOCO) transfer analysis.

In 06_loco.py the LOCO hyperparameters were tuned once on all 34 systems, i.e. including the
system that is later held out. Here, for every held-out system c, the hyperparameters are tuned on
the 33 training systems only, with exactly the procedure of 06_loco.py otherwise: Optuna TPE
(seed SEED, 20 trials, search space models.xgb_params), 5-fold country-grouped inner CV
(GroupKFold(5, shuffle=True, random_state=SEED) over the 33 training systems) on PV1 with early
stopping (MAX_ROUNDS, ES), n_rounds = mean best iteration of the best trial, senate weights
SENWGTS normalised over the 34-system file as in 06_loco.py. A model is then trained on the 33
systems for each of the 5 PVs and evaluated in the held-out system with TOTWGTS: R^2, squared
weighted correlation and mean bias (prediction minus PV).

For a device-matched paired comparison, the non-nested model (results/loco_hparams.json) is refitted
in the same process on the same device (06_loco.py used the GPU; results/loco_metrics.csv is kept beside it).

Design-based SEs: the held-out predictions are kept fixed and the evaluation statistics are
recomputed with each of the 75 JRR replicate weights (SRWGT01-75) of the held-out system;
SV = sum_r (theta_r - theta)^2 per PV, combined over PVs with Rubin's rules (survey.combine_pv).
The same is done for the paired difference nested minus non-nested.

Subset: the 34 systems are put in a seeded random order (numpy default_rng(SEED + 15).permutation
of the alphabetically sorted codes); the analysis covers the first N_TARGET systems of that order,
with N_TARGET fixed from a runtime estimate (budget <= 3 h; see results/loco_nested_meta.json).

All fits: 'hist'; the 'device' column records cpu/cuda per system (in the full run every
completed system ran on the GPU, as 06_loco.py); nested and non-nested refits of a system always
share the device. One subprocess per held-out system; workers claim systems
through exclusive claim files so that workers can be added while running.

Run:   python 15_loco_nested.py --order                 (print the seeded order)
       python 15_loco_nested.py --worker <nthread> [cpu|cuda]  (claims systems until N_TARGET reached)
       python 15_loco_nested.py --one <CNTRY> <nthread>
       python 15_loco_nested.py --collect
Outputs: results/loco_nested/<CNTRY>.pkl, results/loco_nested.csv, results/loco_nested_meta.json
Does not modify any existing script or results file.
"""
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold

import models as M
from config import PROC, RES, FEAT, PVS, REPW, SEED, WGT
from survey import weighted_r2, combine_pv

optuna.logging.set_verbosity(optuna.logging.WARNING)
CACHE = RES / "loco_nested"
CACHE.mkdir(exist_ok=True)
TARGET_FILE = CACHE / "N_TARGET.txt"
N_TRIALS = 20


def wcorr2(y, p, w):
    my, mp = np.average(y, weights=w), np.average(p, weights=w)
    c = np.average((y - my) * (p - mp), weights=w)
    return c ** 2 / (np.average((y - my) ** 2, weights=w) * np.average((p - mp) ** 2, weights=w))


def metrics(y, p, w):
    return np.array([weighted_r2(y, p, w), wcorr2(y, p, w), np.average(p - y, weights=w)])


MET = ["r2", "corr2", "bias"]


def order():
    df = pd.read_parquet(PROC / "analysis.parquet", columns=["CNTRY"])
    sysl = sorted(set(df.CNTRY) - {"DNW"})
    return [str(x) for x in np.random.default_rng(SEED + 15).permutation(sysl)]


def load():
    df = pd.read_parquet(PROC / "analysis.parquet")
    return df[df.CNTRY != "DNW"].reset_index(drop=True)


def run_one(c, nthread, device="cpu"):
    par0 = dict(M.XGB_FIXED, device=device, nthread=nthread)
    df = load()
    X = df[FEAT].to_numpy(np.float32)
    cn = df.CNTRY.to_numpy()
    sw = M.norm_w(df.SENWGTS)
    tr, te = cn != c, cn == c
    Xtr, cntr, swtr = X[tr], cn[tr], sw[tr]
    y1 = df[PVS[0]].to_numpy(float)[tr]
    folds = list(GroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(Xtr, groups=cntr))
    t0 = time.time()

    def obj(trial):
        p = M.xgb_params(trial)
        pred = np.zeros(len(y1))
        rounds = []
        for a, b_ in folds:
            dtr = xgb.DMatrix(Xtr[a], label=y1[a], weight=swtr[a])
            dva = xgb.DMatrix(Xtr[b_], label=y1[b_], weight=swtr[b_])
            b = xgb.train({**par0, **p}, dtr, M.MAX_ROUNDS, evals=[(dva, "va")],
                          early_stopping_rounds=M.ES, verbose_eval=False)
            rounds.append(b.best_iteration + 1)
            pred[b_] = b.predict(dva, iteration_range=(0, b.best_iteration + 1))
        trial.set_user_attr("n_rounds", int(np.mean(rounds)))
        print(c, "trial", trial.number, f"{time.time() - t0:.0f}s", flush=True)
        return weighted_r2(y1, pred, swtr)

    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(obj, n_trials=N_TRIALS)
    best = dict(st.best_params, n_rounds=st.best_trial.user_attrs["n_rounds"])
    t_tune = time.time() - t0
    print(c, "tuned", best, round(st.best_value, 4), f"{t_tune:.0f}s", flush=True)

    nn = json.load(open(RES / "loco_hparams.json"))
    nn = {k: v for k, v in nn.items() if k != "inner_r2"}
    models = {"nested": best, "nonnested": nn}
    g = df[te]
    wt = g[WGT].to_numpy(float)
    R = g[REPW].to_numpy(float)
    preds = {m: {} for m in models}
    for m, hp in models.items():
        p = {k: v for k, v in hp.items() if k != "n_rounds"}
        p["max_depth"] = int(p["max_depth"])
        for pv in PVS:
            y = df[pv].to_numpy(float)
            b = xgb.train({**par0, **p}, xgb.DMatrix(Xtr, label=y[tr], weight=swtr), int(hp["n_rounds"]))
            preds[m][pv] = b.predict(xgb.DMatrix(X[te])).astype(float)
    # metrics and JRR on fixed predictions
    out = {"CNTRY": c, "device": device, "best": best, "inner_r2": st.best_value, "secs_tune": t_tune,
           "secs": time.time() - t0, "preds": preds, "IDSTUD": g.IDSTUD.to_numpy()}
    for m in ["nested", "nonnested", "diff"]:
        full, sv = [], []
        for pv in PVS:
            y = g[pv].to_numpy(float)
            if m == "diff":
                f_ = metrics(y, preds["nested"][pv], wt) - metrics(y, preds["nonnested"][pv], wt)
                reps = np.array([metrics(y, preds["nested"][pv], R[:, r]) -
                                 metrics(y, preds["nonnested"][pv], R[:, r]) for r in range(R.shape[1])])
            else:
                f_ = metrics(y, preds[m][pv], wt)
                reps = np.array([metrics(y, preds[m][pv], R[:, r]) for r in range(R.shape[1])])
            full.append(f_)
            sv.append(((reps - f_) ** 2).sum(0))
        est, se, u, bvar = combine_pv(full, sv)
        out[m] = {"by_pv": np.array(full), "est": est, "se": se, "samp_var": u, "imp_var": bvar}
    print(c, {m: dict(zip(MET, np.round(out[m]["est"], 4))) for m in ["nested", "nonnested", "diff"]},
          f"{out['secs']:.0f}s", flush=True)
    return out


def n_target():
    try:
        return int(TARGET_FILE.read_text().strip())
    except Exception:
        return 34


def worker(nthread, device="cpu"):
    here = Path(__file__).parent
    logdir = RES / "logs"
    logdir.mkdir(exist_ok=True)
    od = order()
    for i, c in enumerate(od):
        if i >= n_target():
            break
        if (CACHE / f"{c}.pkl").exists():
            continue
        try:
            fd = os.open(CACHE / f"{c}.claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            continue
        t0 = time.time()
        with open(logdir / f"15_loco_nested_{c}.log", "w") as f:
            r = subprocess.run([sys.executable, str(here / Path(__file__).name), "--one", c, str(nthread), device],
                               stdout=f, stderr=subprocess.STDOUT, cwd=here)
        print(f"[{i}] {c}: exit={r.returncode} {time.time() - t0:.0f}s", flush=True)
        if r.returncode != 0:
            os.remove(CACHE / f"{c}.claim")


def within_corr2():
    """Within-system squared correlation of the out-of-fold XGBoost predictions."""
    oof = pd.read_parquet(RES / "oof_predictions.parquet")
    dd = pd.read_parquet(PROC / "analysis.parquet", columns=["CNTRY", "IDSTUD", "TOTWGTS"] + PVS).merge(
        oof[["CNTRY", "IDSTUD"] + [f"XGB_pv{j}" for j in range(1, 6)]], on=["CNTRY", "IDSTUD"])
    wc2 = {}
    for c, g in dd.groupby("CNTRY"):
        w = g.TOTWGTS.to_numpy()
        wc2[c] = np.mean([wcorr2(g[pv].to_numpy(), g[f"XGB_pv{j + 1}"].to_numpy(), w)
                          for j, pv in enumerate(PVS)])
    return pd.Series(wc2)


def collect():
    od = order()
    nt = n_target()
    loco06 = pd.read_csv(RES / "loco_metrics.csv").groupby("CNTRY")[MET].mean()
    cvm = pd.read_csv(RES / "cv_metrics.csv")
    wi = cvm[cvm.model == "XGB"].groupby("CNTRY").r2.mean()
    wc = within_corr2()
    rows = []
    for i, c in enumerate(od):
        row = {"CNTRY": c, "order": i, "in_subset": i < nt,
               "loco06_r2": loco06.loc[c, "r2"], "loco06_corr2": loco06.loc[c, "corr2"],
               "loco06_bias": loco06.loc[c, "bias"], "within_r2": wi[c], "within_corr2": wc[c]}
        pk = CACHE / f"{c}.pkl"
        if pk.exists():
            o = pickle.load(open(pk, "rb"))
            row.update({"done": True, "device": o.get("device", "cpu"), "inner_r2": o["inner_r2"], "secs": o["secs"],
                        **{f"hp_{k}": v for k, v in o["best"].items()}})
            for m in ["nested", "nonnested", "diff"]:
                pre = {"nested": "nest", "nonnested": "nn_refit", "diff": "diff"}[m]
                for j, k in enumerate(MET):
                    row[f"{pre}_{k}"] = o[m]["est"][j]
                    row[f"{pre}_{k}_se"] = o[m]["se"][j]
        else:
            row["done"] = False
        rows.append(row)
    d = pd.DataFrame(rows)
    d.to_csv(RES / "loco_nested.csv", index=False)
    s = d[d.done].copy()
    nh = json.load(open(RES / "loco_hparams.json"))
    summ = {"devices": s.device.value_counts().to_dict(), "n_target": nt, "n_done": int(len(s)), "systems": s.CNTRY.tolist(), "order": od,
            "subset_complete": bool(d[d.in_subset].done.all()),
            "nonnested_hparams": nh,
            "runtime_hours_sum_over_systems": float(s.secs.sum() / 3600),
            "runtime_min_per_system_median": float(s.secs.median() / 60)}
    for lab, pre in [("nested", "nest"), ("nonnested_refit", "nn_refit"), ("nonnested_loco06", "loco06")]:
        summ[lab] = {"r2_median": float(s[f"{pre}_r2"].median()),
                     "corr2_median": float(s[f"{pre}_corr2"].median()),
                     "absbias_median": float(s[f"{pre}_bias"].abs().median()),
                     "n_corr2_gt_within_corr2": int((s[f"{pre}_corr2"] > s.within_corr2).sum()),
                     "n_r2_neg": int((s[f"{pre}_r2"] < 0).sum())}
    p34 = loco06.join(wc.rename("wc2")).drop(index="DNW", errors="ignore")
    summ["nonnested_loco06_all34"] = {"r2_median": float(p34.r2.median()), "corr2_median": float(p34.corr2.median()),
                                     "absbias_median": float(p34.bias.abs().median()),
                                     "n_corr2_gt_within_corr2": int((p34.corr2 > p34.wc2).sum()), "n": int(len(p34))}
    summ["within_subset"] = {"r2_median": float(s.within_r2.median()), "corr2_median": float(s.within_corr2.median())}
    summ["diff_nested_minus_nonnested_refit"] = {
        "r2_median": float(s.diff_r2.median()), "corr2_median": float(s.diff_corr2.median()),
        "absbias_median_change": float((s.nest_bias.abs() - s.nn_refit_bias.abs()).median()),
        "n_corr2_diff_sig": int((s.diff_corr2.abs() > 1.96 * s.diff_corr2_se).sum()),
        "n_r2_diff_sig": int((s.diff_r2.abs() > 1.96 * s.diff_r2_se).sum()),
        "max_abs_corr2_diff": float(s.diff_corr2.abs().max()), "max_abs_r2_diff": float(s.diff_r2.abs().max())}
    summ["jrr_se"] = {"corr2_se_median": float(s.nest_corr2_se.median()), "corr2_se_min": float(s.nest_corr2_se.min()),
                      "corr2_se_max": float(s.nest_corr2_se.max()), "bias_se_median": float(s.nest_bias_se.median()),
                      "bias_se_min": float(s.nest_bias_se.min()), "bias_se_max": float(s.nest_bias_se.max()),
                      "r2_se_median": float(s.nest_r2_se.median()),
                      "n_bias_sig": int((s.nest_bias.abs() > 1.96 * s.nest_bias_se).sum()),
                      "n_corr2_sig_gt_within": int((s.nest_corr2 - s.within_corr2 > 1.96 * s.nest_corr2_se).sum())}
    summ["check_refit_vs_loco06"] = {"corr2_maxabs": float((s.nn_refit_corr2 - s.loco06_corr2).abs().max()),
                                      "r2_maxabs": float((s.nn_refit_r2 - s.loco06_r2).abs().max()),
                                      "bias_maxabs": float((s.nn_refit_bias - s.loco06_bias).abs().max())}
    summ["nested_hp_n_rounds"] = {"median": float(s.hp_n_rounds.median()), "min": int(s.hp_n_rounds.min()),
                                  "max": int(s.hp_n_rounds.max())}
    summ["nested_hp_max_depth_counts"] = {str(k): int(v) for k, v in s.hp_max_depth.value_counts().items()}
    json.dump(summ, open(RES / "loco_nested_meta.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))
    print(s[["CNTRY", "nest_r2", "nn_refit_r2", "loco06_r2", "nest_corr2", "nest_corr2_se", "nn_refit_corr2",
             "within_corr2", "nest_bias", "nest_bias_se", "nn_refit_bias", "hp_n_rounds", "hp_max_depth"]]
          .round(3).to_string())


if __name__ == "__main__":
    a = sys.argv[1]
    if a == "--order":
        print(order())
    elif a == "--one":
        c, nth = sys.argv[2], int(sys.argv[3])
        out = run_one(c, nth, sys.argv[4] if len(sys.argv) > 4 else "cpu")      # compute first, then write the cache file
        with open(CACHE / f"{c}.pkl", "wb") as f:
            pickle.dump(out, f)
    elif a == "--worker":
        worker(int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "cpu")
    elif a == "--collect":
        collect()
