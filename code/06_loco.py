"""Cross-national transferability: leave-one-country-out (LOCO) XGBoost (GPU).

34 systems (the DNW benchmark is excluded because it overlaps with DEU). Training weights:
senate weights SENWGTS (each system sums to 500), so every system contributes equally.
Hyperparameters: Optuna (20 trials) with 5-fold country-grouped inner CV on PV1.
For each held-out system and each PV a model is trained on the other 33 systems.
Metrics in the held-out system (TOTWGTS): R2 (accuracy incl. level), squared weighted
correlation (rank/pattern transfer), and mean bias (calibration-in-the-large).

Outputs: results/loco_metrics.csv, results/loco_hparams.json
"""
import json

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold

import models as M
from config import PROC, RES, FEAT, PVS, SEED, WGT
from survey import weighted_r2

GPU = dict(M.XGB_FIXED, device="cuda", nthread=8)


def wcorr2(y, p, w):
    my, mp = np.average(y, weights=w), np.average(p, weights=w)
    c = np.average((y - my) * (p - mp), weights=w)
    return c ** 2 / (np.average((y - my) ** 2, weights=w) * np.average((p - mp) ** 2, weights=w))


def main():
    df = pd.read_parquet(PROC / "analysis.parquet")
    df = df[df.CNTRY != "DNW"].reset_index(drop=True)
    X = df[FEAT].to_numpy(np.float32)
    cn = df.CNTRY.to_numpy()
    sw = M.norm_w(df.SENWGTS)
    y1 = df[PVS[0]].to_numpy(float)

    folds = list(GroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, groups=cn))

    def obj(trial):
        p = M.xgb_params(trial)
        pred = np.zeros(len(y1))
        rounds = []
        for tr, va in folds:
            dtr = xgb.DMatrix(X[tr], label=y1[tr], weight=sw[tr])
            dva = xgb.DMatrix(X[va], label=y1[va], weight=sw[va])
            b = xgb.train({**GPU, **p}, dtr, M.MAX_ROUNDS, evals=[(dva, "va")],
                          early_stopping_rounds=M.ES, verbose_eval=False)
            rounds.append(b.best_iteration + 1)
            pred[va] = b.predict(dva, iteration_range=(0, b.best_iteration + 1))
        trial.set_user_attr("n_rounds", int(np.mean(rounds)))
        return weighted_r2(y1, pred, sw)

    stdy = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    stdy.optimize(obj, n_trials=20)
    best = dict(stdy.best_params, n_rounds=stdy.best_trial.user_attrs["n_rounds"])
    json.dump({**best, "inner_r2": stdy.best_value}, open(RES / "loco_hparams.json", "w"), indent=1)
    print("LOCO tuned", best, round(stdy.best_value, 3), flush=True)
    p = {k: v for k, v in best.items() if k != "n_rounds"}

    rows = []
    for c in sorted(set(cn)):
        tr, te = cn != c, cn == c
        wt = df.loc[te, WGT].to_numpy(float)
        for pv in PVS:
            y = df[pv].to_numpy(float)
            b = xgb.train({**GPU, **p}, xgb.DMatrix(X[tr], label=y[tr], weight=sw[tr]), best["n_rounds"])
            pr = b.predict(xgb.DMatrix(X[te]))
            rows.append({"CNTRY": c, "pv": pv, "r2": weighted_r2(y[te], pr, wt),
                         "corr2": wcorr2(y[te], pr, wt),
                         "bias": np.average(pr - y[te], weights=wt)})
        print(c, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "loco_metrics.csv", index=False)
    print(out.groupby("CNTRY")[["r2", "corr2", "bias"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
