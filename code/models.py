"""Model definitions: weighted ridge (RIDGE), penalised random-intercept model (RLMM),
XGBoost (XGB) and mixed-effects XGBoost via the MERF algorithm (MERF)."""
import warnings

import numpy as np
import optuna
import xgboost as xgb
from sklearn.model_selection import GroupKFold

from config import SEED
from survey import weighted_r2

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

XGB_FIXED = dict(objective="reg:squarederror", tree_method="hist", device="cpu", nthread=4,
                 eval_metric="rmse", seed=SEED)
MAX_ROUNDS = 3000
ES = 100


def norm_w(w):
    w = np.asarray(w, dtype=float)
    return w * (len(w) / w.sum())


def group_folds(groups, k, seed):
    gkf = GroupKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(gkf.split(np.zeros(len(groups)), groups=groups))


# ----------------------------------------------------------------------------- linear
class LinearPrep:
    """Winsorisation at the training 0.5th/99.5th percentiles, median imputation, missing
    indicators and standardisation -- all fitted on the training data only."""

    def fit(self, X, w):
        self.lo = np.nanpercentile(X, 0.5, axis=0)
        self.hi = np.nanpercentile(X, 99.5, axis=0)
        X = self._clip(X)
        self.med = np.nanmedian(X, axis=0)
        self.med = np.where(np.isnan(self.med), 0.0, self.med)
        self.miss_cols = np.where(np.isnan(X).any(axis=0))[0]
        Z = self._fill(X)
        self.mu = np.average(Z, axis=0, weights=w)
        self.sd = np.sqrt(np.average((Z - self.mu) ** 2, axis=0, weights=w))
        self.sd[self.sd == 0] = 1.0
        return self

    def _clip(self, X):
        return np.where(np.isnan(X), np.nan, np.clip(X, self.lo, self.hi))

    def _fill(self, X):
        X = self._clip(X)
        Z = np.where(np.isnan(X), self.med, X)
        M = np.isnan(X[:, self.miss_cols]).astype(float)
        return np.hstack([Z, M])

    def transform(self, X):
        return (self._fill(X) - self.mu) / self.sd


RIDGE_ALPHAS = [0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000]


def _ridge_beta(Z, y, w, pen):
    A = Z.T @ (w[:, None] * Z) + np.diag(pen)
    return np.linalg.solve(A, Z.T @ (w * y))


def _school_var_ratio(r, g):
    """sigma_e^2 / sigma_b^2 for residuals r via the unbalanced one-way ANOVA (moment)
    estimator (Searle, Casella & McCulloch, 1992)."""
    sch, idx = np.unique(g, return_inverse=True)
    nj = np.bincount(idx).astype(float)
    J, N = len(sch), len(r)
    mj = np.bincount(idx, weights=r) / nj
    ssw = ((r - mj[idx]) ** 2).sum()
    ssb = (nj * (mj - r.mean()) ** 2).sum()
    msw, msb = ssw / (N - J), ssb / (J - 1)
    n0 = (N - (nj ** 2).sum() / N) / (J - 1)
    tau2 = max((msb - msw) / n0, 1e-6 * msw)
    return msw / tau2


class PenalizedLinear:
    """Weighted ridge regression (mixed=False) or weighted ridge with school random intercepts
    (mixed=True; penalised random-intercept model). Penalised quasi-likelihood with the
    variance ratio estimated by the ANOVA moment estimator on the ridge residuals. Predictions for new schools use
    the fixed part only."""

    def __init__(self, alpha, mixed=False):
        self.alpha, self.mixed = alpha, mixed

    def fit(self, X, y, w, g=None):
        self.p = LinearPrep().fit(X, w)
        Z = np.hstack([np.ones((len(X), 1)), self.p.transform(X)])
        pen = np.r_[0.0, np.full(Z.shape[1] - 1, self.alpha)]
        beta = _ridge_beta(Z, y, w, pen)
        if self.mixed:
            lam = _school_var_ratio(y - Z @ beta, g)
            sch, idx = np.unique(g, return_inverse=True)
            D = np.zeros((len(X), len(sch)))
            D[np.arange(len(X)), idx] = 1.0
            ZD = np.hstack([Z, D])
            # weights are normalised to mean 1, so lam acts on the residual-variance scale
            beta = _ridge_beta(ZD, y, w, np.r_[pen, np.full(len(sch), lam)])[: Z.shape[1]]
        self.beta = beta
        return self

    def predict(self, X):
        return np.hstack([np.ones((len(X), 1)), self.p.transform(X)]) @ self.beta


def tune_ridge(X, y, w, g, mixed=False, k=3, seed=SEED):
    folds = group_folds(g, k, seed)
    best, best_r2 = None, -np.inf
    for a in RIDGE_ALPHAS:
        pred = np.zeros(len(y))
        for tr, va in folds:
            pred[va] = PenalizedLinear(a, mixed).fit(X[tr], y[tr], w[tr], g[tr]).predict(X[va])
        r2 = weighted_r2(y, pred, w)
        if r2 > best_r2:
            best, best_r2 = a, r2
    return best, best_r2


# ----------------------------------------------------------------------------- XGBoost
def xgb_params(trial):
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        max_depth=trial.suggest_int("max_depth", 2, 6),
        min_child_weight=trial.suggest_float("min_child_weight", 1, 50, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.1, 50, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        gamma=trial.suggest_float("gamma", 1e-3, 5, log=True),
    )


def xgb_cv_score(X, y, w, groups, params, k=3, seed=SEED):
    """Inner grouped CV with early stopping; returns (weighted R2, mean best rounds)."""
    preds = np.zeros(len(y))
    rounds = []
    for tr, va in group_folds(groups, k, seed):
        dtr = xgb.DMatrix(X[tr], label=y[tr], weight=w[tr])
        dva = xgb.DMatrix(X[va], label=y[va], weight=w[va])
        b = xgb.train({**XGB_FIXED, **params}, dtr, MAX_ROUNDS, evals=[(dva, "va")],
                      early_stopping_rounds=ES, verbose_eval=False)
        rounds.append(b.best_iteration + 1)
        preds[va] = b.predict(dva, iteration_range=(0, b.best_iteration + 1))
    return weighted_r2(y, preds, w), int(np.mean(rounds))


def tune_xgb(X, y, w, groups, n_trials, seed=SEED):
    def obj(trial):
        r2, nr = xgb_cv_score(X, y, w, groups, xgb_params(trial), seed=seed)
        trial.set_user_attr("n_rounds", nr)
        return r2

    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    st.enqueue_trial(dict(learning_rate=0.03, max_depth=4, min_child_weight=10, subsample=0.8,
                          colsample_bytree=0.7, reg_lambda=5, reg_alpha=0.01, gamma=0.01))
    st.optimize(obj, n_trials=n_trials)
    best = dict(st.best_params)
    best["n_rounds"] = st.best_trial.user_attrs["n_rounds"]
    return best, st.best_value


def fit_xgb(X, y, w, best):
    p = {k: v for k, v in best.items() if k != "n_rounds"}
    return xgb.train({**XGB_FIXED, **p}, xgb.DMatrix(X, label=y, weight=w), best["n_rounds"])


# ----------------------------------------------------------------------------- MERF-XGB
def _school_blup(r, g, w, sig2, tau2):
    """Random-intercept BLUPs for residuals r: b_j = tau2 / (tau2 + sig2 / n_j) * rbar_j,
    with n_j the (normalised) within-school weight sum and rbar_j the weighted mean residual.
    Returns per-observation b, per-school (b_j, posterior variance v_j)."""
    sch, idx = np.unique(g, return_inverse=True)
    nj = np.bincount(idx, weights=w)
    rbar = np.bincount(idx, weights=w * r) / nj
    shrink = tau2 / (tau2 + sig2 / nj)
    bj = shrink * rbar
    vj = tau2 * (1 - shrink)
    return bj[idx], bj, vj, idx


def fit_merf(X, y, w, g, best, n_iter=5):
    """Mixed-effects XGBoost (MERF algorithm of Hajjem, Bellavance & Larocque, 2014, with
    XGBoost as the fixed-effects learner): y = f(X) + b_school + e. EM alternates between fitting
    f on y - b (weighted) and updating BLUPs b and variance components (sig2, tau2)."""
    b_obs = np.zeros(len(y))
    sig2 = np.average((y - np.average(y, weights=w)) ** 2, weights=w)
    tau2 = 0.1 * sig2
    for _ in range(n_iter):
        f = fit_xgb(X, y - b_obs, w, best)
        r = y - f.predict(xgb.DMatrix(X))
        b_obs, bj, vj, idx = _school_blup(r, g, w, sig2, tau2)
        e = r - b_obs
        sig2 = np.average(e ** 2, weights=w) + np.average(vj[idx], weights=w)
        tau2 = max(np.mean(bj ** 2 + vj), 1e-6 * sig2)
    f = fit_xgb(X, y - b_obs, w, best)
    return f, {"sig2": sig2, "tau2": tau2}


def fit_predict_merf(Xtr, ytr, wtr, gtr, Xte, best):
    # Test schools are new: random effect is zero, prediction = fixed-effects ensemble f(X).
    f, vc = fit_merf(Xtr, ytr, wtr, gtr, best)
    return f.predict(xgb.DMatrix(Xte)), f
