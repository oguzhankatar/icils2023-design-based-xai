"""Design-based estimation helpers for ICILS 2023 (JRR with 75 zones + plausible values).

Sampling variance (User Guide eq. 3.1): SV = sum_i (theta_i - theta_full)^2   (no 1/2 factor)
PV combination (eq. 3.3): Var = mean_p(SV_p) + (1 + 1/P) * Var_between_PV
"""
import numpy as np


def jrr_var(full, reps):
    """full: scalar or array; reps: array (n_rep, ...)."""
    reps = np.asarray(reps)
    return ((reps - np.asarray(full)) ** 2).sum(axis=0)


def combine_pv(full_by_pv, sv_by_pv):
    """full_by_pv: (P, ...) full-sample estimates; sv_by_pv: (P, ...) sampling variances.
    Returns (estimate, se, sampling_var, imputation_var)."""
    f = np.asarray(full_by_pv, dtype=float)
    sv = np.asarray(sv_by_pv, dtype=float)
    P = f.shape[0]
    est = f.mean(axis=0)
    u = sv.mean(axis=0)
    b = f.var(axis=0, ddof=1)
    tot = u + (1 + 1 / P) * b
    return est, np.sqrt(tot), u, b


def wmean(x, w):
    return np.sum(x * w) / np.sum(w)


def weighted_r2(y, yhat, w):
    ybar = wmean(y, w)
    return 1 - np.sum(w * (y - yhat) ** 2) / np.sum(w * (y - ybar) ** 2)
