"""SHAP summary statistics shared by 04_shap_jrr.py and 04b_robust.py."""
import numpy as np

from config import FEAT, FEATURES

BLOCKS = {}
for j, f in enumerate(FEAT):
    BLOCKS.setdefault(FEATURES[f][0], []).append(j)
BLK_NAMES = list(BLOCKS)


def stats(phi, X, w):
    """phi: n x F SHAP values (bias column removed)."""
    sw = w.sum()
    imp = (w[:, None] * np.abs(phi)).sum(0) / sw
    slope = np.full(phi.shape[1], np.nan)
    for j in range(phi.shape[1]):
        ok = ~np.isnan(X[:, j]) & (w > 0)
        if ok.sum() < 30:
            continue
        x, p, ww = X[ok, j], phi[ok, j], w[ok]
        mx, mp = np.average(x, weights=ww), np.average(p, weights=ww)
        vx = np.average((x - mx) ** 2, weights=ww)
        if vx > 0:
            slope[j] = np.average((x - mx) * (p - mp), weights=ww) / np.sqrt(vx)
    blk = np.array([(w * np.abs(phi[:, idx].sum(1))).sum() / sw for idx in BLOCKS.values()])
    return imp, slope, blk
