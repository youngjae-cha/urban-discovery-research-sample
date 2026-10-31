"""Selected original research functions; no data or trained parameters included.
Functions from olsutil.py; function bodies are unchanged.
"""
import numpy as np
import pandas as pd
from scipy import stats

def zscore(d):
    return (d - d.mean()) / d.std(ddof=0)


def ols(df, y, xs, w=None):
    """OLS of y on xs (all z-scored within the estimation sample), HC1 SEs, t-based 95% CI.
    w: optional column of weights (WLS with sqrt-weights; HC1 on the weighted design)."""
    cols = [y] + xs + ([w] if w else [])
    d = df[cols].dropna()
    Z = zscore(d[[y] + xs])
    X = np.column_stack([np.ones(len(Z)), Z[xs].to_numpy()]); yy = Z[y].to_numpy()
    if w:
        sw = np.sqrt(d[w].to_numpy()); Xw = X * sw[:, None]; yw = yy * sw
        b = np.linalg.lstsq(Xw, yw, rcond=None)[0]; e = yy - X @ b; ew = e * sw
        XtXi = np.linalg.inv(Xw.T @ Xw); V = XtXi @ (Xw.T * (ew ** 2)) @ Xw @ XtXi
    else:
        b = np.linalg.lstsq(X, yy, rcond=None)[0]; e = yy - X @ b
        XtXi = np.linalg.inv(X.T @ X); V = XtXi @ (X.T * (e ** 2)) @ X @ XtXi
    n, k = X.shape; V *= n / (n - k)
    se = np.sqrt(np.diag(V)); t = b / se; p = 2 * stats.t.sf(np.abs(t), n - k)
    tc = stats.t.ppf(0.975, n - k); r2 = 1 - e.var() / yy.var()
    coef = {x: dict(b=float(b[i + 1]), se=float(se[i + 1]), t=float(t[i + 1]), p=float(p[i + 1]),
                    lo=float(b[i + 1] - tc * se[i + 1]), hi=float(b[i + 1] + tc * se[i + 1])) for i, x in enumerate(xs)}
    result = dict(n=int(n), k=int(k), r2=float(r2), coef=coef,
                  r2_metric="unweighted_residual_variance_reduction" if w else "ordinary_R2")
    if w:
        weights = d[w].to_numpy()
        result["r2_weighted"] = float(1 - np.sum(weights * e**2) / np.sum(weights * (yy - np.average(yy, weights=weights))**2))
    return result


def cell(df, y, x, ctrl, extra=None, w=None):
    """Coefficient of x with controls (+extra covariates); dR2 = R2(full) - R2(controls + extra)."""
    xs = [x] + (extra or []) + ctrl
    # The reduced model must use exactly the full model's complete cases.
    required = list(dict.fromkeys([y] + xs + ([w] if w else [])))
    common = df[required].dropna()
    m = ols(common, y, xs, w); m0 = ols(common, y, (extra or []) + ctrl, w)
    assert m["n"] == m0["n"]
    r = dict(m["coef"][x]); r["n"] = m["n"]; r["dR2"] = m["r2"] - m0["r2"]; r["r2_full"] = m["r2"]; r["r2_base"] = m0["r2"]
    r["n_baseline"] = m0["n"]
    r["r2_metric"] = m["r2_metric"]
    if w:
        r["r2_weighted_full"] = m["r2_weighted"]
        r["r2_weighted_base"] = m0["r2_weighted"]
        r["dR2_weighted"] = m["r2_weighted"] - m0["r2_weighted"]
    if extra:
        r["extra"] = {e: m["coef"][e] for e in extra}
    return r
