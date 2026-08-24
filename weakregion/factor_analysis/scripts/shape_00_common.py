"""Common loading + helpers for shape-vs-category analysis."""
import numpy as np
import pandas as pd
from scipy.special import expit

CSV = "/data/xinyua11/tmp/factor_analysis_scratch/pooled_episodes.csv"

def load():
    df = pd.read_csv(CSV)
    df["success"] = df["success"].astype(int)
    for c in ["obj_height", "obj_width", "max_lift", "min_sink_dist"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["aspect"] = df["obj_height"] / df["obj_width"]
    df["log_aspect"] = np.log(df["aspect"])
    # grasp outcome: grasped = success OR failed after grasping
    df["grasped"] = (df["failure_phase"] != "fail_no_grasp").astype(int)
    return df

def zscore(x):
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)

def logit_fit(X, y, l2=1e-4, iters=200):
    """Simple IRLS/Newton logistic regression with small L2. X without intercept col."""
    n, d = X.shape
    Xb = np.column_stack([np.ones(n), X])
    w = np.zeros(d + 1)
    for _ in range(iters):
        p = expit(Xb @ w)
        g = Xb.T @ (y - p) - l2 * np.r_[0, w[1:]]
        Wdiag = np.clip(p * (1 - p), 1e-6, None)
        H = (Xb * Wdiag[:, None]).T @ Xb + l2 * np.eye(d + 1)
        step = np.linalg.solve(H, g)
        w += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w

def logit_fit_runFE(X, y, run_idx, l2=1e-4, iters=100):
    """Logistic regression with run fixed effects (one intercept per run)."""
    n, d = X.shape
    R = run_idx.max() + 1
    D = np.zeros((n, R))
    D[np.arange(n), run_idx] = 1.0
    Xb = np.column_stack([D, X])
    w = np.zeros(R + d)
    lam = np.r_[np.full(R, 1e-6), np.full(d, l2)]
    for _ in range(iters):
        p = expit(Xb @ w)
        g = Xb.T @ (y - p) - lam * w
        Wdiag = np.clip(p * (1 - p), 1e-6, None)
        H = (Xb * Wdiag[:, None]).T @ Xb + np.diag(lam)
        step = np.linalg.solve(H, g)
        w += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w  # first R entries run intercepts, last d slopes

def auc(scores, y):
    """Mann-Whitney AUC."""
    y = np.asarray(y)
    s = np.asarray(scores, float)
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    order = np.argsort(np.concatenate([neg, pos]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    # tie-corrected ranks
    allv = np.concatenate([neg, pos])
    sortv = np.sort(allv)
    # use scipy rankdata for ties
    from scipy.stats import rankdata
    r = rankdata(allv)
    rpos = r[len(neg):]
    return (rpos.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

def within_run_auc(df, score_col, y_col="success", min_per_class=3):
    """AUC computed per run then averaged weighted by n; returns (mean_auc, per-run list)."""
    rows = []
    for run, g in df.groupby("run"):
        y = g[y_col].values
        if y.sum() < min_per_class or (len(y) - y.sum()) < min_per_class:
            continue
        a = auc(g[score_col].values, y)
        rows.append((run, a, len(y)))
    if not rows:
        return np.nan, []
    aucs = np.array([r[1] for r in rows]); ns = np.array([r[2] for r in rows])
    return float(np.sum(aucs * ns) / ns.sum()), rows

def perm_p_within_run(df, stat_fn, nperm=2000, seed=0):
    """Permutation p-value: shuffle y within run. stat_fn(df_with_y) -> scalar."""
    rng = np.random.default_rng(seed)
    obs = stat_fn(df)
    null = np.empty(nperm)
    ycol = df["_permy_base"].values.copy()
    runs = df["run"].values
    order = np.argsort(runs, kind="mergesort")
    for i in range(nperm):
        yp = ycol.copy()
        d2 = df.copy()
        for run, g in df.groupby("run", sort=False):
            idx = g.index.values
            yp_local = ycol[df.index.get_indexer(idx)]
            rng.shuffle(yp_local)
            yp[df.index.get_indexer(idx)] = yp_local
        d2["_permy"] = yp
        null[i] = stat_fn(d2, permuted=True)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (nperm + 1)
    return obs, p, null
