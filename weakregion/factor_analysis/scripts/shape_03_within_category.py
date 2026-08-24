"""Angle: shape-vs-category. Part 3: does WITHIN-category shape variation predict failure,
or is shape only a between-category proxy?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from shape_00_common import load, zscore, logit_fit_runFE, within_run_auc

SCRATCH = os.path.dirname(os.path.abspath(__file__))
df = load().dropna(subset=["obj_height", "obj_width"]).reset_index(drop=True)

# how much height variance is within vs between category?
cat_mean = df.groupby("object_category")["obj_height"].transform("mean")
df["h_between"] = cat_mean
df["h_within"] = df["obj_height"] - cat_mean
tot_var = df["obj_height"].var()
print(f"height variance decomposition: between-cat {df['h_between'].var()/tot_var:.3f}, "
      f"within-cat {df['h_within'].var()/tot_var:.3f} (total sd {df['obj_height'].std():.3f} m, "
      f"within sd {df['h_within'].std():.3f} m)")
wsd = df.groupby("object_category")["obj_height"].std()
print("per-category height sd: median", round(wsd.median(), 4), "IQR",
      np.round(wsd.quantile([.25, .75]).values, 4))

w_cat_mean = df.groupby("object_category")["obj_width"].transform("mean")
df["w_between"] = w_cat_mean
df["w_within"] = df["obj_width"] - w_cat_mean
print(f"width variance decomposition: between {df['w_between'].var()/df['obj_width'].var():.3f}, "
      f"within {df['w_within'].var()/df['obj_width'].var():.3f}")

# ---------- FE logistic: run FE + category FE + within-height ----------
runs = sorted(df["run"].unique())
run_idx = df["run"].map({r: i for i, r in enumerate(runs)}).values
cats = sorted(df["object_category"].unique())
cat_idx = df["object_category"].map({c: i for i, c in enumerate(cats)}).values
y = df["success"].values.astype(float)
ygrasp = df["grasped"].values.astype(float)

from scipy.special import expit
def fe2_logit(X, y, run_idx, cat_idx, l2_fe=1e-2, l2_x=1e-4, iters=100):
    """Logistic with run FE + category FE (ridge-penalized) + features X."""
    n, d = X.shape
    R = run_idx.max() + 1; C = cat_idx.max() + 1
    DR = np.zeros((n, R)); DR[np.arange(n), run_idx] = 1
    DC = np.zeros((n, C)); DC[np.arange(n), cat_idx] = 1
    DC = DC[:, 1:]  # drop one category to avoid collinearity with run intercepts
    Z = np.column_stack([DR, DC, X])
    lam = np.r_[np.full(R, 1e-6), np.full(C - 1, l2_fe), np.full(d, l2_x)]
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = expit(Z @ w)
        g = Z.T @ (y - p) - lam * w
        Wd = np.clip(p * (1 - p), 1e-6, None)
        H = (Z * Wd[:, None]).T @ Z + np.diag(lam)
        step = np.linalg.solve(H, g)
        w += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w, R, C - 1

def slope_and_perm(ycol_vals, Xcols, label, nperm=400, seed=5):
    X = np.column_stack([zscore(df[c].values) for c in Xcols])
    w, R, Cm = fe2_logit(X, ycol_vals, run_idx, cat_idx)
    slopes = w[R + Cm:]
    # permutation: shuffle y within run x category cells (preserves both margins)
    rng = np.random.default_rng(seed)
    cell = df["run"].astype(str) + "|" + df["object_category"].astype(str)
    cell_codes = pd.factorize(cell)[0]
    order = np.argsort(cell_codes, kind="mergesort")
    null = np.zeros((nperm, len(Xcols)))
    for i in range(nperm):
        yp = ycol_vals.copy()
        # shuffle within each cell
        for cc in np.unique(cell_codes):
            m = cell_codes == cc
            if m.sum() > 1:
                yp[m] = rng.permutation(yp[m])
        wp, _, _ = fe2_logit(X, yp, run_idx, cat_idx)
        null[i] = wp[R + Cm:]
    ps = [(np.sum(np.abs(null[:, j]) >= abs(slopes[j])) + 1) / (nperm + 1) for j in range(len(Xcols))]
    print(f"{label}: slopes(z)={np.round(slopes,3)} cell-perm p={np.round(ps,4)} "
          f"(null sd {np.round(null.std(0),3)})")
    return slopes, ps

print("\n--- run FE + category FE logistic (within-run, within-category effect) ---")
slope_and_perm(y.copy(), ["obj_height"], "SUCCESS ~ height | run,cat")
slope_and_perm(ygrasp.copy(), ["obj_height"], "GRASP ~ height | run,cat")
slope_and_perm(y.copy(), ["obj_height", "obj_width"], "SUCCESS ~ h,w | run,cat")

# ---------- decomposed within/between scores: within-run AUC ----------
print("\n--- within vs between component AUC (run FE logistic, no category FE) ---")
for colset, label in [(["h_within"], "h_within only"),
                      (["h_between"], "h_between (cat mean) only"),
                      (["h_within", "h_between"], "both")]:
    X = np.column_stack([zscore(df[c].values) for c in colset])
    w = logit_fit_runFE(X, y, run_idx)
    sl = w[run_idx.max() + 1:]
    d2 = df.copy(); d2["_s"] = X @ sl
    a, per = within_run_auc(d2, "_s", "success")
    print(f"{label:28s} slopes={np.round(sl,3)} within-run AUC={a:.3f}")

# same for grasp outcome
for colset, label in [(["h_within"], "GRASP h_within"), (["h_between"], "GRASP h_between")]:
    X = np.column_stack([zscore(df[c].values) for c in colset])
    w = logit_fit_runFE(X, ygrasp, run_idx)
    sl = w[run_idx.max() + 1:]
    d2 = df.copy(); d2["_s"] = X @ sl; d2["_yg"] = ygrasp.astype(int)
    a, per = within_run_auc(d2, "_s", "_yg")
    print(f"{label:28s} slopes={np.round(sl,3)} within-run AUC={a:.3f}")

# ---------- nonparametric within-category check: split each category at its median height ----------
print("\n--- per-category median-split (tall half vs short half within category) ---")
df["succ_dm"] = df["success"] - df.groupby("run")["success"].transform("mean")
df["grasp_dm"] = df["grasped"] - df.groupby("run")["grasped"].transform("mean")
rows = []
for c, g in df.groupby("object_category"):
    med = g["obj_height"].median()
    hi = g[g["obj_height"] > med]; lo = g[g["obj_height"] <= med]
    if len(hi) < 8 or len(lo) < 8 or g["obj_height"].std() < 1e-4:
        continue
    rows.append(dict(cat=c, n=len(g), d_succ=hi["succ_dm"].mean() - lo["succ_dm"].mean(),
                     d_grasp=hi["grasp_dm"].mean() - lo["grasp_dm"].mean(),
                     h_lo=lo["obj_height"].mean(), h_hi=hi["obj_height"].mean()))
res = pd.DataFrame(rows)
print(f"categories with usable split: {len(res)}")
print(f"mean tall-minus-short d_succ (run-demeaned): {res['d_succ'].mean():+.4f} "
      f"(weighted {np.average(res['d_succ'], weights=res['n']):+.4f})")
print(f"mean tall-minus-short d_grasp: {res['d_grasp'].mean():+.4f} "
      f"(weighted {np.average(res['d_grasp'], weights=res['n']):+.4f})")
print(f"frac categories with negative d_succ: {(res['d_succ']<0).mean():.3f}")
# sign test via binomial
from scipy.stats import binomtest, wilcoxon
neg = int((res["d_succ"] < 0).sum()); tot = int((res["d_succ"] != 0).sum())
print("sign test (d_succ<0):", binomtest(neg, tot).pvalue.__round__(4))
print("wilcoxon d_succ:", wilcoxon(res["d_succ"]).pvalue.__round__(4),
      " d_grasp:", wilcoxon(res["d_grasp"]).pvalue.__round__(4))
print("mean within-cat height gap between halves:", (res["h_hi"] - res["h_lo"]).mean().round(4), "m")
res.to_csv(os.path.join(SCRATCH, "shape_within_cat_mediansplit.csv"), index=False)
