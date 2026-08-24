"""Angle: shape-vs-category. Part 2: which PHASE does shape hurt (grasp vs transport/place)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shape_00_common import load, zscore, logit_fit_runFE, within_run_auc

SCRATCH = os.path.dirname(os.path.abspath(__file__))
df = load().dropna(subset=["obj_height", "obj_width"]).reset_index(drop=True)

# Outcomes:
# grasp stage: y_grasp = grasped (success or failed later) vs fail_no_grasp  [all episodes]
# transport/place stage: y_tp = success vs failed-after-grasp  [episodes with grasped==1]
df["y_grasp"] = df["grasped"]
sub_tp = df[df["grasped"] == 1].copy()
sub_tp["y_tp"] = sub_tp["success"]
print("grasp-stage n =", len(df), " grasp rate =", df.y_grasp.mean().round(3))
print("transport/place-stage n =", len(sub_tp), " cond. SR =", sub_tp.y_tp.mean().round(3))

def fe_auc(d, ycol, cols):
    d = d.copy()
    ridx = d["run"].map({r: i for i, r in enumerate(sorted(d["run"].unique()))}).values
    X = np.column_stack([zscore(d[c].values) for c in cols])
    y = d[ycol].values.astype(float)
    w = logit_fit_runFE(X, y, ridx)
    slope = w[ridx.max() + 1:]
    d["_score"] = X @ slope
    a, per = within_run_auc(d, "_score", ycol)
    return slope, a, per

def perm_p(d, ycol, cols, obs_auc, nperm=300, seed=2):
    rng = np.random.default_rng(seed)
    runs = d["run"].values
    X = np.column_stack([zscore(d[c].values) for c in cols])
    ridx = d["run"].map({r: i for i, r in enumerate(sorted(d["run"].unique()))}).values
    null = []
    for _ in range(nperm):
        yp = d[ycol].values.copy()
        for r in np.unique(runs):
            m = runs == r
            yp[m] = rng.permutation(yp[m])
        w = logit_fit_runFE(X, yp.astype(float), ridx)
        d2 = d.copy(); d2["_s"] = X @ w[ridx.max() + 1:]; d2["_yp"] = yp
        a, _ = within_run_auc(d2, "_s", "_yp")
        null.append(a)
    null = np.array(null)
    return (np.sum(null >= obs_auc) + 1) / (nperm + 1), null.mean(), null.std()

results = {}
for label, d, ycol in [("GRASP (grasped vs no_grasp)", df, "y_grasp"),
                       ("TRANSPORT/PLACE (success | grasped)", sub_tp, "y_tp")]:
    for cols in [["obj_height"], ["obj_width"], ["aspect"], ["obj_height", "obj_width", "aspect"]]:
        slope, a, per = fe_auc(d, ycol, cols)
        key = (label, tuple(cols))
        results[key] = (slope, a, len(per))
        print(f"{label:38s} {str(cols):42s} slopes={np.round(slope,3)} AUC={a:.3f} (n_runs={len(per)})")

# permutation p for height in each phase
for label, d, ycol in [("GRASP", df, "y_grasp"), ("TP", sub_tp, "y_tp")]:
    _, a, _ = fe_auc(d, ycol, ["obj_height"])
    p, nm, ns = perm_p(d, ycol, ["obj_height"], a)
    print(f"height-only {label}: AUC={a:.3f}, perm p={p:.4f} (null {nm:.3f}+-{ns:.3f})")

# run-bootstrap CI for height AUC per phase
def run_boot_ci(per, nboot=4000, seed=3):
    rng = np.random.default_rng(seed)
    arr = np.array([(x[1], x[2]) for x in per])
    boots = []
    for _ in range(nboot):
        idx = rng.integers(0, len(arr), len(arr))
        boots.append(np.sum(arr[idx, 0] * arr[idx, 1]) / arr[idx, 1].sum())
    return np.percentile(boots, [2.5, 97.5])

for label, d, ycol in [("GRASP", df, "y_grasp"), ("TP", sub_tp, "y_tp")]:
    slope, a, per = fe_auc(d, ycol, ["obj_height"])
    lo, hi = run_boot_ci(per)
    print(f"height AUC {label}: {a:.3f} 95% run-boot CI [{lo:.3f},{hi:.3f}]")

# ---------- binned per-phase curves (height & width), run-demeaned ----------
def binned(d, xcol, ycol, nbins=8):
    x = d[xcol].values; y = d[ycol].values.astype(float)
    ydm = y - d.groupby("run")[ycol].transform("mean").values
    qs = np.quantile(x, np.linspace(0, 1, nbins + 1)); qs[0] -= 1e-9; qs[-1] += 1e-9
    b = np.digitize(x, qs) - 1
    rows = []
    for k in range(nbins):
        m = b == k
        if m.sum() < 10: continue
        sr = y[m].mean(); n = m.sum()
        rows.append(dict(mid=x[m].mean(), n=n, sr=sr, se=np.sqrt(sr * (1 - sr) / n),
                         sr_dm=ydm[m].mean(), se_dm=ydm[m].std() / np.sqrt(n)))
    return pd.DataFrame(rows)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, (xcol, xlab) in enumerate([("obj_height", "object height (m)"), ("obj_width", "object width (m)")]):
    for j, (d, ycol, t) in enumerate([(df, "y_grasp", "P(grasp)"), (sub_tp, "y_tp", "P(success | grasped)")]):
        c = binned(d, xcol, ycol)
        ax = axes[j, i]
        ax.errorbar(c["mid"], c["sr"], yerr=1.96 * c["se"], fmt="o-", capsize=3, color=f"C{j}")
        ax.set_xlabel(xlab); ax.set_ylabel(t)
        ax.set_title(f"{t} vs {xlab}")
        ax.grid(alpha=0.3)
plt.suptitle("Phase-resolved shape effects (pooled rates, 95% binomial CI)")
plt.tight_layout()
plt.savefig(os.path.join(SCRATCH, "shape_sr_vs_bins_perphase.png"), dpi=130)
print("saved shape_sr_vs_bins_perphase.png")

# print binned tables for reporting
for xcol in ["obj_height", "obj_width"]:
    print(f"\nGRASP-stage binned {xcol}:\n", binned(df, xcol, "y_grasp").round(3).to_string(index=False))
    print(f"\nTP-stage binned {xcol}:\n", binned(sub_tp, xcol, "y_tp").round(3).to_string(index=False))
