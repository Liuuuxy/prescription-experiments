"""Angle: shape-vs-category. Part 1: EDA + binned SR curves + within-run logistic AUC."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shape_00_common import load, zscore, logit_fit_runFE, within_run_auc, auc
from scipy.special import expit

SCRATCH = os.path.dirname(os.path.abspath(__file__))
df = load()

print("n rows", len(df), "runs", df["run"].nunique(), "cats", df["object_category"].nunique())
print("overall SR", df["success"].mean().round(3))
print("run SR range", df.groupby("run")["success"].mean().agg(["min", "max"]).round(3).to_dict())
print("missing:", df[["obj_height", "obj_width", "max_lift", "min_sink_dist"]].isna().sum().to_dict())
print("failure_phase counts:\n", df["failure_phase"].value_counts())
print("height quantiles", df["obj_height"].quantile([0, .1, .25, .5, .75, .9, 1]).round(3).to_dict())
print("width quantiles", df["obj_width"].quantile([0, .1, .25, .5, .75, .9, 1]).round(3).to_dict())
print("aspect quantiles", df["aspect"].quantile([0, .1, .25, .5, .75, .9, 1]).round(3).to_dict())

# sanity: max_lift vs grasped definition
print("\nmax_lift by phase:\n", df.groupby("failure_phase")["max_lift"].describe().round(3))
print("frac max_lift>0.05 among fail_no_grasp:",
      (df.loc[df.failure_phase == "fail_no_grasp", "max_lift"] > 0.05).mean().round(3))
print("frac max_lift>0.05 among grasped-failures:",
      (df.loc[df.failure_phase.isin(["fail_grasped_no_transport", "fail_reached_sink_no_place"]), "max_lift"] > 0.05).mean().round(3))

# drop rows with missing shape (50/7112)
df = df.dropna(subset=["obj_height", "obj_width"]).reset_index(drop=True)
print("\nafter dropping NaN shape rows:", len(df))

# ---------- run-demeaned success for binned curves ----------
df["sr_run"] = df.groupby("run")["success"].transform("mean")
df["succ_dm"] = df["success"] - df["sr_run"]  # run-demeaned

def binned_curve(x, y, ydm, nbins=8):
    qs = np.quantile(x, np.linspace(0, 1, nbins + 1))
    qs[0] -= 1e-9; qs[-1] += 1e-9
    bin_id = np.digitize(x, qs) - 1
    out = []
    for b in range(nbins):
        m = bin_id == b
        n = m.sum()
        if n < 10: continue
        sr = y[m].mean()
        se = np.sqrt(sr * (1 - sr) / n)
        srdm = ydm[m].mean()
        sedm = ydm[m].std() / np.sqrt(n)
        out.append(dict(lo=qs[b], hi=qs[b + 1], mid=x[m].mean(), n=n, sr=sr, se=se, sr_dm=srdm, se_dm=sedm))
    return pd.DataFrame(out)

feats = {"obj_height": "height (m)", "obj_width": "width (m)", "aspect": "aspect = h/w"}
curves = {}
for f in feats:
    curves[f] = binned_curve(df[f].values, df["success"].values, df["succ_dm"].values)
    print(f"\nBinned SR: {f}\n", curves[f].round(3).to_string(index=False))

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for j, (f, lab) in enumerate(feats.items()):
    c = curves[f]
    ax = axes[0, j]
    ax.errorbar(c["mid"], c["sr"], yerr=1.96 * c["se"], fmt="o-", capsize=3)
    ax.axhline(df["success"].mean(), color="gray", ls="--", lw=1)
    ax.set_xlabel(lab); ax.set_ylabel("raw SR"); ax.set_title(f"SR vs {lab} (pooled, 95% CI)")
    ax = axes[1, j]
    ax.errorbar(c["mid"], c["sr_dm"], yerr=1.96 * c["se_dm"], fmt="s-", color="C1", capsize=3)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel(lab); ax.set_ylabel("run-demeaned SR"); ax.set_title("run-demeaned")
plt.tight_layout()
plt.savefig(os.path.join(SCRATCH, "shape_sr_vs_bins_overall.png"), dpi=130)
print("\nsaved shape_sr_vs_bins_overall.png")

# ---------- within-run logistic AUC (run fixed effects) ----------
runs = sorted(df["run"].unique())
run_idx = df["run"].map({r: i for i, r in enumerate(runs)}).values

def fe_logit_auc(cols, y=None, sub=None, label=""):
    d = df if sub is None else sub
    yv = d["success"].values if y is None else y
    ridx = d["run"].map({r: i for i, r in enumerate(sorted(d['run'].unique()))}).values
    X = np.column_stack([zscore(d[c].values) for c in cols])
    w = logit_fit_runFE(X, yv.astype(float), ridx)
    R = ridx.max() + 1
    slope = w[R:]
    score = X @ slope  # run intercepts excluded from score -> within-run ranking
    d2 = d.copy(); d2["_score"] = score; d2["_y"] = yv
    a, per = within_run_auc(d2, "_score", "_y")
    return slope, a, per

for cols in [["obj_height"], ["obj_width"], ["aspect"], ["obj_height", "obj_width"],
             ["obj_height", "obj_width", "aspect"]]:
    slope, a, per = fe_logit_auc(cols)
    per_aucs = [x[1] for x in per]
    print(f"FE-logit {cols}: slopes(z)={np.round(slope,3)} within-run AUC={a:.3f} "
          f"(per-run mean {np.mean(per_aucs):.3f}, sd {np.std(per_aucs):.3f}, n_runs={len(per)})")

# bootstrap CI on within-run AUC for full shape model (resample runs)
slope, a_full, per = fe_logit_auc(["obj_height", "obj_width", "aspect"])
per_arr = np.array([(x[1], x[2]) for x in per])
rng = np.random.default_rng(0)
boot = []
for _ in range(4000):
    idx = rng.integers(0, len(per_arr), len(per_arr))
    aa, nn = per_arr[idx, 0], per_arr[idx, 1]
    boot.append(np.sum(aa * nn) / nn.sum())
print(f"shape(h,w,aspect) within-run AUC={a_full:.3f} run-bootstrap 95% CI "
      f"[{np.percentile(boot,2.5):.3f},{np.percentile(boot,97.5):.3f}]")

# permutation p: shuffle success within run, refit, AUC
rng = np.random.default_rng(1)
nperm = 300
null = []
X = np.column_stack([zscore(df[c].values) for c in ["obj_height", "obj_width", "aspect"]])
for i in range(nperm):
    yp = df["success"].values.copy()
    for r in runs:
        m = df["run"].values == r
        yp[m] = rng.permutation(yp[m])
    w = logit_fit_runFE(X, yp.astype(float), run_idx)
    score = X @ w[run_idx.max() + 1:]
    d2 = df.copy(); d2["_score"] = score; d2["_y"] = yp
    a, _ = within_run_auc(d2, "_score", "_y")
    null.append(a)
null = np.array(null)
p = (np.sum(null >= a_full) + 1) / (nperm + 1)
print(f"perm test (within-run shuffle, {nperm} perms): null AUC mean {null.mean():.3f} sd {null.std():.3f}, p={p:.4f}")
