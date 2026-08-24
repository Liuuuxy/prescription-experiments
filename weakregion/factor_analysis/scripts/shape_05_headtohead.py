"""Angle: shape-vs-category. Part 5: head-to-head within-run AUC, shape-only vs category-only,
train on half the runs / evaluate on the held-out half (guards against overfitting 81 categories)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shape_00_common import load, logit_fit_runFE, within_run_auc

SCRATCH = os.path.dirname(os.path.abspath(__file__))
df = load().dropna(subset=["obj_height", "obj_width"]).reset_index(drop=True)
df["succ_dm"] = df["success"] - df.groupby("run")["success"].transform("mean")
runs = np.array(sorted(df["run"].unique()))

def zfit(x):  # returns (transform fn) fitted on train
    mu, sd = np.nanmean(x), np.nanstd(x) + 1e-12
    return lambda v: (v - mu) / sd

def shape_features(d, ztrans=None, cols_cache={}):
    h = d["obj_height"].values; wd = d["obj_width"].values
    feats = {"h": h, "h2": h ** 2, "w": wd, "la": np.log(h / wd)}
    return feats

def build_shape_X(d, trans):
    f = shape_features(d)
    return np.column_stack([trans[k](f[k]) for k in ["h", "h2", "w", "la"]])

def cat_score_table(dtr, m=10.0):
    """Empirical-Bayes shrunken per-category run-demeaned mean success."""
    g = dtr.groupby("object_category")["succ_dm"]
    tab = g.agg(["sum", "count"])
    tab["score"] = tab["sum"] / (tab["count"] + m)
    return tab["score"].to_dict()

rng = np.random.default_rng(21)
nsplits = 120
rows = []
for s in range(nsplits):
    perm = rng.permutation(runs)
    tr, te = set(perm[:14]), set(perm[14:])
    dtr = df[df["run"].isin(tr)].copy()
    dte = df[df["run"].isin(te)].copy()

    # ---- shape model (nonlinear: h, h^2, w, log-aspect), run FE logistic on train
    ftr = shape_features(dtr)
    trans = {k: zfit(v) for k, v in ftr.items()}
    Xtr = build_shape_X(dtr, trans)
    ridx = dtr["run"].map({r: i for i, r in enumerate(sorted(tr))}).values
    w = logit_fit_runFE(Xtr, dtr["success"].values.astype(float), ridx)
    slope = w[ridx.max() + 1:]
    Xte = build_shape_X(dte, trans)
    dte["_shape"] = Xte @ slope

    # linear-height-only variant
    w1 = logit_fit_runFE(trans["h"](ftr["h"]).reshape(-1, 1),
                         dtr["success"].values.astype(float), ridx)
    dte["_h"] = trans["h"](dte["obj_height"].values) * w1[-1]

    # ---- category model
    for m in [0.0, 10.0, 30.0]:
        tab = cat_score_table(dtr, m=m)
        dte[f"_cat{int(m)}"] = dte["object_category"].map(tab).fillna(0.0)

    # ---- combined: shape score + cat score (m=10), refit tiny logistic on train? avoid; just sum z-scores
    tab10 = cat_score_table(dtr, m=10.0)
    dtr["_catS"] = dtr["object_category"].map(tab10).fillna(0.0)
    dtr["_shapeS"] = Xtr @ slope
    zc, zs = zfit(dtr["_catS"].values), zfit(dtr["_shapeS"].values)
    # fit 2-feature logistic on train for proper weighting
    Xc = np.column_stack([zc(dtr["_catS"].values), zs(dtr["_shapeS"].values)])
    wc = logit_fit_runFE(Xc, dtr["success"].values.astype(float), ridx)
    sl2 = wc[ridx.max() + 1:]
    dte["_comb"] = np.column_stack([zc(dte[f"_cat10"].values), zs(dte["_shape"].values)]) @ sl2

    r = {"split": s}
    for col, name in [("_shape", "shape_nl"), ("_h", "height_lin"), ("_cat0", "cat_m0"),
                      ("_cat10", "cat_m10"), ("_cat30", "cat_m30"), ("_comb", "shape+cat")]:
        a, _ = within_run_auc(dte, col, "success")
        r[name] = a
    rows.append(r)

res = pd.DataFrame(rows).set_index("split")
print("Held-out within-run AUC over", nsplits, "random half-run splits (mean +- sd):")
print(res.agg(["mean", "std"]).T.round(4).to_string())
d = res["cat_m10"] - res["shape_nl"]
print(f"\npaired diff cat_m10 - shape_nl: mean {d.mean():+.4f} sd {d.std():.4f}, "
      f"frac splits cat>shape: {(d>0).mean():.3f}")
d2 = res["shape+cat"] - res["cat_m10"]
print(f"paired diff (shape+cat) - cat_m10: mean {d2.mean():+.4f} sd {d2.std():.4f}, "
      f"frac>0: {(d2>0).mean():.3f}")
d3 = res["shape+cat"] - res["shape_nl"]
print(f"paired diff (shape+cat) - shape_nl: mean {d3.mean():+.4f} sd {d3.std():.4f}, frac>0: {(d3>0).mean():.3f}")

# how much of category signal does shape explain? category residual after shape:
# per-category mean of (succ_dm - shape-based predicted dm) — do one representative split analysis pooled:
# simpler global: correlation between per-category mean success (demeaned) and per-category mean height
cat = df.groupby("object_category").agg(sr_dm=("succ_dm", "mean"), h=("obj_height", "mean"),
                                        w=("obj_width", "mean"), n=("success", "size"))
cat["aspect"] = cat["h"] / cat["w"]
from scipy.stats import pearsonr, spearmanr
r_h, p_h = pearsonr(cat["h"], cat["sr_dm"])
rs, ps = spearmanr(cat["h"], cat["sr_dm"])
print(f"\nper-category: corr(mean height, demeaned SR) pearson r={r_h:.3f} (p={p_h:.2g}), "
      f"spearman {rs:.3f} (p={ps:.2g}), n=81 cats")
# R2 of category SR explained by shape (h, h2, w, log aspect), weighted by n
Xc = np.column_stack([cat["h"], cat["h"]**2, cat["w"], np.log(cat["aspect"])])
Xc = (Xc - Xc.mean(0)) / Xc.std(0)
Xc = np.column_stack([np.ones(len(cat)), Xc])
W = np.diag(cat["n"].values.astype(float))
beta = np.linalg.solve(Xc.T @ W @ Xc, Xc.T @ W @ cat["sr_dm"].values)
pred = Xc @ beta
ssr = np.sum(cat["n"] * (cat["sr_dm"] - pred) ** 2)
sst = np.sum(cat["n"] * (cat["sr_dm"] - np.average(cat["sr_dm"], weights=cat["n"])) ** 2)
print(f"weighted R^2 of per-category SR explained by shape(h,h2,w,logAR): {1 - ssr/sst:.3f}")

# plot
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
order = ["height_lin", "shape_nl", "cat_m0", "cat_m10", "cat_m30", "shape+cat"]
means = res[order].mean(); sds = res[order].std()
ax[0].bar(range(len(order)), means - 0.5, bottom=0.5, yerr=sds, capsize=4,
          color=["C0", "C0", "C3", "C3", "C3", "C2"], alpha=0.8)
ax[0].set_xticks(range(len(order)), order, rotation=20)
ax[0].axhline(0.5, color="k", lw=1)
ax[0].set_ylabel("held-out within-run AUC")
ax[0].set_title("Shape vs category: held-out AUC (120 half-run splits)")
ax[1].scatter(res["shape_nl"], res["cat_m10"], s=12, alpha=0.6)
lims = [min(res["shape_nl"].min(), res["cat_m10"].min()) - .005,
        max(res["shape_nl"].max(), res["cat_m10"].max()) + .005]
ax[1].plot(lims, lims, "k--", lw=1)
ax[1].set_xlabel("shape-only AUC"); ax[1].set_ylabel("category-only AUC (m=10)")
ax[1].set_title("per-split paired comparison")
plt.tight_layout()
plt.savefig(os.path.join(SCRATCH, "shape_vs_category_headtohead.png"), dpi=130)
print("saved shape_vs_category_headtohead.png")
res.to_csv(os.path.join(SCRATCH, "shape_headtohead_splits.csv"))
