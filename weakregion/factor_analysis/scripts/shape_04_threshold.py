"""Angle: shape-vs-category. Part 4: threshold/cliff effects on height (+ per-meter slope comparability)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from shape_00_common import load, zscore, logit_fit_runFE, within_run_auc

SCRATCH = os.path.dirname(os.path.abspath(__file__))
df = load().dropna(subset=["obj_height", "obj_width"]).reset_index(drop=True)
runs = sorted(df["run"].unique())
run_idx = df["run"].map({r: i for i, r in enumerate(runs)}).values
y = df["success"].values.astype(float)

# --- per-meter slope comparability (within vs between), unstandardized ---
cm = df.groupby("object_category")["obj_height"].transform("mean")
df["h_within"] = df["obj_height"] - cm
df["h_between"] = cm
X = df[["h_within", "h_between"]].values
w = logit_fit_runFE(X, y, run_idx, l2=1e-6)
sl = w[run_idx.max() + 1:]
print(f"per-METER logit slopes (run FE): h_within {sl[0]:.2f}, h_between {sl[1]:.2f} logit/m")
# grasp outcome
yg = df["grasped"].values.astype(float)
wg = logit_fit_runFE(X, yg, run_idx, l2=1e-6)
slg = wg[run_idx.max() + 1:]
print(f"per-METER logit slopes GRASP:    h_within {slg[0]:.2f}, h_between {slg[1]:.2f} logit/m")

# bootstrap CI over runs for the within slope
rng = np.random.default_rng(7)
bw, bb = [], []
for _ in range(600):
    rsel = rng.choice(runs, len(runs), replace=True)
    parts = [df[df["run"] == r] for r in rsel]
    db = pd.concat(parts, ignore_index=True)
    ridx = np.concatenate([[i] * len(p) for i, p in enumerate(parts)]).astype(int)
    Xb = db[["h_within", "h_between"]].values
    yb = db["success"].values.astype(float)
    try:
        wb = logit_fit_runFE(Xb, yb, ridx, l2=1e-6)
        bw.append(wb[ridx.max() + 1]); bb.append(wb[ridx.max() + 2])
    except np.linalg.LinAlgError:
        pass
print(f"run-bootstrap 95% CI: h_within [{np.percentile(bw,2.5):.2f},{np.percentile(bw,97.5):.2f}], "
      f"h_between [{np.percentile(bb,2.5):.2f},{np.percentile(bb,97.5):.2f}] logit/m")

# ---------- threshold search on height ----------
df["succ_dm"] = df["success"] - df.groupby("run")["success"].transform("mean")
df["grasp_dm"] = df["grasped"] - df.groupby("run")["grasped"].transform("mean")

def scan(dcol, xcol="obj_height", lo_q=0.1, hi_q=0.9, n=60):
    ths = np.quantile(df[xcol], np.linspace(lo_q, hi_q, n))
    out = []
    for t in np.unique(ths):
        hi = df[df[xcol] > t]; lo = df[df[xcol] <= t]
        if len(hi) < 200 or len(lo) < 200:
            continue
        d = hi[dcol].mean() - lo[dcol].mean()
        # AUC of the binary indicator = balanced accuracy-ish; use effect size instead
        out.append((t, d, len(hi)))
    return pd.DataFrame(out, columns=["thr", "diff", "n_above"])

sc = scan("succ_dm")
best = sc.loc[sc["diff"].abs().idxmax()]
print(f"\nbest single height threshold (success, run-demeaned diff): thr={best.thr:.3f} m, "
      f"SR(above)-SR(below)={best['diff']:+.3f}, n_above={int(best.n_above)}")
scg = scan("grasp_dm")
bestg = scg.loc[scg["diff"].abs().idxmax()]
print(f"best threshold (grasp): thr={bestg.thr:.3f} m, diff={bestg['diff']:+.3f}, n_above={int(bestg.n_above)}")
print("\nthreshold scan (success), every 5th:")
print(sc.iloc[::5].round(3).to_string(index=False))

# honest out-of-sample: pick threshold on half the runs, evaluate on other half (50 splits)
rng = np.random.default_rng(11)
oos = []
thr_sel = []
for _ in range(50):
    perm = rng.permutation(runs)
    tr, te = set(perm[:14]), set(perm[14:])
    dtr = df[df["run"].isin(tr)]; dte = df[df["run"].isin(te)]
    ths = np.quantile(dtr["obj_height"], np.linspace(0.1, 0.9, 60))
    bestd, bestt = 0, None
    for t in np.unique(ths):
        m = dtr["obj_height"] > t
        if m.sum() < 100 or (~m).sum() < 100: continue
        d = dtr.loc[m, "succ_dm"].mean() - dtr.loc[~m, "succ_dm"].mean()
        if abs(d) > abs(bestd): bestd, bestt = d, t
    m = dte["obj_height"] > bestt
    oos.append(dte.loc[m, "succ_dm"].mean() - dte.loc[~m, "succ_dm"].mean())
    thr_sel.append(bestt)
print(f"\nOOS threshold effect (choose thr on 14 runs, eval on 14): mean diff {np.mean(oos):+.3f} "
      f"(sd {np.std(oos):.3f}), selected thr median {np.median(thr_sel):.3f} m "
      f"IQR [{np.percentile(thr_sel,25):.3f},{np.percentile(thr_sel,75):.3f}]")

# threshold AUC: binary feature AUC within-run
t = best.thr
d2 = df.copy(); d2["_s"] = -(df["obj_height"] > t).astype(float)
a, _ = within_run_auc(d2, "_s", "success")
print(f"binary height>{t:.3f} within-run AUC (success): {a:.3f}")

# cliff vs linear: compare AUC of piecewise (hinge at best thr) vs linear height
h = df["obj_height"].values
for name, Xf in [("linear h", np.column_stack([zscore(h)])),
                 ("hinge max(0,h-thr)", np.column_stack([zscore(np.maximum(0, h - t))])),
                 ("h + hinge", np.column_stack([zscore(h), zscore(np.maximum(0, h - t))])),
                 ("h + h^2", np.column_stack([zscore(h), zscore(h ** 2)]))]:
    wf = logit_fit_runFE(Xf, y, run_idx)
    slf = wf[run_idx.max() + 1:]
    d2 = df.copy(); d2["_s"] = Xf @ slf
    a, _ = within_run_auc(d2, "_s", "success")
    print(f"{name:22s} slopes={np.round(slf,3)} AUC={a:.3f}")

# SR by side of cliff, raw pooled for interpretability
m = df["obj_height"] > t
print(f"\nraw SR: height<= {t:.3f}: {df.loc[~m,'success'].mean():.3f} (n={int((~m).sum())}), "
      f"height> {t:.3f}: {df.loc[m,'success'].mean():.3f} (n={int(m.sum())})")
print(f"raw grasp rate: below {df.loc[~m,'grasped'].mean():.3f}, above {df.loc[m,'grasped'].mean():.3f}")
