"""Part 3: do height/rim/style predict DEPTH of failure (progress), run-controlled.
Depth measures:
  - ordinal depth D over all failures (0 never_touched .. 5 reached_sink)
  - continuous max_lift among fail_no_grasp (avoids definitional coupling)
Caveat: conditioning on failure => selection effects possible; noted.
Also: config-level split-half reliability of depth across runs (repeated configs).
"""
import pandas as pd, numpy as np

SP = "/data/xinyua11/tmp/factor_analysis_scratch"
rng = np.random.default_rng(0)
df = pd.read_csv(f"{SP}/pooled_episodes.csv")

# ordinal depth among failures
f = df[df.success == 0].copy()
def depth(r):
    if r.failure_phase == "fail_reached_sink_no_place": return 5
    if r.failure_phase == "fail_grasped_no_transport":
        return 4 if r.min_sink_dist <= 0.3 else 3
    if r.max_lift > 0.02: return 2
    if r.max_lift > 0.005: return 1
    return 0
f["depth"] = f.apply(depth, axis=1)
print("depth counts:", f.depth.value_counts().sort_index().to_dict())

# run-demean helper
def demean(sub, col):
    return sub[col] - sub.groupby("run")[col].transform("mean")

f["d_depth"] = demean(f, "depth")
fng = f[f.failure_phase == "fail_no_grasp"].copy()
fng["d_lift"] = demean(fng, "max_lift")

def strat_perm_diff(sub, mask, ycol, n=5000, seed=1):
    """mean(y[mask]) - mean(y[~mask]), permutation of mask within run."""
    rng = np.random.default_rng(seed)
    obs = sub.loc[mask, ycol].mean() - sub.loc[~mask, ycol].mean()
    runs = sub.run.values; y = sub[ycol].values; m = mask.values.copy()
    null = np.empty(n)
    order = np.argsort(runs, kind="stable")
    runs_s, y_s, m_s = runs[order], y[order], m[order]
    # run boundaries
    bounds = np.flatnonzero(np.r_[True, runs_s[1:] != runs_s[:-1], True])
    for i in range(n):
        mp = m_s.copy()
        for a, b in zip(bounds[:-1], bounds[1:]):
            rng.shuffle(mp[a:b])
        null[i] = y_s[mp].mean() - y_s[~mp].mean()
    p = (np.abs(null) >= abs(obs)).mean()
    return obs, p, null.std()

# ---------- HEIGHT ----------
print("\n===== HEIGHT =====")
for name, sub, ycol in [("ordinal depth (all fails)", f, "d_depth"),
                        ("max_lift (fail_no_grasp only)", fng, "d_lift")]:
    peak = (sub.obj_height >= 0.04) & (sub.obj_height <= 0.08)
    tall = sub.obj_height > 0.21
    print(f"\n{name}: n={len(sub)}")
    print("  band means (raw):")
    bands = pd.cut(sub.obj_height, [0, 0.04, 0.08, 0.14, 0.21, 1.0])
    print(sub.groupby(bands, observed=True)[ycol].agg(["mean", "count"]).round(4).to_string())
    obs, p, sd = strat_perm_diff(sub, peak, ycol)
    print(f"  peak(4-8cm) vs rest: diff={obs:+.4f}  perm p={p:.4f}")
    obs, p, sd = strat_perm_diff(sub, tall, ycol)
    print(f"  tall(>21cm) vs rest: diff={obs:+.4f}  perm p={p:.4f}")

# ---------- RIM ----------
print("\n===== RIM (max(|x_rel|,|y_rel|)>0.65) =====")
for name, sub, ycol in [("ordinal depth (all fails)", f, "d_depth"),
                        ("max_lift (fail_no_grasp only)", fng, "d_lift")]:
    rim = np.maximum(sub.obj_x_rel.abs(), sub.obj_y_rel.abs()) > 0.65
    obs, p, sd = strat_perm_diff(sub, rim, ycol)
    print(f"{name}: rim n={rim.sum()}/{len(sub)}  diff={obs:+.4f}  perm p={p:.4f}")

# ---------- STYLE ----------
# style effect on binary success (run-demeaned), then does it predict failure depth?
# NOTE: style success effect is computed from ALL episodes; depth is a different
# variable measured only on failures, so no mechanical overlap, but conditioning
# on failure can induce selection (collider) effects - report as such.
print("\n===== STYLE =====")
df["d_succ"] = demean(df, "success")
style_eff = df.groupby("style_id")["d_succ"].mean()  # per-style success effect

def style_depth_corr(sty_arr, d_arr, eff, min_n=20):
    t = pd.DataFrame({"s": sty_arr, "d": d_arr}).groupby("s")["d"].agg(["mean", "count"])
    t = t[t["count"] >= min_n]
    e = eff.reindex(t.index)
    ok = e.notna()
    return np.corrcoef(e[ok], t["mean"][ok])[0, 1], ok.sum()

for name, sub, ycol in [("ordinal depth (all fails)", f, "d_depth"),
                        ("max_lift (fail_no_grasp only)", fng, "d_lift")]:
    sty = sub.style_id.values.astype(int)
    dd = sub[ycol].values
    runs = sub.run.values
    r_obs, nsty = style_depth_corr(sty, dd, style_eff)
    order = np.argsort(runs, kind="stable")
    runs_s, sty_s, dd_s = runs[order], sty[order], dd[order]
    bounds = np.flatnonzero(np.r_[True, runs_s[1:] != runs_s[:-1], True])
    nperm = 1000
    null_r = np.empty(nperm)
    for i in range(nperm):
        sp = sty_s.copy()
        for a, b in zip(bounds[:-1], bounds[1:]):
            rng.shuffle(sp[a:b])
        null_r[i], _ = style_depth_corr(sp, dd_s, style_eff)
    p = (np.abs(null_r) >= abs(r_obs)).mean()
    print(f"{name}: corr(style succ-effect, style mean depth) over {nsty} styles = {r_obs:.3f}, perm p={p:.4f} (null sd {null_r.std():.3f})")

# ---------- CONFIG-LEVEL REPLICABILITY OF DEPTH ----------
# same eval config repeated across runs: if BOTH episodes fail, is depth correlated?
print("\n===== depth replicability across runs (repeated configs, both failed) =====")
key = ["object_category", "layout_id", "style_id"]
df["cfg"] = (df.object_category.astype(str) + "|" + df.layout_id.astype(str) + "|" +
             df.style_id.astype(str) + "|" + df.obj_x_rel.round(3).astype(str) + "|" +
             df.obj_y_rel.round(3).astype(str))
fails2 = df[df.success == 0].copy()
fails2["depth"] = f["depth"].reindex(fails2.index)
fails2["d_depth"] = f["d_depth"].reindex(fails2.index)
pairs = []
for cfg, grp in fails2.groupby("cfg"):
    if grp.run.nunique() >= 2:
        g = grp.drop_duplicates("run")
        vals = g.d_depth.values.copy()
        rng.shuffle(vals)
        if len(vals) >= 2:
            pairs.append((vals[0], vals[1]))
pairs = np.array(pairs)
if len(pairs) > 10:
    r = np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1]
    # bootstrap CI
    bs = [np.corrcoef(*pairs[rng.integers(0, len(pairs), len(pairs))].T)[0, 1] for _ in range(1000)]
    print(f"n config-pairs (failed in >=2 runs): {len(pairs)}; split-pair corr of demeaned depth r={r:.3f} [CI {np.percentile(bs,2.5):.3f},{np.percentile(bs,97.5):.3f}]")
else:
    print("too few repeated-failure configs:", len(pairs))
