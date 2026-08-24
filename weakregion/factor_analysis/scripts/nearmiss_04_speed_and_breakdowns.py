"""Part 4: steps_to_success as difficulty measure among successes (run-controlled).
Plus interpretable depth breakdowns: P(never_touched) by rim/height among fail_no_grasp.
"""
import pandas as pd, numpy as np

SP = "/data/xinyua11/tmp/factor_analysis_scratch"
rng = np.random.default_rng(0)
df = pd.read_csv(f"{SP}/pooled_episodes.csv")

def demean(sub, col):
    return sub[col] - sub.groupby("run")[col].transform("mean")

def strat_perm_diff(sub, mask, ycol, n=5000, seed=1):
    rng = np.random.default_rng(seed)
    obs = sub.loc[mask, ycol].mean() - sub.loc[~mask, ycol].mean()
    runs = sub.run.values; y = sub[ycol].values; m = np.asarray(mask).copy()
    order = np.argsort(runs, kind="stable")
    runs_s, y_s, m_s = runs[order], y[order], m[order]
    bounds = np.flatnonzero(np.r_[True, runs_s[1:] != runs_s[:-1], True])
    null = np.empty(n)
    for i in range(n):
        mp = m_s.copy()
        for a, b in zip(bounds[:-1], bounds[1:]):
            rng.shuffle(mp[a:b])
        null[i] = y_s[mp].mean() - y_s[~mp].mean()
    return obs, (np.abs(null) >= abs(obs)).mean()

# ---- interpretable depth breakdowns among fail_no_grasp ----
fng = df[(df.success == 0) & (df.failure_phase == "fail_no_grasp")].copy()
fng["never"] = (fng.max_lift <= 0.005).astype(float)
fng["d_never"] = demean(fng, "never")
rim = np.maximum(fng.obj_x_rel.abs(), fng.obj_y_rel.abs()) > 0.65
print("=== P(never_touched | fail_no_grasp) ===")
print(f"rim:    {fng.loc[rim,'never'].mean():.3f} (n={rim.sum()})  center: {fng.loc[~rim,'never'].mean():.3f} (n={(~rim).sum()})")
obs, p = strat_perm_diff(fng, rim, "d_never")
print(f"rim-center diff (demeaned): {obs:+.4f}, perm p={p:.4f}")
tall = fng.obj_height > 0.21
obs, p = strat_perm_diff(fng, tall, "d_never")
print(f"tall(>21cm) vs rest, P(never): {fng.loc[tall,'never'].mean():.3f} vs {fng.loc[~tall,'never'].mean():.3f}, diff={obs:+.4f}, p={p:.4f}")
peak = (fng.obj_height >= 0.04) & (fng.obj_height <= 0.08)
obs, p = strat_perm_diff(fng, peak, "d_never")
print(f"peak(4-8cm) vs rest, P(never): {fng.loc[peak,'never'].mean():.3f} vs {fng.loc[~peak,'never'].mean():.3f}, diff={obs:+.4f}, p={p:.4f}")

# ---- Part 4: steps among successes ----
s = df[(df.success == 1) & df.steps_to_success.notna()].copy()
s["steps"] = s.steps_to_success.astype(float)
s["logsteps"] = np.log(s.steps)
s["d_logsteps"] = demean(s, "logsteps")
s["d_steps"] = demean(s, "steps")
print(f"\n=== steps_to_success among successes (n={len(s)}) ===")

rim_s = np.maximum(s.obj_x_rel.abs(), s.obj_y_rel.abs()) > 0.65
tall_s = s.obj_height > 0.21
peak_s = (s.obj_height >= 0.04) & (s.obj_height <= 0.08)
for name, mask in [("rim", rim_s), ("tall>21cm", tall_s), ("peak 4-8cm", peak_s)]:
    obs, p = strat_perm_diff(s, mask, "d_steps")
    obs_l, p_l = strat_perm_diff(s, mask, "d_logsteps")
    print(f"{name}: n={mask.sum()}; steps diff={obs:+.1f} (p={p:.4f}); logsteps diff={obs_l:+.4f} (p={p_l:.4f}); raw means {s.loc[mask,'steps'].mean():.0f} vs {s.loc[~mask,'steps'].mean():.0f}")

# height bands raw
bands = pd.cut(s.obj_height, [0, 0.04, 0.08, 0.14, 0.21, 1.0])
print("\nmean demeaned steps by height band:")
print(s.groupby(bands, observed=True)["d_steps"].agg(["mean", "count"]).round(2).to_string())

# style: corr(style success effect, style mean demeaned steps)
df["d_succ"] = demean(df, "success")
style_eff = df.groupby("style_id")["d_succ"].mean()
t = s.groupby("style_id")["d_steps"].agg(["mean", "count"])
t = t[t["count"] >= 20]
e = style_eff.reindex(t.index)
r_obs = np.corrcoef(e, t["mean"])[0, 1]
# permutation within run
sty = s.style_id.values.astype(int); dd = s.d_steps.values; runs = s.run.values
order = np.argsort(runs, kind="stable")
sty_s, dd_s, runs_s = sty[order], dd[order], runs[order]
bounds = np.flatnonzero(np.r_[True, runs_s[1:] != runs_s[:-1], True])
null_r = np.empty(2000)
for i in range(2000):
    sp = sty_s.copy()
    for a, b in zip(bounds[:-1], bounds[1:]):
        rng.shuffle(sp[a:b])
    tt = pd.DataFrame({"s": sp, "d": dd_s}).groupby("s")["d"].agg(["mean", "count"])
    tt = tt[tt["count"] >= 20]
    ee = style_eff.reindex(tt.index)
    null_r[i] = np.corrcoef(ee, tt["mean"])[0, 1]
p = (np.abs(null_r) >= abs(r_obs)).mean()
print(f"\nstyle: corr(succ-effect, mean steps) over {len(t)} styles = {r_obs:.3f}, perm p={p:.4f} (null sd {null_r.std():.3f})")
print("(negative corr = hard styles -> slower successes, i.e. difficulty continuum)")

# category-level: corr between category success effect and category mean steps
cat_eff = df.groupby("object_category")["d_succ"].mean()
tc = s.groupby("object_category")["d_steps"].agg(["mean", "count"])
tc = tc[tc["count"] >= 15]
ec = cat_eff.reindex(tc.index)
r_cat = np.corrcoef(ec, tc["mean"])[0, 1]
cat = s.object_category.values; runs = s.run.values
order = np.argsort(runs, kind="stable")
cat_s = cat[order]
null_rc = np.empty(2000)
for i in range(2000):
    cp = cat_s.copy()
    for a, b in zip(bounds[:-1], bounds[1:]):
        rng.shuffle(cp[a:b])
    tt = pd.DataFrame({"c": cp, "d": dd_s}).groupby("c")["d"].agg(["mean", "count"])
    tt = tt[tt["count"] >= 15]
    ee = cat_eff.reindex(tt.index)
    null_rc[i] = np.corrcoef(ee, tt["mean"])[0, 1]
p_cat = (np.abs(null_rc) >= abs(r_cat)).mean()
print(f"category: corr(succ-effect, mean steps) over {len(tc)} cats = {r_cat:.3f}, perm p={p_cat:.4f} (null sd {null_rc.std():.3f})")

# also lift among successes: do tall successes lift higher (they must clear counter edge?) - context only
print("\ncontext: success max_lift by tall:", s.groupby(tall_s)["max_lift"].mean().round(3).to_dict())
