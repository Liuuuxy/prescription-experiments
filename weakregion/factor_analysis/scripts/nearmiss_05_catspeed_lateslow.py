"""Fix category-speed permutation (use integer codes) + composition of late successes."""
import pandas as pd, numpy as np

SP = "/data/xinyua11/tmp/factor_analysis_scratch"
rng = np.random.default_rng(0)
df = pd.read_csv(f"{SP}/pooled_episodes.csv")

def demean(sub, col):
    return sub[col] - sub.groupby("run")[col].transform("mean")

df["d_succ"] = demean(df, "success")
s = df[(df.success == 1) & df.steps_to_success.notna()].copy()
s["steps"] = s.steps_to_success.astype(float)
s["d_steps"] = demean(s, "steps")

cat_codes, cat_uniques = pd.factorize(df.object_category)
df["cat_code"] = cat_codes
s["cat_code"] = df.cat_code.reindex(s.index)
cat_eff = df.groupby("cat_code")["d_succ"].mean()

tc = s.groupby("cat_code")["d_steps"].agg(["mean", "count"])
tc = tc[tc["count"] >= 15]
ec = cat_eff.reindex(tc.index)
r_cat = np.corrcoef(ec, tc["mean"])[0, 1]

cat = s.cat_code.to_numpy(dtype=np.int64)
dd = s.d_steps.to_numpy()
runs = s.run.to_numpy(dtype=object)
order = np.argsort(runs.astype(str), kind="stable")
cat_s, dd_s, runs_s = cat[order], dd[order], runs[order]
bounds = np.flatnonzero(np.r_[True, runs_s[1:] != runs_s[:-1], True])
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

# ---- who are the LATE successes (steps>480, last 20% of horizon)? ----
late = s.steps > 480
early = s.steps <= 300
rim_s = np.maximum(s.obj_x_rel.abs(), s.obj_y_rel.abs()) > 0.65
style_eff = df.groupby("style_id")["d_succ"].mean()
s["sty_eff"] = style_eff.reindex(s.style_id).to_numpy()
s["cat_eff"] = cat_eff.reindex(s.cat_code).to_numpy()
tall_s = s.obj_height > 0.21
print(f"\nlate successes (steps>480): n={late.sum()}; early (<=300): n={early.sum()}")
for name, col in [("frac rim", rim_s), ("frac tall", tall_s)]:
    print(f"  {name}: late {col[late].mean():.3f} vs early {col[early].mean():.3f}")
for name in ["sty_eff", "cat_eff", "obj_height"]:
    print(f"  mean {name}: late {s.loc[late,name].mean():+.4f} vs early {s.loc[early,name].mean():+.4f}")

# predicted difficulty composite: is d_steps correlated with (negative) predicted success?
# simple composite: cat_eff + sty_eff + rim indicator effect
comp = -(s.cat_eff + s.sty_eff) + 0.18 * rim_s.to_numpy()
r = np.corrcoef(comp, s.d_steps)[0, 1]
print(f"\ncorr(difficulty composite, demeaned steps) episode-level = {r:.3f} (n={len(s)})")
