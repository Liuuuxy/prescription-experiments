"""Permutation p-values for late-success (steps>480) enrichment of rim/tall."""
import pandas as pd, numpy as np

SP = "/data/xinyua11/tmp/factor_analysis_scratch"
df = pd.read_csv(f"{SP}/pooled_episodes.csv")
s = df[(df.success == 1) & df.steps_to_success.notna()].copy()
s["steps"] = s.steps_to_success.astype(float)
s["late"] = (s.steps > 480).astype(float)
s["d_late"] = s.late - s.groupby("run")["late"].transform("mean")

def strat_perm_diff(sub, mask, ycol, n=5000, seed=1):
    rng = np.random.default_rng(seed)
    obs = sub.loc[mask, ycol].mean() - sub.loc[~mask, ycol].mean()
    runs = sub.run.to_numpy(dtype=object); y = sub[ycol].to_numpy(); m = np.asarray(mask).copy()
    order = np.argsort(runs.astype(str), kind="stable")
    runs_s, y_s, m_s = runs[order], y[order], m[order]
    bounds = np.flatnonzero(np.r_[True, runs_s[1:] != runs_s[:-1], True])
    null = np.empty(n)
    for i in range(n):
        mp = m_s.copy()
        for a, b in zip(bounds[:-1], bounds[1:]):
            rng.shuffle(mp[a:b])
        null[i] = y_s[mp].mean() - y_s[~mp].mean()
    return obs, (np.abs(null) >= abs(obs)).mean()

rim = np.maximum(s.obj_x_rel.abs(), s.obj_y_rel.abs()) > 0.65
tall = s.obj_height > 0.21
for name, mask in [("rim", rim), ("tall>21cm", tall)]:
    obs, p = strat_perm_diff(s, mask, "d_late")
    print(f"{name}: P(late|success) {s.loc[mask,'late'].mean():.3f} vs {s.loc[~mask,'late'].mean():.3f}, demeaned diff={obs:+.4f}, perm p={p:.4f}")
