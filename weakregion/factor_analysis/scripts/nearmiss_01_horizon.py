"""Angle: timeouts/near-misses. Part 1: horizon inference from steps_to_success."""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SP = "/data/xinyua11/tmp/factor_analysis_scratch"
df = pd.read_csv(f"{SP}/pooled_episodes.csv")
print("rows", len(df), "runs", df.run.nunique())

s = df[df.success == 1].copy()
s["steps"] = s.steps_to_success.astype(float)
print("\n=== steps_to_success overall ===")
print(s.steps.describe())
print("max:", s.steps.max())

# per-run max & quantiles
print("\n=== per-run steps stats ===")
g = s.groupby("run")["steps"].agg(["count", "min", "median", lambda x: x.quantile(0.9), "max"])
g.columns = ["n_succ", "min", "median", "q90", "max"]
print(g.sort_values("max").to_string())

# histogram of steps
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(s.steps, bins=60, color="steelblue", edgecolor="k", lw=0.3)
axes[0].set_xlabel("steps_to_success"); axes[0].set_ylabel("count"); axes[0].set_title("All successes (n=%d)" % len(s))
# zoom on last 20% of horizon
H = s.steps.max()
tail = s[s.steps > 0.8 * H]
axes[1].hist(tail.steps, bins=40, color="tomato", edgecolor="k", lw=0.3)
axes[1].set_xlabel("steps_to_success (top 20%% of horizon, H=%d)" % H)
axes[1].set_title("Tail: n=%d (%.1f%% of successes)" % (len(tail), 100*len(tail)/len(s)))
plt.tight_layout(); plt.savefig(f"{SP}/nearmiss_steps_hist.png", dpi=110)

# fraction of successes in last 5% / 10% / 20% of horizon (per run, using per-run max as horizon proxy? or global)
print("\n=== bunching near limit (global horizon H=%d) ===" % H)
for frac in [0.95, 0.90, 0.80, 0.50]:
    n = (s.steps > frac * H).sum()
    print(f"steps > {frac:.2f}*H: {n} ({100*n/len(s):.2f}% of successes)")

# If failures were "slow near-successes" we'd expect a pile-up of successes just under H.
# Compare density in last decile vs a uniform-tail expectation from the preceding decile.
d_last = ((s.steps > 0.9*H)).sum()
d_prev = ((s.steps > 0.8*H) & (s.steps <= 0.9*H)).sum()
print(f"successes in (0.8H,0.9H]: {d_prev}, in (0.9H,H]: {d_last} -> ratio {d_last/max(d_prev,1):.2f}")

# per-run horizon: do runs share the same max?
print("\nunique per-run maxima:", sorted(g["max"].unique()))
print("\nglobal quantiles:", s.steps.quantile([0.5,0.9,0.95,0.99,1.0]).to_dict())
