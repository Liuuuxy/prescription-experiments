"""Part 1b: censoring estimate; Part 2: near-miss taxonomy of failures."""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SP = "/data/xinyua11/tmp/factor_analysis_scratch"
df = pd.read_csv(f"{SP}/pooled_episodes.csv")

# groot anomaly check
gr = df[df.run == "groot_PickPlaceCounterToSink"]
print("groot run: n=%d, success=%d, steps recorded for %d" %
      (len(gr), gr.success.sum(), gr[gr.success==1].steps_to_success.notna().sum()))
print("groot success rate:", gr.success.mean().round(3))

s = df[(df.success == 1) & df.steps_to_success.notna()].copy()
s["steps"] = s.steps_to_success.astype(float)
H = 600
fails = df[df.success == 0]
nf = len(fails)
print("\ntotal fails:", nf, "successes with steps:", len(s))

# hazard in successive 60-step windows: successes finishing in window / episodes still running at window start
print("\n=== success hazard per 60-step window (all runs pooled, steps-bearing runs only) ===")
runs_with_steps = s.run.unique()
d2 = df[df.run.isin(runs_with_steps)]
nf2 = (d2.success == 0).sum()
for lo in range(0, 600, 60):
    finishing = ((s.steps > lo) & (s.steps <= lo + 60)).sum()
    running = nf2 + (s.steps > lo).sum()
    print(f"  ({lo:3d},{lo+60:3d}]: finish={finishing:4d} running={running:5d} hazard={finishing/running:.4f}")
h_last = ((s.steps > 540)).sum() / (nf2 + (s.steps > 540).sum())
print(f"\nlast-window hazard={h_last:.4f} -> extending horizon +60 steps would rescue ~{h_last*nf2:.0f} of {nf2} fails (~{100*h_last:.1f}%)")

# ============ PART 2: taxonomy ============
print("\n=== failure_phase counts ===")
print(df.failure_phase.value_counts().to_string())

print("\n=== max_lift / min_sink_dist by phase ===")
for ph in ["success", "fail_no_grasp", "fail_grasped_no_transport", "fail_reached_sink_no_place"]:
    sub = df[df.failure_phase == ph]
    print(f"\n{ph} (n={len(sub)}):")
    print("  max_lift:      ", sub.max_lift.describe()[["mean","25%","50%","75%","max"]].round(3).to_dict())
    print("  min_sink_dist: ", sub.min_sink_dist.describe()[["mean","min","25%","50%","75%"]].round(3).to_dict())

# calibrate 'in sink' with successes' min_sink_dist
print("\nsuccess min_sink_dist quantiles:", df[df.success==1].min_sink_dist.quantile([0.5,0.9,0.95,0.99]).round(3).to_dict())

# Fine-grained taxonomy
fng = fails[fails.failure_phase == "fail_no_grasp"]
fgt = fails[fails.failure_phase == "fail_grasped_no_transport"]
frs = fails[fails.failure_phase == "fail_reached_sink_no_place"]

print("\n=== fail_no_grasp: max_lift split ===")
bins = [(-1, 0.005, "never_touched (<5mm lift)"),
        (0.005, 0.02, "nudged (0.5-2cm)"),
        (0.02, 0.05, "partial_lift (2-5cm)"),
        (0.05, 10, "lifted>5cm (label mismatch?)")]
for lo, hi, name in bins:
    n = ((fng.max_lift > lo) & (fng.max_lift <= hi)).sum()
    print(f"  {name}: {n} ({100*n/len(fng):.1f}%)")

print("\n=== fail_grasped_no_transport (n=%d): min_sink_dist ===" % len(fgt))
for lo, hi in [(0,0.1),(0.1,0.2),(0.2,0.3),(0.3,0.5),(0.5,99)]:
    n = ((fgt.min_sink_dist > lo) & (fgt.min_sink_dist <= hi)).sum()
    print(f"  ({lo},{hi}]m: {n} ({100*n/len(fgt):.1f}%)")
print("  max_lift quantiles:", fgt.max_lift.quantile([0.25,0.5,0.75,0.9]).round(3).to_dict())

print("\n=== fail_reached_sink_no_place (n=%d): min_sink_dist ===" % len(frs))
for lo, hi in [(0,0.05),(0.05,0.1),(0.1,0.2),(0.2,99)]:
    n = ((frs.min_sink_dist > lo) & (frs.min_sink_dist <= hi)).sum()
    print(f"  ({lo},{hi}]m: {n} ({100*n/len(frs):.1f}%)")

# overall near-miss counts: any failure with min_sink_dist < 0.2
for thr in [0.1, 0.15, 0.2]:
    n = (fails.min_sink_dist < thr).sum()
    print(f"\nfails with min_sink_dist < {thr}m: {n} ({100*n/nf:.1f}% of all fails)")
    print(fails[fails.min_sink_dist < thr].failure_phase.value_counts().to_dict())

# also: min_sink_dist distribution for fail_no_grasp (object starts at some distance; did it get knocked toward sink?)
print("\nfail_no_grasp min_sink_dist quantiles:", fng.min_sink_dist.quantile([0.05,0.25,0.5]).round(3).to_dict())

# plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist([fng.max_lift.clip(0, 0.10)], bins=40, color="gray", edgecolor="k", lw=0.3)
axes[0].set_title("fail_no_grasp max_lift (clipped 10cm)"); axes[0].set_xlabel("m")
axes[1].hist(fgt.min_sink_dist.clip(0, 1.2), bins=40, color="orange", edgecolor="k", lw=0.3)
axes[1].set_title("fail_grasped_no_transport min_sink_dist"); axes[1].set_xlabel("m")
axes[2].hist(frs.min_sink_dist.clip(0, 0.6), bins=40, color="tomato", edgecolor="k", lw=0.3)
axes[2].set_title("fail_reached_sink_no_place min_sink_dist"); axes[2].set_xlabel("m")
plt.tight_layout(); plt.savefig(f"{SP}/nearmiss_taxonomy_hists.png", dpi=110)
print("\nsaved nearmiss_taxonomy_hists.png")
