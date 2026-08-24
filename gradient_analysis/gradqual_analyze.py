"""Morning read-out for the gradient-quality causal test (2026-08-13 launch).

Compares gradqual_hi vs gradqual_lo (paired seed 1003, B=200, 20k recipe, frozen E)
against the pre-registered comparators, and prints the per-stratum breakdown plus a
paired-bootstrap CI on the hi-lo gap computed from the per-episode eval rows.

Pre-registered read (memory: gradqual-causal-test):
  hi - lo >= ~+5pp   -> the gradient quality score reproduces the style race's quality
                        effect -> zero-rollout pre-training filter is causally validated
  hi - lo ~ 0        -> Q6 signal is real but NOT actionable; do not build on it

Run: /data/xinyua11/conda/envs/robocasa/bin/python gradient_analysis/gradqual_analyze.py
"""
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "/data/xinyua11/robocasa")
from bandit_v1 import ledger  # noqa: E402

SIGMA_E = 0.0333
COMPARATORS = ["random_j3", "style_hi_j3", "style_lo_j3", "mid_band_j3",
               "tall_vessel_grasp_fail_j3", "easy_band_j3", "gradarm_a_j3", "gradarm_b_j3"]


def main():
    p = ledger.read("pulls")
    done = p[p.status.isin(["ok", "smoke"])]
    have = {str(r.pull_id): float(r.delta) for _, r in done.iterrows()}
    rounds = [j for j in (3, 4, 5) if f"gradqual_hi_j{j}" in have and f"gradqual_lo_j{j}" in have]
    if not rounds:
        print("[wait] no round has both arms yet")
        return
    print(f"complete paired rounds: {rounds}\n")
    gaps = []
    for j in rounds:
        h, l = have[f"gradqual_hi_j{j}"], have[f"gradqual_lo_j{j}"]
        gaps.append(h - l)
        print(f"  round {j} (seed {1000+j}): hi {h*100:+.2f}  lo {l*100:+.2f}  gap {(h-l)*100:+.2f} pp")
    g = np.array(gaps)
    print(f"\n  mean gap over {len(g)} rounds: {g.mean()*100:+.2f} pp; "
          f"signs {'all positive' if (g>0).all() else 'MIXED — sign flip'}; "
          f"per-round sd {g.std(ddof=1)*100:.2f} pp" if len(g) > 1 else "")
    if len(g) > 1:
        se = g.std(ddof=1) / np.sqrt(len(g))
        print(f"  mean/SE = {g.mean()/se:.2f}  (style race for reference: +8.9/+2.9/+2.9, mean +4.9, p~.07)")
    print()
    if 3 not in rounds:
        return

    hi, lo = have["gradqual_hi_j3"], have["gradqual_lo_j3"]  # round-3 headline (kept)
    print(f"gradqual_hi_j3  delta {hi*100:+.2f} pp")
    print(f"gradqual_lo_j3  delta {lo*100:+.2f} pp")
    print(f"HI - LO gap     {(hi-lo)*100:+.2f} pp   (style race hi-lo: +8.9/+2.9/+2.9, mean +4.9)")
    print(f"per-pull noise sigma_e = {SIGMA_E*100:.1f} pp\n")
    print("paired-seed comparators (same seed 1003):")
    for c in COMPARATORS:
        if c in have:
            print(f"  {c:30s} {have[c]*100:+.2f} pp")

    # per-stratum + paired bootstrap on the shared eval starts
    ep = ledger.read("episodes"); em = ledger.read("E_manifest")
    ev = ep[ep.phase == "eval"].merge(em[["start_id", "stratum"]], on="start_id")
    sub = ev[ev.pull_id.isin(["gradqual_hi_j3", "gradqual_lo_j3"])]
    if len(sub):
        print("\nper-stratum success:")
        print(sub.groupby(["pull_id", "stratum"]).success.mean().unstack().round(3).to_string())
        h = sub[sub.pull_id == "gradqual_hi_j3"].groupby("start_id").success.mean()
        l = sub[sub.pull_id == "gradqual_lo_j3"].groupby("start_id").success.mean()
        common = h.index.intersection(l.index)
        d = (h[common] - l[common]).to_numpy()
        rng = np.random.RandomState(0)
        boot = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(5000)]
        lo_ci, hi_ci = np.percentile(boot, [2.5, 97.5])
        print(f"\npaired per-start hi-lo: {d.mean()*100:+.2f} pp "
              f"[95% CI {lo_ci*100:+.2f}, {hi_ci*100:+.2f}] over {len(d)} shared starts")
        if lo_ci > 0:
            print("VERDICT: CI excludes 0 -> the gradient quality score causally moves outcomes.")
        elif hi_ci < 0:
            print("VERDICT: CI excludes 0 in the WRONG direction -> the score is anti-predictive.")
        else:
            print("VERDICT: CI includes 0 -> NOT significant at this many rounds. Direction is "
                  f"{'as predicted (hi > lo)' if d.mean() > 0 else 'against prediction'}; "
                  "more paired rounds are required before any claim.")


if __name__ == "__main__":
    main()
