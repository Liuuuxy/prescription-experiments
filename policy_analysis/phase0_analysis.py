"""Phase 0 analysis: turn baseline DP runs into the go/no-go for the real experiment.

Inputs: success rate of DP trained at several dataset sizes N, each with several
seeds (k successes / n rollouts). Computes:
  - data-efficiency slope (is the baseline data-SENSITIVE? if flat, stop).
  - sigma_seed = between-seed SD of success (THE unknown the power analysis needs),
    properly separated from binomial eval noise.
  - the resulting seeds x rollouts budget needed for Phase 1, and a verdict.

Two input modes:
  (A) --results 'N,seed,k,n' triples (quick / manual), or
  (B) --evals_dir : scan eval_log.json files under <dir>/N*_seed*/.../eval_log.json
Self-test: `python phase0_analysis.py --selftest`.
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

Z = 1.959964 + 0.841621  # power 0.80, alpha 0.05 two-sided


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


def parse_eval_log(path):
    d = json.load(open(path))
    rate = next((v for k, v in d.items() if k.startswith("success_rate/")), None)
    n = sum(1 for k in d if k.startswith("test/sim_max_reward_")) or None
    return rate, n


def load_evals_dir(d):
    rows = []
    for p in glob.glob(os.path.join(d, "**", "eval_log.json"), recursive=True):
        m = re.search(r"N(\d+)_seed(\d+)", p)
        if not m:
            continue
        N, seed = int(m.group(1)), int(m.group(2))
        rate, n = parse_eval_log(p)
        if rate is None or not n:
            continue
        rows.append((N, seed, int(round(rate * n)), n))
    return rows


def analyze(rows):
    # rows: list of (N, seed, k, n)
    byN = defaultdict(list)
    for N, seed, k, n in rows:
        byN[N].append((seed, k, n))
    Ns = sorted(byN)
    print(f"{'N':>5} {'seeds':>6} {'mean_succ':>10} {'sigma_seed':>11} {'eval_noise':>11} {'per-seed rates'}")
    sigma_by_N = {}
    mean_by_N = {}
    for N in Ns:
        rates = np.array([k / n for _, k, n in byN[N]])
        ns = np.array([n for _, _, n in byN[N]])
        mean = rates.mean()
        total_sd = rates.std(ddof=1) if len(rates) > 1 else float("nan")
        # expected binomial (eval) variance, averaged over seeds
        eval_var = np.mean([p * (1 - p) / n for p, n in zip(rates, ns)])
        # between-seed (epistemic) variance = total - eval (floored at 0)
        between_var = max((total_sd ** 2 if len(rates) > 1 else 0.0) - eval_var, 0.0)
        sigma_seed = np.sqrt(between_var)
        sigma_by_N[N] = sigma_seed
        mean_by_N[N] = mean
        rate_str = " ".join(f"{r:.0%}" for r in rates)
        print(f"{N:>5} {len(rates):>6} {mean:>9.1%} {sigma_seed:>10.3f} {np.sqrt(eval_var):>10.3f}   {rate_str}")

    # data-efficiency slope (success vs log N)
    print("\n=== data-efficiency ===")
    if len(Ns) >= 2:
        x = np.log(np.array(Ns, float)); y = np.array([mean_by_N[N] for N in Ns])
        slope = np.polyfit(x, y, 1)[0]
        print(f"slope d(success)/d(lnN) = {slope:+.3f}  "
              f"(=> +{slope*np.log(2):.1%} success per doubling of demos)")
        if slope < 0.02:
            print("  WARNING: baseline looks DATA-INSENSITIVE -> added data may not move it."
                  " Re-pick N (smaller baseline) or task before Phase 1.")
        else:
            print("  Baseline is data-sensitive (good — added data should help).")
    # pooled sigma_seed estimate
    sig = np.nanmean([s for s in sigma_by_N.values() if not np.isnan(s)])
    print(f"\nEstimated sigma_seed (between-seed SD of success) ~ {sig:.3f}")

    # Phase-1 budget given measured sigma_seed
    print("\n=== Phase 1 budget (seeds/arm for 80% power, weak-region fraction f=0.5) ===")
    p = float(np.mean(list(mean_by_N.values())))
    f = 0.5
    print(f"  (using baseline p={p:.2f})")
    for delta in (0.05, 0.10, 0.15):
        row = []
        for R in (100, 200):
            Rw = max(f * R, 1)
            S = 2 * (sig ** 2 + p * (1 - p) / Rw) * (Z / delta) ** 2
            row.append(f"R={R}:{S:.0f}")
        print(f"  detect Delta={delta:.0%}: " + "  ".join(row) + " seeds/arm")
    print("\nVerdict: pick the smallest Delta you believe the algorithm produces, read the")
    print("seeds/arm, and budget S x 2 arms x #N-points trainings. If that's infeasible,")
    print("strengthen the acquisition signal / weaken the baseline to enlarge Delta first.")


def selftest():
    rng = np.random.default_rng(0)
    rows = []
    truth = {25: 0.20, 50: 0.35, 100: 0.50}   # data-sensitive baseline
    sigma = 0.06
    for N, base in truth.items():
        for seed in range(4):
            tr = np.clip(rng.normal(base, sigma), 0.01, 0.99)
            n = 100; k = rng.binomial(n, tr)
            rows.append((N, seed, k, n))
    print("SELF-TEST (truth: slope>0, sigma_seed~0.06):")
    analyze(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals_dir")
    ap.add_argument("--results", help="semicolon list of 'N,seed,k,n'")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.evals_dir:
        rows = load_evals_dir(a.evals_dir)
    elif a.results:
        rows = [tuple(int(x) for x in t.split(",")) for t in a.results.split(";") if t.strip()]
    else:
        print("provide --evals_dir, --results, or --selftest"); return
    if not rows:
        print("no results found"); return
    analyze(rows)


if __name__ == "__main__":
    main()
