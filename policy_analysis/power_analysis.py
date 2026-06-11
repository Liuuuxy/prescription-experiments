"""Power analysis for the targeted-vs-random data experiment.

Two arms (targeted, random): each trains S DP seeds, evaluates each on R
rollouts, and we compare mean success in the WEAK region (where targeting should
help). Variance per arm-mean = (sigma_seed^2 + p(1-p)/R_w) / S, where
R_w = f * R rollouts land in the weak region (fraction f). To detect an absolute
difference Delta at alpha=0.05, power=0.80 (two-sided):
    Delta >= 2.80 * sqrt( 2 * (sigma_seed^2 + p(1-p)/R_w) / S )
so   S   >= 2 * (sigma_seed^2 + p(1-p)/R_w) * (2.80/Delta)^2 .

sigma_seed (between-seed SD of success) is THE key unknown — measure it by
training 2-3 baseline seeds first. Defaults span plausible deep-IL values.
Includes a Monte-Carlo check of the analytic power.
"""
import numpy as np

Z = 1.959964 + 0.841621  # z_{.975} + z_{.80} = 2.80 (power 0.80, alpha 0.05 two-sided)


def seeds_needed(delta, sigma_seed, p, R, f):
    Rw = max(f * R, 1)
    var_unit = sigma_seed ** 2 + p * (1 - p) / Rw
    return 2 * var_unit * (Z / delta) ** 2


def mde(S, sigma_seed, p, R, f):
    Rw = max(f * R, 1)
    var_unit = sigma_seed ** 2 + p * (1 - p) / Rw
    return Z * np.sqrt(2 * var_unit / S)


def mc_power(delta, sigma_seed, p, R, f, S, trials=4000, seed=0):
    """Monte-Carlo power: simulate S seeds/arm, Welch t-test on seed means."""
    rng = np.random.default_rng(seed)
    Rw = int(max(round(f * R), 1))
    from math import erf
    hits = 0
    for _ in range(trials):
        # targeted arm true per-seed rates ~ N(p+delta, sigma_seed), clipped
        tr = np.clip(rng.normal(p + delta, sigma_seed, S), 0.01, 0.99)
        rr = np.clip(rng.normal(p, sigma_seed, S), 0.01, 0.99)
        # measured = binomial eval on Rw weak-region rollouts
        tm = rng.binomial(Rw, tr) / Rw
        rm = rng.binomial(Rw, rr) / Rw
        m1, m2 = tm.mean(), rm.mean()
        s = np.sqrt(tm.var(ddof=1) / S + rm.var(ddof=1) / S) + 1e-9
        t = (m1 - m2) / s
        # ~normal approx for the two-sided p<0.05 (|t|>1.96); conservative-ish at small S
        if abs(t) > 1.96:
            hits += 1
    return hits / trials


def main():
    p = 0.45          # baseline success in the weak region (plausible mid-range)
    f = 0.5           # weak region = ~half the init states ("tall objects")
    print(f"Assumptions: baseline weak-region success p={p}, weak-region fraction f={f}")
    print("(p,f from our pi0 data; sigma_seed is the key unknown -> measure with 2-3 baseline seeds.)\n")

    print("=== Seeds per arm needed for 80% power (alpha=0.05) ===")
    print(f"{'Delta':>6} {'sigma_seed':>11} {'R=50':>7} {'R=100':>7} {'R=200':>7}")
    for delta in (0.05, 0.10, 0.15):
        for sig in (0.05, 0.10):
            row = [f"{seeds_needed(delta, sig, p, R, f):>6.1f}" for R in (50, 100, 200)]
            print(f"{delta:>6.0%} {sig:>11.2f} " + " ".join(f"{x:>6}" for x in row))

    print("\n=== Minimum detectable effect (MDE) for a FEASIBLE budget ===")
    for S in (3, 5):
        for R in (100, 200):
            for sig in (0.05, 0.10):
                print(f"  S={S} seeds, R={R} rollouts, sigma_seed={sig}: "
                      f"MDE = {mde(S, sig, p, R, f):.1%} absolute success-rate diff")

    print("\n=== Monte-Carlo power check (analytic vs simulated) ===")
    for delta, sig, R, S in [(0.10, 0.05, 100, 3), (0.10, 0.10, 100, 5),
                             (0.15, 0.10, 200, 5), (0.10, 0.05, 200, 3)]:
        an = "yes" if seeds_needed(delta, sig, p, R, f) <= S else "no"
        sim = mc_power(delta, sig, p, R, f, S)
        print(f"  Delta={delta:.0%} sigma={sig} R={R} S={S}: "
              f"MC power={sim:.0%}  (analytic says S enough for 80%? {an})")

    print("\n=== TAKEAWAYS ===")
    print("- If sigma_seed is LARGE (~0.10), between-seed noise dominates: more ROLLOUTS")
    print("  barely help; you need more SEEDS. If small (~0.05), ~3-5 seeds x 100-200 rollouts")
    print("  can detect a 10-15% effect.")
    print("- Our targeting effect looks WEAK (geometry R^2~0.08), so the realized Delta may be")
    print("  small -> budget for >=5 seeds and R>=200, and FIRST measure sigma_seed (2-3 baseline")
    print("  seeds) before committing the full grid. Consider a within-region paired design and a")
    print("  continuous progress metric to cut variance.")


if __name__ == "__main__":
    main()
