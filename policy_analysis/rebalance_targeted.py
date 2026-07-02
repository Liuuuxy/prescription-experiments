"""Re-derive the 'hard' (targeted) object categories from a BALANCED weak-region eval.

Merges the balanced baseline-pi0 eval shards (equal-ish episodes per category), recomputes the
per-category failure rate + Wilson lower-bound on the balanced samples, and re-ranks. This removes
the small-sample / frequency-distributed seam in the original targeted-10 (which rested on ~5-11
frequency-distributed episodes per category).

  python policy_analysis/rebalance_targeted.py --dirs weakregion/eval_bal_A weakregion/eval_bal_B \
      --topk 10 --z 1.96
"""
import argparse
import json
import math
import os
from collections import defaultdict

OLD_TARGETED = ["juice", "spray", "pitcher", "canned_food", "soap_dispenser",
                "tupperware", "cheese_grater", "ice_cube", "cream_cheese_stick", "jar"]


def wilson_lb(k, n, z=1.96):
    """Wilson score lower bound for a proportion k/n."""
    if n == 0:
        return 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["weakregion/eval_bal_A", "weakregion/eval_bal_B"])
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--z", type=float, default=1.96)
    a = ap.parse_args()

    by = defaultdict(lambda: [0, 0])  # cat -> [fails, n]
    seen = set()
    for d in a.dirs:
        p = os.path.join(d, "weakregion.json")
        if not os.path.exists(p):
            print(f"!! missing {p}"); continue
        for e in json.load(open(p))["episodes"]:
            key = (e.get("object_category"), e.get("episode"), d)  # episodes are seed-disjoint across shards
            c = e.get("object_category")
            if c is None:
                continue
            by[c][0] += int(not e["success"]); by[c][1] += 1
    rows = [(c, by[c][0], by[c][1], by[c][0] / by[c][1], wilson_lb(by[c][0], by[c][1], a.z))
            for c in by]
    rows.sort(key=lambda r: -r[4])  # by Wilson LB of failure rate

    print(f"{'category':22s} {'fail/n':>8s} {'rate':>5s} {'WilsonLB':>9s}  old?")
    for c, f, n, r, lb in rows:
        mark = "  <-- OLD targeted-10" if c in OLD_TARGETED else ""
        print(f"{c:22s} {f:>3}/{n:<3} {r:>5.2f} {lb:>9.3f}{mark}")

    new = [r[0] for r in rows[:a.topk]]
    old = set(OLD_TARGETED)
    print(f"\n=== NEW top-{a.topk} hard categories (by balanced Wilson-LB) ===")
    print(new)
    print(f"\noverlap with OLD targeted-10: {len(set(new) & old)}/{a.topk}")
    print(f"  dropped from old: {sorted(old - set(new))}")
    print(f"  newly hard:       {sorted(set(new) - old)}")
    tot_n = sum(by[c][1] for c in by)
    print(f"\nbalanced eval: {len(by)} categories, {tot_n} episodes, "
          f"median {sorted(by[c][1] for c in by)[len(by)//2]} eps/cat")
    json.dump({"new_targeted": new, "old_targeted": OLD_TARGETED,
               "per_category": {c: {"fails": by[c][0], "n": by[c][1],
                                    "rate": by[c][0]/by[c][1], "wilson_lb": wilson_lb(by[c][0], by[c][1], a.z)}
                                for c in by}},
              open("weakregion/targeted_rebalanced.json", "w"), indent=2)
    print("wrote weakregion/targeted_rebalanced.json")


if __name__ == "__main__":
    main()
