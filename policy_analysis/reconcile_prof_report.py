"""Reconcile the professor's n=30 PickPlaceCounterToSink report against our
n=100 (GR00T) / n=150 (pi0) weak-region runs.

(b) GR00T grasp-vs-transport failure split — tests the report's claim that
    "GR00T = transport stall (grasps but stalls), pi0 = grasp failure".
(c) Merge the report's per-object success table with our per-category rates and
    object heights — tests whether object identity / the report's table is a
    usable targeting signal, and whether it reduces to height.

Run: python reconcile_prof_report.py
Reads the two weakregion.json logs under ../weakregion/ (paths below).
"""
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
WR = os.path.normpath(os.path.join(HERE, "..", "weakregion"))
PI0 = os.path.join(WR, "pi0_PickPlaceCounterToSink_n150", "weakregion.json")
GROOT = os.path.join(WR, "groot_PickPlaceCounterToSink", "weakregion.json")


def load(path):
    return json.load(open(path))["episodes"]


def corr(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x == x and y == y]
    if len(pairs) < 2:
        return float("nan")
    a, b = zip(*pairs)
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((y - mb) ** 2 for y in b) ** 0.5
    return cov / (va * vb) if va * vb else float("nan")


def part_b():
    print("=" * 64)
    print("(b) GRASP vs TRANSPORT failure split")
    print("=" * 64)
    for name, path in [("pi0 (n150)", PI0), ("GR00T (n100)", GROOT)]:
        eps = load(path)
        fails = [e for e in eps if not e["success"]]
        sr = sum(e["success"] for e in eps) / len(eps)
        from collections import Counter
        phase = Counter(e["failure_phase"] for e in fails)
        no_grasp = phase.get("fail_no_grasp", 0)
        transport = len(fails) - no_grasp
        lift2 = sum(1 for e in fails if (e.get("max_lift") or 0) > 0.02)
        lift5 = sum(1 for e in fails if (e.get("max_lift") or 0) > 0.05)
        print(f"\n{name}: success {sr:.0%}, {len(fails)} failures")
        print(f"  phase-tag:  no_grasp {no_grasp}/{len(fails)} ({no_grasp/len(fails):.0%})"
              f" | transport/place {transport}/{len(fails)} ({transport/len(fails):.0%})")
        print(f"  max_lift>2cm (object meaningfully lifted): {lift2}/{len(fails)}"
              f" ({lift2/len(fails):.0%})   >5cm: {lift5}/{len(fails)} ({lift5/len(fails):.0%})")
    print("\nVERDICT: both policies are GRASP-dominated (~80% no-grasp). The report's"
          "\n  direction is right — GR00T's transport-stall fraction is ~2x pi0's —"
          "\n  but it is GR00T's SECONDARY mode, not its primary one.")


# professor's per-object table (object -> pi0, groot, combined, our_category)
PROF = {
    "carrot":        (1.00, 0.67, 0.83, "carrot"),
    "bottled drink": (1.00, 0.67, 0.83, "bottled_drink"),
    "sponge":        (1.00, 0.00, 0.50, "sponge"),
    "glass cup":     (0.33, 0.33, 0.33, "glass_cup"),
    "cream cheese":  (0.33, 0.33, 0.33, "cream_cheese_stick"),
    "salt/pepper":   (0.33, 0.33, 0.33, "salt_and_pepper_shaker"),
    "avocado":       (0.00, 0.67, 0.33, "avocado"),
    "milk":          (0.00, 0.33, 0.17, "milk"),
    "shrimp":        (0.00, 0.33, 0.17, "shrimp"),
    "beer":          (0.00, 0.33, 0.17, "beer"),
}


def part_c():
    print("\n" + "=" * 64)
    print("(c) MERGE: professor per-object table vs our rates + height")
    print("=" * 64)
    rows = []
    for path in (PI0, GROOT):
        for e in load(path):
            rows.append((e["object_category"], e["success"], e.get("obj_height")))

    def ours(cat):
        rs = [r for r in rows if r[0] == cat]
        if not rs:
            return 0, float("nan"), float("nan")
        sr = sum(r[1] for r in rs) / len(rs)
        hs = [r[2] for r in rs if r[2] is not None]
        return len(rs), sr, (statistics.mean(hs) * 100 if hs else float("nan"))

    print(f"\n{'object':<14}{'profPi0':>8}{'profG':>7}{'profComb':>9}"
          f"{'ourComb':>9}{'ourN':>6}{'height':>8}")
    pc, oc, hh = [], [], []
    for o, (p, g, c, cat) in PROF.items():
        n, sr, h = ours(cat)
        print(f"{o:<14}{p:>8.2f}{g:>7.2f}{c:>9.2f}{sr:>8.0%}{n:>6}{h:>8.1f}")
        if n > 0:
            pc.append(c); oc.append(sr); hh.append(h)
    print(f"\n  prof-combined vs our-combined (n={len(oc)} objects): r={corr(pc, oc):+.2f}"
          "  -> evals only weakly agree at single-instance level")
    print(f"  prof-combined vs height (his 10 objects):     r={corr(pc, hh):+.2f}"
          "  -> height washes out on a small object subset")

    # robust: height vs success across ALL our categories with n>=3
    agg = {}
    for c0, s, h in rows:
        agg.setdefault(c0, []).append((s, h))
    H, S = [], []
    for c0, v in agg.items():
        if len(v) >= 3:
            S.append(sum(s for s, _ in v) / len(v))
            hs = [h for _, h in v if h is not None]
            H.append(statistics.mean(hs) * 100 if hs else float("nan"))
    print(f"\n  height vs success across ALL {len(S)} our categories (n>=3): r={corr(H, S):+.2f}"
          f"  (R^2~{corr(H, S)**2:.2f})")
    print("\nVERDICT: object identity is a NOISY per-episode label (evals agree only"
          "\n  r=0.36); height is the robust AGGREGATE signal (r=-0.40 over 51 cats)"
          "\n  but vanishes on small subsets. Hand-picked geometry/identity is too"
          "\n  weak for per-episode targeting -> need uncertainty/disagreement.")


if __name__ == "__main__":
    part_b()
    part_c()
