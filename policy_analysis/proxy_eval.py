"""Training-free proxy evaluation of a data-selection (Stage 1: coverage / relevance).

Scores any selected set of mg-pool episodes by how well it TARGETS pi0's measured
failure distribution -- no fine-tuning, no model forward passes. Used to (a) rank
selection methods cheaply and (b) calibrate against the real core-vs-random
fine-tune once it returns (does the proxy predict the winner?).

Metrics (per set):
  targeting_score  = mean pi0 failure-rate of the set's categories (E_demo[ p_fail(cat) ]).
                     Higher = the set concentrates on categories pi0 fails. Random ~= pi0's
                     base failure rate; a good targeted set is well above it.
  failure_recall   = fraction of pi0's TOTAL failure mass whose category appears in the set.
  overlap          = sum_c min(F[c], S[c])  (distribution overlap of set vs failure mass).

Heavier proxies (next tier, need more infra) are intentionally NOT here:
  - pi0 policy-surprise (BC loss): needs openpi RobocasaInputs to map mg->pi0 obs correctly.
  - influence / TracIn: needs pi0 LoRA gradients (H100).
  - kNN-policy rollout / DP-proxy student: need the env / a training run (H100).

  python policy_analysis/proxy_eval.py --student weakregion/pi0_student_n500/weakregion.json \
      --arms weakregion/arms.json --mg_dir <mg lerobot>
"""
import argparse
import json
import re
from collections import defaultdict, Counter


def norm(s):
    return str(s).strip().lower().replace(" ", "_")


def episode_categories(mg_dir):
    out = {}
    for line in open(f"{mg_dir}/meta/episodes.jsonl"):
        e = json.loads(line)
        obj = None
        for t in e.get("tasks", []):
            m = re.match(r"Pick the (.+?) from the counter", t)
            if m:
                obj = norm(m.group(1)); break
        out[e["episode_index"]] = obj
    return out


def failure_profile(student_path):
    eps = json.load(open(student_path))["episodes"]
    fail, n = defaultdict(int), defaultdict(int)
    for e in eps:
        c = norm(e["object_category"])
        n[c] += 1
        fail[c] += int(not e["success"])
    p_fail = {c: fail[c] / n[c] for c in n}           # per-category failure rate
    F_total = sum(fail.values())
    F = {c: fail[c] / F_total for c in fail if fail[c] > 0}  # failure-mass distribution
    base_rate = sum(fail.values()) / sum(n.values())
    return p_fail, F, base_rate


def score_set(eps, ep_cat, p_fail, F):
    cats = [ep_cat[e] for e in eps if ep_cat.get(e)]
    cnt = Counter(cats)
    tot = sum(cnt.values()) or 1
    S = {c: cnt[c] / tot for c in cnt}
    targeting = sum(S[c] * p_fail.get(c, 0.0) for c in S)      # E_demo[p_fail(cat)]
    recall = sum(F[c] for c in F if c in cnt)                  # failure mass covered
    overlap = sum(min(F.get(c, 0.0), S.get(c, 0.0)) for c in set(F) | set(S))
    return dict(n=len(eps), targeting=targeting, failure_recall=recall, overlap=overlap)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student", required=True)
    p.add_argument("--arms", required=True)
    p.add_argument("--mg_dir", required=True)
    args = p.parse_args()

    p_fail, F, base_rate = failure_profile(args.student)
    ep_cat = episode_categories(args.mg_dir)
    arms = json.load(open(args.arms))

    sets = {
        "core":        arms["core_episodes"],
        "random":      arms["random_episodes"],
        "base":        arms.get("base_episodes", []),
        "base+core":   arms.get("base_episodes", []) + arms["core_episodes"],
        "base+random": arms.get("base_episodes", []) + arms["random_episodes"],
    }
    print(f"pi0 base failure rate (random-data expectation): {base_rate:.3f}\n")
    print(f"{'set':14s} {'n':>4s} {'targeting':>10s} {'fail_recall':>12s} {'overlap':>8s}")
    out = {}
    for name, eps in sets.items():
        if not eps:
            continue
        s = score_set(eps, ep_cat, p_fail, F)
        out[name] = s
        print(f"{name:14s} {s['n']:>4d} {s['targeting']:>10.3f} {s['failure_recall']:>12.3f} {s['overlap']:>8.3f}")

    print("\nproxy prediction (Stage-1, relevance):")
    if "core" in out and "random" in out:
        dt = out["core"]["targeting"] - out["random"]["targeting"]
        print(f"  core vs random targeting: {out['core']['targeting']:.3f} vs "
              f"{out['random']['targeting']:.3f}  (Δ={dt:+.3f}) -> "
              f"{'CORE more relevant' if dt > 0 else 'RANDOM more relevant'}")
        print("  CALIBRATION: when the real pi0 fine-tune returns, check whether the arm with the "
              "higher targeting score actually wins. If yes, trust this proxy to rank new methods.")
    json.dump(out, open("weakregion/proxy_scores.json", "w"), indent=2)
    print("\nwrote weakregion/proxy_scores.json")


if __name__ == "__main__":
    main()
