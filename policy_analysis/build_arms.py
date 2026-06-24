"""Turn the acquisition allocation into concrete mg-pool episode indices for the
two experiment arms:
  - CORE arm  : the allocated per-category demos (the algorithm's pick)
  - RANDOM arm: same total, sampled uniformly from the whole pool (the control)

Outputs episode-index lists (small, portable to the H100) for building the two
LeRobot fine-tune subsets.

  python policy_analysis/build_arms.py --plan weakregion/subsample_plan.json \
      --mg_dir <mg lerobot dir> --out weakregion/arms.json
"""
import argparse
import json
import os
import re
import random
from collections import defaultdict


def norm(s):
    return str(s).strip().lower().replace(" ", "_")


def episode_categories(mg_dir):
    """episode_index -> normalized object category."""
    out = {}
    for line in open(os.path.join(mg_dir, "meta", "episodes.jsonl")):
        e = json.loads(line)
        obj = None
        for t in e.get("tasks", []):
            m = re.match(r"Pick the (.+?) from the counter", t)
            if m:
                obj = norm(m.group(1)); break
        out[e["episode_index"]] = obj
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", default=None,
                   help="category-allocation plan (the heuristic scorer's output)")
    p.add_argument("--core_file", default=None,
                   help="explicit selected-episode list from ANY scorer (pluggable harness); "
                        "json list or {'core_episodes': [...]}")
    p.add_argument("--mg_dir", required=True)
    p.add_argument("--out", default="weakregion/arms.json")
    p.add_argument("--base_n", type=int, default=0,
                   help="also sample a shared base set of N random demos, disjoint from both arms")
    p.add_argument("--base_file", default=None,
                   help="reuse an EXACT base set from another arms.json (keeps the base identical "
                        "across selection methods so only the core differs)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = random.Random(args.seed)

    ep_cat = episode_categories(args.mg_dir)
    # CORE arm: from an explicit scorer selection (pluggable) OR the category allocation
    if args.core_file:
        sel = json.load(open(args.core_file))
        core = sorted(sel["core_episodes"] if isinstance(sel, dict) else sel)
    else:
        assert args.plan, "need --plan or --core_file"
        alloc = json.load(open(args.plan))["allocation"]
        by_cat = defaultdict(list)
        for ep, c in ep_cat.items():
            if c is not None:
                by_cat[c].append(ep)
        for c in by_cat:
            by_cat[c].sort()
        core = []
        for cat, k in alloc.items():
            avail = by_cat.get(cat, [])
            core += rng.sample(avail, min(k, len(avail)))
        core = sorted(core)

    # RANDOM arm: same total, uniform over the whole pool
    all_eps = sorted(ep for ep, c in ep_cat.items() if c is not None)
    rnd = sorted(rng.sample(all_eps, len(core)))

    # SHARED base (common to both arms, disjoint from both): keeps pi0 anchored.
    # --base_file reuses an EXACT base from a prior arms.json so the base is identical
    # across selection methods (only the core differs -> clean comparison).
    base = []
    if args.base_file:
        b = json.load(open(args.base_file))
        base = sorted(b["base_episodes"] if isinstance(b, dict) else b)
    elif args.base_n:
        used = set(core) | set(rnd)
        base_pool = [e for e in all_eps if e not in used]
        base = sorted(rng.sample(base_pool, min(args.base_n, len(base_pool))))

    # sanity: category mix of each arm
    def mix(eps):
        m = defaultdict(int)
        for e in eps:
            m[ep_cat[e]] += 1
        return dict(sorted(m.items(), key=lambda x: -x[1]))

    out = {"n_core": len(core), "n_random": len(rnd), "n_base": len(base), "seed": args.seed,
           "core_episodes": core, "random_episodes": rnd, "base_episodes": base}
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"CORE arm: {len(core)} demos across {len(set(ep_cat[e] for e in core if ep_cat.get(e)))} categories")
    print(f"RANDOM arm: {len(rnd)} demos, uniform over pool")
    if base:
        print(f"SHARED base: {len(base)} demos, uniform, disjoint from both arms")
    print(f"\ncore mix:   {mix(core)}")
    print(f"\nrandom mix (top 12): {dict(list(mix(rnd).items())[:12])}")
    print(f"\noverlap core∩random: {len(set(core) & set(rnd))} | "
          f"base∩(core∪random): {len(set(base) & (set(core) | set(rnd)))}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
