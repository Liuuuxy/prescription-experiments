"""Core-algo allocation for the SUBSAMPLING setting (student = pi0, pool = mg).

No teacher term: selecting from a pre-existing pool has no per-attempt cost, so the
score collapses to  P(student fails) x uncertainty  (Wilson-LCB robust). Demos are
allocated proportional to score, then CAPPED by how many the mg pool actually has
for that object category (water-filling redistributes any capped overflow).

  python policy_analysis/acquisition_subsample.py \
    --student weakregion/pi0_student_n500/weakregion.json \
    --mg_dir <mg lerobot dir> --budget 200 --min_n 5
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from acquisition import wilson


def norm(name):
    return str(name).strip().lower().replace(" ", "_")


def pool_counts(mg_dir):
    tasks = {}
    for line in open(os.path.join(mg_dir, "meta", "tasks.jsonl")):
        d = json.loads(line)
        m = re.match(r"Pick the (.+?) from the counter", d["task"])
        if m:
            tasks[d["task_index"]] = norm(m.group(1))
    cnt = Counter()
    for line in open(os.path.join(mg_dir, "meta", "episodes.jsonl")):
        e = json.loads(line)
        obj = None
        for t in e.get("tasks", []):
            m = re.match(r"Pick the (.+?) from the counter", t)
            if m:
                obj = norm(m.group(1)); break
        if obj:
            cnt[obj] += 1
    return cnt


def student_stats(path):
    eps = json.load(open(path))["episodes"]
    st = defaultdict(lambda: dict(fail=0, n=0, usum=0.0, uc=0))
    for e in eps:
        c = norm(e["object_category"])
        st[c]["n"] += 1
        st[c]["fail"] += int(not e["success"])
        u = e.get("uncertainty")
        if u is not None:
            st[c]["usum"] += float(u); st[c]["uc"] += 1
    return st


def water_fill(scores, caps, budget):
    """Allocate `budget` demos proportional to score, capped per-category; iterate."""
    alloc = {c: 0 for c in scores}
    remaining = budget
    active = {c for c in scores if caps.get(c, 0) > 0 and scores[c] > 0}
    while remaining > 0.5 and active:
        tot = sum(scores[c] for c in active)
        if tot <= 0:
            break
        any_capped = False
        for c in list(active):
            want = alloc[c] + remaining * scores[c] / tot
            if want >= caps[c]:
                remaining -= (caps[c] - alloc[c]); alloc[c] = caps[c]
                active.discard(c); any_capped = True
        if not any_capped:
            for c in active:
                alloc[c] += remaining * scores[c] / tot
            remaining = 0
    return {c: round(v) for c, v in alloc.items() if round(v) > 0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student", required=True)
    p.add_argument("--mg_dir", required=True)
    p.add_argument("--budget", type=int, default=200)
    p.add_argument("--min_n", type=int, default=5)
    p.add_argument("--top_k", type=int, default=0,
                   help="concentrate the budget on the top-K highest-score categories (0=all)")
    p.add_argument("--no_unc", action="store_true",
                   help="drop the uncertainty term (action-variance is a dud for pi0, AUC 0.595): score = P(fail)")
    args = p.parse_args()

    pool = pool_counts(args.mg_dir)
    st = student_stats(args.student)
    use_unc = (not args.no_unc) and any(v["uc"] > 0 for v in st.values())

    rows = []
    for c, v in st.items():
        if v["n"] < args.min_n:
            continue
        p_fail = wilson(v["fail"], v["n"], bound="lower")
        unc = (v["usum"] / v["uc"]) if (v["uc"] and use_unc) else 1.0
        score = p_fail * unc
        rows.append(dict(cat=c, score=score, p_fail=p_fail, unc=round(unc, 4),
                         n=v["n"], pool=pool.get(c, 0)))
    # only categories present in the pool can be subsampled
    inpool = [r for r in rows if r["pool"] > 0]
    inpool.sort(key=lambda x: -x["score"])
    if args.top_k:
        inpool = inpool[:args.top_k]  # concentrate budget on the hardest categories
    scores = {r["cat"]: r["score"] for r in inpool}
    caps = {r["cat"]: r["pool"] for r in inpool}
    alloc = water_fill(scores, caps, args.budget)

    print(f"use_uncertainty={use_unc} | budget={args.budget} | "
          f"categories scored={len(rows)} in-pool={len(inpool)} allocated={len(alloc)}")
    print(f"\n{'category':22s} {'p_fail':>6s} {'unc':>7s} {'score':>7s} {'pool':>5s} {'TAKE':>5s}")
    for r in sorted(inpool, key=lambda x: -x["score"]):
        take = alloc.get(r["cat"], 0)
        mark = "  <-" if take else ""
        print(f"{r['cat']:22s} {r['p_fail']:>6.2f} {r['unc']:>7.4f} {r['score']:>7.4f} "
              f"{r['pool']:>5d} {take:>5d}{mark}")
    not_in_pool = [r["cat"] for r in rows if r["pool"] == 0]
    if not_in_pool:
        print(f"\nscored but NOT in pool (can't subsample): {not_in_pool}")
    print(f"\ntotal demos allocated: {sum(alloc.values())} / budget {args.budget}")
    json.dump({"budget": args.budget, "use_uncertainty": use_unc, "allocation": alloc},
              open("weakregion/subsample_plan.json", "w"), indent=2)
    print("wrote weakregion/subsample_plan.json")


if __name__ == "__main__":
    main()
