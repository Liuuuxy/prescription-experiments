"""Core-algorithm allocation: join per-scene teacher (pi0) + student (DP) success
on the SAME gym.make seeds, aggregate per object category, and run
acquisition.allocate -> which categories to collect demos from (the core-algo arm).

Teacher and student are evaluated on identical scenes (seed i == pi0 episode i ==
DP eval seed i), so per-category teacher_n == student_n (paired).

  python policy_analysis/run_acquisition.py \
    --teacher weakregion/pi0_PickPlaceCounterToSink_n150/weakregion.json \
    --student weakregion/dp_seed0_ep400_pool150/dp_eval.json \
    --budget 200 --min_n 3
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import acquisition  # policy_analysis/acquisition.py


def load_teacher(path):
    eps = json.load(open(path))["episodes"]
    return {e["episode"]: e for e in eps}  # episode index == gym.make seed


def load_student(path):
    res = json.load(open(path))["results"]
    return {r["seed"]: r for r in res}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True)
    p.add_argument("--student", required=True)
    p.add_argument("--budget", type=int, default=200, help="total generation attempts to allocate")
    p.add_argument("--min_n", type=int, default=3, help="drop categories with fewer than this many scenes")
    p.add_argument("--uncertainty", default=None, help="optional json: {category: uncertainty}")
    args = p.parse_args()

    teacher = load_teacher(args.teacher)
    student = load_student(args.student)
    seeds = sorted(set(teacher) & set(student))
    print(f"paired scenes: {len(seeds)} (teacher {len(teacher)}, student {len(student)})")

    cat = defaultdict(lambda: dict(tk=0, tn=0, sk=0, sn=0, usum=0.0, uc=0))
    for s in seeds:
        c = teacher[s]["object_category"]
        cat[c]["tn"] += 1
        cat[c]["tk"] += int(bool(teacher[s]["success"]))
        cat[c]["sn"] += 1
        cat[c]["sk"] += int(bool(student[s]["success"]))
        u = student[s].get("uncertainty")
        if u is not None:
            cat[c]["usum"] += float(u)
            cat[c]["uc"] += 1

    # per-category mean uncertainty (DP action-variance); scale cancels in allocate
    use_unc = any(v["uc"] > 0 for v in cat.values())
    regions = [acquisition.Region(
                   name=c, teacher_k=v["tk"], teacher_n=v["tn"],
                   student_k=v["sk"], student_n=v["sn"],
                   uncertainty=(v["usum"] / v["uc"] if v["uc"] else 1.0))
               for c, v in cat.items() if v["tn"] >= args.min_n]
    regions.sort(key=lambda r: r.name)
    print(f"regions (categories) with n>={args.min_n}: {len(regions)} "
          f"(of {len(cat)} total) | use_uncertainty={use_unc}\n")

    plan = acquisition.allocate(regions, args.budget, use_uncertainty=use_unc)
    print(f"{'category':20s} {'teacher':>9s} {'student':>9s} {'score':>7s} {'attempts':>8s}")
    for row in plan:
        c = row["region"]
        v = cat[c]
        print(f"{c:20s} {v['tk']:>3d}/{v['tn']:<5d} {v['sk']:>3d}/{v['sn']:<5d} "
              f"{row['score']:>7.4f} {row.get('attempts', row.get('alloc', 0)):>8}")

    out = {"budget": args.budget, "min_n": args.min_n, "use_uncertainty": use_unc,
           "n_paired": len(seeds), "plan": plan}
    json.dump(out, open("weakregion/acquisition_plan.json", "w"), indent=2)
    print("\nwrote weakregion/acquisition_plan.json")


if __name__ == "__main__":
    main()
