"""Turn the sandbox's predictor-pick vs bandit-pick allocations into REAL, runnable
pi0 fine-tune arms: emit build_arms.py-style category plans + the openpi config
snippets + the exact build/train/eval commands.

Pool-selected (not generated), so the arms are directly comparable to core/random and
avoid the GR00T action-space landmine. Category<->group mapping is grounded in the real
difficulty data (weak-region reports + value postmortem); availability is read from the
9,885-episode pool index.
"""
from __future__ import annotations

import json
import os
import re
import collections

import numpy as np

from .calibrate import robocasa_env, anchor_allocations

# The real MimicGen pool (9,885 episodes) that build_arms.py --mg_dir points at.
MG_DIR = ("/data/xinyua11/robocasa_pkg/datasets/v1.0/pretrain/atomic/"
          "PickPlaceCounterToSink/20250819/mg/demo/2025-08-20-22-32-27/lerobot")

# group -> concrete pool categories (all verified present in the pool).
GROUP_TO_CATEGORIES = {
    # the ~0-11% tail (jar/juice/cheese_grater) + tall/awkward: near-uncollectable, no headroom
    "unfixable_hard":   ["juice", "jar", "cheese_grater", "wine", "water_bottle"],
    # mid-difficulty with real, collectable headroom (targeted_rebalanced categories)
    "hard_collectable": ["pitcher", "reamer", "colander", "bottled_water", "jug", "yogurt", "spray"],
    # the value arm's elongated/handled grasp cluster: collectable but retention-toxic
    "retention_toxic":  ["tongs", "dish_brush", "mug", "rolling_pin", "straw", "whisk", "wooden_spoon", "ladle"],
    # already-solved common objects: untargeted spread
    "easy_majority":    ["cucumber", "corn", "carrot", "broccoli", "pear", "apple", "banana", "lemon"],
}


def distribute(count, categories, availability):
    """Split `count` demos across `categories` proportional to availability, integer,
    capped per-category, summing to min(count, total available)."""
    count = int(round(count))
    avail = {c: int(availability.get(c, 0)) for c in categories}
    total = sum(avail.values())
    if count <= 0 or total == 0:
        return {}
    alloc = {c: 0 for c in categories}
    target = min(count, total)
    exact = {c: target * avail[c] / total for c in categories}
    for c in categories:
        alloc[c] = min(int(exact[c]), avail[c])
    while sum(alloc.values()) < target:
        cand = [c for c in categories if alloc[c] < avail[c]]
        if not cand:
            break
        best = max(cand, key=lambda c: exact[c] - alloc[c])
        alloc[best] += 1
    return {c: k for c, k in alloc.items() if k > 0}


def build_category_plan(region_names, group_counts, availability):
    """Combine per-group demo counts into a single {category: k} plan."""
    plan = {}
    for name, cnt in zip(region_names, group_counts):
        plan.update(distribute(cnt, GROUP_TO_CATEGORIES[name], availability))
    return plan


def pool_availability(mg_dir=MG_DIR):
    """Per-category episode counts read from the pool's meta/episodes.jsonl."""
    path = os.path.join(mg_dir, "meta", "episodes.jsonl")
    cnt = collections.Counter()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = (json.loads(line).get("tasks") or [""])[0]
            m = re.search(r"Pick the (.+?) from the counter", t)
            if m:
                cnt[m.group(1).strip().lower().replace(" ", "_")] += 1
    return dict(cnt)


def make_plans(demo_budget=200, measure_budget=20_000, seed=0):
    env = robocasa_env()
    picks = anchor_allocations(env, demo_budget, measure_budget, np.random.default_rng(seed))
    avail = pool_availability()
    plans = {}
    for arm, key in (("predanchor", "predictor_pick"), ("bandanchor", "bandit_pick")):
        cat_plan = build_category_plan(picks["region_names"], picks[key], avail)
        plans[arm] = {
            "budget": demo_budget,
            "use_uncertainty": False,
            "source": f"prescription_compare {key} on robocasa twin (fidelity {env.feature_fidelity:.3f})",
            "group_allocation": dict(zip(picks["region_names"], [round(float(x), 1) for x in picks[key]])),
            "allocation": cat_plan,
        }
    return plans
