"""Clean object-factor test — batch construction (owner spec, 2026-08-12).

Two arms defined ONLY by object category, everything else exact-matched:
  hard_obj : blender_jug, tupperware, cream_cheese_stick, cheese_grater,
             juice, ketchup            (base-policy success 6-17%, diagnosis)
  easy_obj : banana, carrot, mushroom, ladle, corn, salt_and_pepper_shaker
             (base success 73-96%)
Criterion pre-registered from the 2,400-rollout diagnosis only.

Construction: apply the standard eval/D0 contamination gate
(draw._conflict_mask, eps=config.EPS_XY) -> exact-match the two arms
cell-by-cell on (layout, side, 4x4 position grid) -> FIXED batches of equal
size B (target 150; = min(matched, 150)), plus a fixed random control of the
same B from the gated pool. Batches are frozen: every round retrains the same
sets (rounds differ only by seed; draw noise eliminated by design).
Output: gradient_analysis/objtest/batches.json + three built datasets.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/xinyua11/robocasa")
os.chdir("/data/xinyua11/robocasa")

from bandit_v1 import pool, draw, eval_set, config, pull

import os as _o
OUT = _o.environ.get("OBJTEST_OUT", "/data/xinyua11/robocasa/gradient_analysis/objtest")
HARD = _o.environ.get("OBJ_HARD", "blender_jug,tupperware,cream_cheese_stick,cheese_grater,juice,ketchup").split(",")
EASY = _o.environ.get("OBJ_EASY", "banana,carrot,mushroom,ladle,corn,salt_and_pepper_shaker").split(",")
B_TARGET = 150
SEED = int(_o.environ.get("OBJ_SEED", "20260812"))


def main():
    os.makedirs(OUT, exist_ok=True)
    W = pool.build_pool_table(write=False)
    d0_df = W[W.in_d0]
    W = W[~W.in_d0].copy()
    ef = eval_set.load_manifest()
    conf_e = draw._conflict_mask(W, ef, config.EPS_XY)
    conf_d = draw._conflict_mask(W, d0_df, config.EPS_XY)
    Wg = W[~conf_e & ~conf_d].copy()
    print(f"[objtest] pool {len(W)} -> gated {len(Wg)}", flush=True)

    Wg["xb"] = pd.cut(Wg.x_rel, 4, labels=False)
    Wg["yb"] = pd.cut(Wg.y_rel, 4, labels=False)
    A = Wg[Wg.category.isin(HARD)]
    Bd = Wg[Wg.category.isin(EASY)]
    rng = np.random.default_rng(SEED)
    keys = ["layout", "side", "xb", "yb"]
    ga, gb = A.groupby(keys), Bd.groupby(keys)
    cells = sorted(set(ga.groups) & set(gb.groups))
    sel_a, sel_b = [], []
    for c in cells:
        ia, ib = ga.groups[c], gb.groups[c]
        k = min(len(ia), len(ib))
        sel_a += list(rng.choice(A.loc[ia].episode_index, k, replace=False))
        sel_b += list(rng.choice(Bd.loc[ib].episode_index, k, replace=False))
    M = len(sel_a)
    Bsz = min(B_TARGET, M)
    if M > Bsz:  # proportional thinning keeps the matched distributions equal
        keep = sorted(rng.choice(M, Bsz, replace=False))
        sel_a = [sel_a[i] for i in keep]
        sel_b = [sel_b[i] for i in keep]
    ctrl = sorted(int(x) for x in rng.choice(Wg.episode_index, Bsz, replace=False))
    batches = {"hard_obj": sorted(int(x) for x in sel_a),
               "easy_obj": sorted(int(x) for x in sel_b),
               "random_ctrl": ctrl}
    json.dump({"batches": batches, "B": Bsz, "matched_cells": len(cells),
               "matched_total": M, "seed": SEED,
               "criterion": "diagnosis per-category base success extremes"},
              open(f"{OUT}/batches.json", "w"))
    for name, ids in batches.items():
        sub = W[W.episode_index.isin(ids)]
        print(f"[objtest] {name}: n={len(ids)} mean_h={sub.h.mean():.3f} "
              f"layouts={sorted(sub.layout.unique())}", flush=True)

    d0 = pull.load_d0_episode_ids()
    for name, ids in batches.items():
        pull.assemble_episode_ids(d0, ids)
        aj = f"{OUT}/{name}_arms.json"
        pull.write_pull_arms_json(f"{_o.environ.get('OBJ_TAG','objtest')}_{name}", d0, ids, path=aj)
        pfx = _o.environ.get("OBJ_DST_PREFIX", "ppc2sink_bandit_")   # ppccab_bandit_ for track 2
        dst = f"/data/xinyua11/ft_arms/{pfx}{_o.environ.get('OBJ_TAG','objtest')}_{name}"
        if not os.path.exists(dst):
            pull.run_dataset_build(aj, "base+pull", dst)
        print(f"[objtest] dataset ready: {dst}", flush=True)
    print("[objtest] BUILD COMPLETE", flush=True)


if __name__ == "__main__":
    main()
