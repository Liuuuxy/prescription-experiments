"""Build the planted corrupted-demo arm (overnight 2026-08-06).

planted_bad = 800 random pool candidates (rng 777) -> standard draw of 200 at
j=101 (replicating run_pull's exact rng so the ledger row's demo_ids match the
built dataset) -> dataset built -> the 200 PULL episodes (files 400-599,
ordering verified byte-exact in preflight) get their per-step 'action' column
TEMPORALLY SHUFFLED (deterministic per-episode rng). Marginal action
distribution preserved (norm-stats safe); action-observation correspondence
destroyed -> harmful by construction. Base episodes 0-399 untouched.
Writes: arms_r3.json (arms.json + planted_bad, NEW file), corruption manifest,
and verifies one corrupted episode. Original pool files are never touched.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/xinyua11/robocasa")
os.chdir("/data/xinyua11/robocasa")

from bandit_v1 import pull, draw, pool, eval_set

UCB = "/data/xinyua11/robocasa/gradient_analysis/ucb_robot"
import os as _os
J = int(_os.environ.get("PLANT_J", "101"))
ARM = "planted_bad"


def log(*a):
    print("[prep_planted]", *a, flush=True)


def main():
    arms = json.load(open(f"{UCB}/arms.json"))
    lists_pool = sorted(int(e) for ids in [arms["mid_band"], arms["easy_band"],
                        arms["tall_vessel_grasp_fail"]] for e in ids)
    rng777 = np.random.default_rng(777)
    cand = sorted(int(x) for x in rng777.choice(lists_pool, 800, replace=False))
    arms_r3 = dict(arms)
    arms_r3[ARM] = cand
    json.dump(arms_r3, open(f"{UCB}/arms_r3.json", "w"))
    log(f"arms_r3.json written ({len(arms_r3)} arms; planted candidates={len(cand)})")

    pid = pull.pull_id_for(ARM, J)
    pool_df = pool.build_pool_table(write=False)
    e_features = eval_set.load_manifest()
    regions = pd.Series({e: ARM for e in cand}, dtype=object)
    regions.index.name = "episode_index"
    rng = np.random.default_rng(pull.pull_rng_seed(ARM, J))
    demo_ids = draw.pull_demos(ARM, 200, rng, pool_df=pool_df, regions=regions,
                               e_features=e_features)
    d0 = pull.load_d0_episode_ids()
    pull.assemble_episode_ids(d0, demo_ids)
    aj = pull.write_pull_arms_json(pid, d0, demo_ids)
    dst = f"/data/xinyua11/ft_arms/ppc2sink_bandit_{ARM}_j{J}"
    if not os.path.exists(dst):
        pull.run_dataset_build(f"/data/xinyua11/robocasa/bandit_v1/ledger/pull_arms/{pid}.json",
                               "base+pull", dst)
    log(f"dataset built: {dst}")

    corrupted = []
    for k in range(400, 600):
        fs = glob.glob(f"{dst}/**/episode_{k:06d}.parquet", recursive=True)
        assert fs, f"missing episode {k}"
        f = fs[0]
        df = pd.read_parquet(f)
        acts = np.stack(df["action"].to_numpy())
        erng = np.random.default_rng(10_000 + k)
        perm = erng.permutation(len(acts))
        df["action"] = list(acts[perm])
        df.to_parquet(f, index=False)
        corrupted.append(k)
    json.dump({"pull_id": pid, "demo_ids": [int(x) for x in demo_ids],
               "corruption": "temporal shuffle of per-step action column",
               "episodes_corrupted": corrupted, "rng": "default_rng(10000+k)"},
              open(f"{UCB}/planted_corruption_manifest.json", "w"))

    # verify: corrupted file's action multiset == original, order != original
    f = glob.glob(f"{dst}/**/episode_000400.parquet", recursive=True)[0]
    a_new = np.stack(pd.read_parquet(f, columns=["action"])["action"].to_numpy())
    src = glob.glob("/data/xinyua11/robocasa_pkg/datasets/v1.0/pretrain/atomic/"
                    f"PickPlaceCounterToSink/*/mg/demo/*/lerobot/data/chunk-*/"
                    f"episode_{demo_ids[0]:06d}.parquet")[0]
    a_old = np.stack(pd.read_parquet(src, columns=["action"])["action"].to_numpy())
    same_set = np.allclose(np.sort(a_new, axis=0), np.sort(a_old, axis=0))
    same_order = np.allclose(a_new, a_old)
    log(f"verify ep400: multiset_equal={same_set} order_equal={same_order} "
        f"(want True/False)")
    assert same_set and not same_order, "corruption verification FAILED"
    log("PLANTED ARM READY")


if __name__ == "__main__":
    main()
