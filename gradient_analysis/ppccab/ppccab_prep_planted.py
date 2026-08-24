"""Task-2 planted poison arm (prep_planted.py at ppccab bindings).

Standard draw replicating run_pull's rng (pull_rng_seed(planted_bad, J)) from
the 800 candidates in ppccab/ucb_robot/arms_r3.json, dataset built at
pull.dataset_dir_for (profile-aware), then episodes 400-599 (the B=200 pull
demos) get their per-step action column temporally shuffled (rng 10000+k).
run_pull later finds the dataset present and REUSES it (verified pull.py
behavior), so the corruption survives into training. PLANT_J env (default 140).
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/xinyua11/robocasa")
os.chdir("/data/xinyua11/robocasa")
assert os.environ.get("BANDIT_TASK_PROFILE") == "ppccab"

from bandit_v1 import pull, draw, pool, eval_set

UCB = "/data/xinyua11/robocasa/gradient_analysis/ppccab/ucb_robot"
J = int(os.environ.get("PLANT_J", "140"))
ARM = "planted_bad"
SRC_GLOB = ("/data/xinyua11/robocasa_pkg/datasets/v1.0/pretrain/atomic/"
            "PickPlaceCounterToCabinet/*/mg/demo/*/lerobot/data/chunk-*/")


def log(*a):
    print("[ppccab_planted]", *a, flush=True)


def main():
    arms = json.load(open(f"{UCB}/arms_r3.json"))
    cand = [int(x) for x in arms[ARM]]
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
    aj, _which = pull.write_pull_arms_json(pid, d0, demo_ids)
    dst = str(pull.dataset_dir_for(ARM, J))
    if not os.path.exists(dst):
        pull.run_dataset_build(str(aj), "base+pull", dst)
    log(f"dataset built: {dst}")

    corrupted = []
    for k in range(400, 600):
        fs = glob.glob(f"{dst}/**/episode_{k:06d}.parquet", recursive=True)
        assert fs, f"missing episode {k}"
        f = fs[0]
        df = pd.read_parquet(f)
        acts = np.stack(df["action"].to_numpy())
        erng = np.random.default_rng(10_000 + k)
        df["action"] = list(acts[erng.permutation(len(acts))])
        df.to_parquet(f, index=False)
        corrupted.append(k)
    json.dump({"pull_id": pid, "demo_ids": [int(x) for x in demo_ids],
               "corruption": "temporal shuffle of per-step action column",
               "episodes_corrupted": corrupted, "rng": "default_rng(10000+k)"},
              open(f"{UCB}/planted_corruption_manifest.json", "w"))

    f = glob.glob(f"{dst}/**/episode_000400.parquet", recursive=True)[0]
    a_new = np.stack(pd.read_parquet(f, columns=["action"])["action"].to_numpy())
    src = glob.glob(SRC_GLOB + f"episode_{demo_ids[0]:06d}.parquet")[0]
    a_old = np.stack(pd.read_parquet(src, columns=["action"])["action"].to_numpy())
    same_set = np.allclose(np.sort(a_new, axis=0), np.sort(a_old, axis=0))
    same_order = np.allclose(a_new, a_old)
    log(f"verify ep400: multiset_equal={same_set} order_equal={same_order} (want True/False)")
    assert same_set and not same_order, "corruption verification FAILED"
    log("PPCCAB PLANTED ARM READY")


if __name__ == "__main__":
    main()
