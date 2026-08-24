"""Driver for gradient-cluster arm pulls ("well-defined arms" test, 2026-08-01).

Runs pull.run_pull for arms defined by gradient_analysis/gradarm_assignment.json
(demo-id sets from k-means in whitened LoRA-gradient-sketch space), at the
race's paired seeds (j=3 -> 1003, j=4 -> 1004), full 20k-step recipe, frozen E.

Ledger hygiene: smoke=True -> rows land with status="smoke" (full delta /
per-stratum / demo_ids recorded) so the frozen race's scheduler arithmetic,
T-cap, and any future run_race resume never see them.

Injection is the zero-modification seam: a custom `regions` pd.Series
(episode_index -> arm name) passed to run_pull; the standard eps-gates + FPS
draw then run unchanged, so gradarm draws are mechanically identical to the
race arms' draws. Restart-safe: skips (arm, j) whose ok/smoke row exists.

Env: DRIVER_ARMS="gradarm_a" DRIVER_ROUNDS="3,4" DRIVER_SLOT=a DRIVER_GPU=0
Run in the robocasa env, detached, from /data/xinyua11/robocasa.
"""
import os
import sys
import time

sys.path.insert(0, "/data/xinyua11/robocasa")
os.chdir("/data/xinyua11/robocasa")

import json
import pandas as pd

from bandit_v1 import pull, run_race, ledger, eval_set, pool

ARMS = [a for a in os.environ.get("DRIVER_ARMS", "gradarm_a").split(",") if a]
ROUNDS = [int(x) for x in os.environ.get("DRIVER_ROUNDS", "3,4").split(",")]
SLOT = os.environ.get("DRIVER_SLOT", "a")
GPU = int(os.environ.get("DRIVER_GPU", "0"))
B = 200
ASSIGN = "/data/xinyua11/robocasa/gradient_analysis/gradarm_assignment.json"


def log(*a):
    print(f"[gradarm_driver {time.strftime('%H:%M:%S')}]", *a, flush=True)


def row_exists(pull_id):
    df = ledger.read("pulls")
    if df is None or len(df) == 0:
        return False
    hit = df[(df["pull_id"] == pull_id) & (df["status"].isin(["ok", "smoke"]))]
    return len(hit) > 0


assign = json.load(open(ASSIGN))
pairs = {}
for name in ("gradarm_a", "gradarm_b"):
    for e in assign[name]:
        pairs[int(e)] = name
regions = pd.Series(pairs, dtype=object)
regions.index.name = "episode_index"
log(f"regions built: {regions.value_counts().to_dict()} (k={assign['k_used']}, "
    f"clusters {assign['clusters']})")

b, per_stratum_b, sigma_e_eval = run_race.load_baseline()
pool_df = pool.build_pool_table(write=False)
e_features = eval_set.load_manifest()
eval_fn = run_race._make_eval_fn()

for j in ROUNDS:
    for arm in ARMS:
        pid = pull.pull_id_for(arm, j)
        if row_exists(pid):
            log(f"{pid}: row already in ledger -- skipping")
            continue
        log(f"pull start: arm={arm!r} round={j} slot={SLOT} gpu={GPU} "
            f"(gradient-cluster arm, smoke-status)")
        row = pull.run_pull(arm, j, SLOT, B, eval_fn=eval_fn,
                            pool_df=pool_df, regions=regions,
                            e_features=e_features,
                            baseline=b, baseline_per_stratum=per_stratum_b,
                            gpu=GPU, smoke=True, log=log)
        log(f"pull done: pull_id={row['pull_id']} status={row['status']} "
            f"delta={row.get('delta')}")
        if row["status"] not in ("ok", "smoke"):
            log(f"DRIVER HALT: {pid} status={row['status']} -- human attention needed")
            sys.exit(1)

log(f"GRADARM DRIVER COMPLETE: arms={ARMS} rounds={ROUNDS}")
