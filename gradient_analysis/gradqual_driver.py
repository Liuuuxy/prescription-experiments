"""Driver for the GRADIENT-QUALITY arm test (causal validation of the Q6 filter, 2026-08-13).

Q6 showed demo execution-quality IS gradient-visible at pi_0 before any training
(norm+direction AUC 0.778, vs 0.577 for the failure region). This driver runs the
causal test: two arms drawn from the SAME pool at the SAME paired seed, differing
only by the gradient-quality score
    score(z) = z-score[cos(g(z), g_style_hi - g_style_lo)] - z-score[||g(z)||]
  gradqual_hi = top 200,  gradqual_lo = bottom 200   (gradient_analysis/gradqual_assignment.json)
All style-arm demos are EXCLUDED from the candidate set, so this is an independent
selection, not a re-run of the style race.

Read: hi - lo at a paired seed is the same contrast that gave the style race its only
consistent-sign effect (+4.9pp mean, 3/3 seeds). If the gradient score reproduces it,
the free filter is causally validated and transfers to any new data source.

Same injection seam as gradarm_driver.py: a custom `regions` Series -> run_pull, with
smoke=True so rows never touch the frozen race's scheduler arithmetic.

Env: DRIVER_ARMS=gradqual_hi DRIVER_ROUNDS=3 DRIVER_SLOT=a DRIVER_GPU=0
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

ARMS = [a for a in os.environ.get("DRIVER_ARMS", "gradqual_hi").split(",") if a]
ROUNDS = [int(x) for x in os.environ.get("DRIVER_ROUNDS", "3").split(",")]
SLOT = os.environ.get("DRIVER_SLOT", "a")
GPU = int(os.environ.get("DRIVER_GPU", "0"))
B = 200
ASSIGN = "/data/xinyua11/robocasa/gradient_analysis/gradqual_assignment.json"


def log(*a):
    print(f"[gradqual_driver {time.strftime('%H:%M:%S')}]", *a, flush=True)


def row_exists(pull_id):
    df = ledger.read("pulls")
    if df is None or len(df) == 0:
        return False
    return len(df[(df["pull_id"] == pull_id) & (df["status"].isin(["ok", "smoke"]))]) > 0


assign = json.load(open(ASSIGN))
pairs = {}
for name in ("gradqual_hi", "gradqual_lo"):
    for e in assign[name]:
        pairs[int(e)] = name
regions = pd.Series(pairs, dtype=object)
regions.index.name = "episode_index"
log(f"regions: {regions.value_counts().to_dict()} | method: {assign['method']}")

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
        log(f"pull start: arm={arm!r} round={j} slot={SLOT} gpu={GPU} (gradient-quality arm, smoke)")
        row = pull.run_pull(arm, j, SLOT, B, eval_fn=eval_fn,
                            pool_df=pool_df, regions=regions,
                            e_features=e_features,
                            baseline=b, baseline_per_stratum=per_stratum_b,
                            gpu=GPU, smoke=True, log=log)
        log(f"pull done: pull_id={row['pull_id']} status={row['status']} delta={row.get('delta')}")
        if row["status"] not in ("ok", "smoke"):
            log(f"DRIVER HALT: {pid} status={row['status']} -- human attention needed")
            sys.exit(1)

log(f"GRADQUAL DRIVER COMPLETE: arms={ARMS} rounds={ROUNDS}")
