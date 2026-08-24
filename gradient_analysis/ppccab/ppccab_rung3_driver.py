"""Task-2 KILL+UNIFORM race driver — rung3_driver.py at ppccab bindings.

Differences from rung3_driver: arms from ppccab/ucb_robot/arms_r3.json; all
paths profile-aware through bandit_v1 (BANDIT_TASK_PROFILE=ppccab asserted);
no shadow probing (task-1's probe demo lists don't exist here); dataset delete
via pull.dataset_dir_for. Env: DRIVER_ARMS, DRIVER_SLOT, DRIVER_GPU, DRIVER_J.
Pull = B=200 fresh draw (pull_rng_seed), 10k steps, smoke status, paired
scoring vs 'random' handled downstream (decision metric = in-round diff).
"""
import json
import os
import shutil
import sys
import time
import traceback

sys.path.insert(0, "/data/xinyua11/robocasa")
os.chdir("/data/xinyua11/robocasa")
assert os.environ.get("BANDIT_TASK_PROFILE") == "ppccab", "set BANDIT_TASK_PROFILE=ppccab"

import pandas as pd

from bandit_v1 import pull, pool, eval_set, ledger, run_race

UCB = "/data/xinyua11/robocasa/gradient_analysis/ppccab/ucb_robot"
ARMS = [a for a in os.environ["DRIVER_ARMS"].split(",") if a]
SLOT = os.environ.get("DRIVER_SLOT", "a")
GPU = int(os.environ.get("DRIVER_GPU", "0"))
J = int(os.environ.get("DRIVER_J", "140"))
B = 200
STEPS = 10000
MIN_FREE_GB = 30


def log(*a):
    print(f"[rung3ppc {time.strftime('%H:%M:%S')}]", *a, flush=True)


def free_gb():
    return shutil.disk_usage("/data").free / 1e9


def row_exists(pid):
    try:
        df = ledger.read("pulls")
    except FileNotFoundError:
        return False
    return df is not None and len(df) and \
        len(df[(df.pull_id == pid) & (df.status.isin(["ok", "smoke"]))]) > 0


def main():
    arms_def = json.load(open(f"{UCB}/arms_r3.json"))
    pool_df = pool.build_pool_table(write=False)
    e_features = eval_set.load_manifest()
    b, per_stratum_b, _ = run_race.load_baseline()
    eval_fn = run_race._make_eval_fn()
    log(f"baseline b={b:.4f}, arms={ARMS}, j={J}, slot={SLOT}, gpu={GPU}")

    for arm in ARMS:
        pid = pull.pull_id_for(arm, J)
        if row_exists(pid):
            log(f"{pid}: already done -- skip")
            continue
        if free_gb() < MIN_FREE_GB:
            log(f"HARD STOP: disk free {free_gb():.0f}GB < {MIN_FREE_GB}GB")
            open(f"{UCB}/HALT_DISK.txt", "w").write(
                f"halted before {pid} at {time.ctime()}, free={free_gb():.0f}GB\n")
            sys.exit(2)
        if arm == "random":
            # task-1 convention (ucb_robot_setup): random = EMPTY region series,
            # which run_pull accepts and draw resolves to the uniform-pool path.
            # regions=None is rejected for non-null arms (bit us 2026-08-19,
            # cost round 1 its control until the makeup pull).
            regions = pd.Series(dtype=object)
            regions.index.name = "episode_index"
        else:
            regions = pd.Series({int(e): arm for e in arms_def[arm]}, dtype=object)
            regions.index.name = "episode_index"
        log(f"pull start: {pid} slot={SLOT} gpu={GPU} steps={STEPS} "
            f"candidates={'pool' if regions is None else len(regions)}")
        try:
            row = pull.run_pull(arm, J, SLOT, B, eval_fn=eval_fn,
                                pool_df=pool_df, regions=regions,
                                e_features=e_features, baseline=b,
                                baseline_per_stratum=per_stratum_b,
                                num_train_steps=STEPS, gpu=GPU, smoke=True, log=log)
            log(f"pull done: {pid} status={row['status']} delta={row.get('delta')}")
        except Exception:
            log(f"pull FAILED: {pid}\n{traceback.format_exc()[-800:]}")
            continue
        ds = pull.dataset_dir_for(arm, J)
        if ds.is_dir():
            shutil.rmtree(str(ds), ignore_errors=True)
            log(f"disk: removed dataset {ds} (rebuildable from demo_ids)")
        ck = f"/data/xinyua11/openpi/checkpoints/{pull.train_config_name_for_slot(SLOT)}/{pid}"
        if os.path.isdir(f"{ck}/5000"):
            shutil.rmtree(f"{ck}/5000", ignore_errors=True)
        shadow_probe(pid, ck)
    log("RUNG3PPC QUEUE COMPLETE")


def shadow_probe(pid, ck_root, step=9999):
    """phi collection for the trust-gated index (2026-08-18 build): probe the
    kept final checkpoint on this driver's own GPU (between pulls, so no
    contention), save ucb_robot/shadow/<pid>_probe9999.json, then delete the
    checkpoint (reproducible: seed-deterministic training). Non-fatal."""
    import subprocess
    if not os.path.isdir(f"{ck_root}/{step}"):
        log(f"shadow: {pid}/{step} missing -- skip")
        return
    os.makedirs(f"{UCB}/shadow", exist_ok=True)
    out = f"{UCB}/shadow/{pid}_probe{step}.json"
    env = {k: v for k, v in os.environ.items() if k != "PYOPENGL_PLATFORM"}
    env.update({"CUDA_VISIBLE_DEVICES": str(GPU), "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.3", "OMP_NUM_THREADS": "4"})
    try:
        r = subprocess.run(["/data/xinyua11/conda/envs/openpi/bin/python", "-u",
                            "gradient_analysis/ppccab/ppccab_probe_ckpt.py",
                            "--ckpt_root", ck_root, "--step", str(step), "--out", out],
                           env=env, capture_output=True, text=True, timeout=2400)
        if r.returncode == 0:
            log(f"shadow: {pid}@{step} -> {json.load(open(out))}")
            shutil.rmtree(f"{ck_root}/{step}", ignore_errors=True)
            log(f"disk: removed {pid}/{step} ckpt after probing")
        else:
            log(f"shadow: probe {pid}@{step} FAILED (ckpt kept): {r.stderr[-300:]}")
    except Exception as ex:
        log(f"shadow: probe {pid}@{step} error (non-fatal): {ex}")


if __name__ == "__main__":
    main()
