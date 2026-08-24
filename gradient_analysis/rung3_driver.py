"""Rung-3 KILL+UNIFORM driver — paired round pulls (overnight 2026-08-06).

Pull = run_pull with num_train_steps=10000 (ablation-validated recipe),
smoke status (race ledger stays decision-inert), B=200, arms from
ucb_robot/arms_r3.json (12 arms + planted_bad). j=101 -> shared round seed
1101. Decision metric = paired in-round diff vs 'random' (baseline b recorded
but not decisive; recipe changed 20k->10k).

Shadow logging per pull: probe_ckpt.py on the 5000 and 9999 checkpoints ->
ucb_robot/shadow/. Disk protocol (started at 105GB free): delete the pull's
dataset dir after training+eval; delete the 5000 ckpt after probing; keep
9999. HARD STOP if free disk < 30GB. Fail-forward on per-arm errors.

Env: DRIVER_ARMS, DRIVER_SLOT, DRIVER_GPU, DRIVER_J (default 101).
"""
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

sys.path.insert(0, "/data/xinyua11/robocasa")
os.chdir("/data/xinyua11/robocasa")

import pandas as pd

from bandit_v1 import pull, pool, eval_set, ledger, run_race

UCB = "/data/xinyua11/robocasa/gradient_analysis/ucb_robot"
ARMS = [a for a in os.environ["DRIVER_ARMS"].split(",") if a]
SLOT = os.environ.get("DRIVER_SLOT", "a")
GPU = int(os.environ.get("DRIVER_GPU", "0"))
J = int(os.environ.get("DRIVER_J", "101"))
B = 200
STEPS = 10000
OPENPI_PY = "/data/xinyua11/conda/envs/openpi/bin/python"
MIN_FREE_GB = 30


def log(*a):
    print(f"[rung3 {time.strftime('%H:%M:%S')}]", *a, flush=True)


def free_gb():
    return shutil.disk_usage("/data").free / 1e9


def row_exists(pid):
    df = ledger.read("pulls")
    return df is not None and len(df) and \
        len(df[(df.pull_id == pid) & (df.status.isin(["ok", "smoke"]))]) > 0


def shadow_probe(pid, cfg_name):
    os.makedirs(f"{UCB}/shadow", exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "PYOPENGL_PLATFORM"}
    env.update({"CUDA_VISIBLE_DEVICES": str(GPU),
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.3",
                "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
    root = f"/data/xinyua11/openpi/checkpoints/{cfg_name}/{pid}"
    for step in (5000, 9999):
        if not os.path.isdir(f"{root}/{step}"):
            log(f"shadow: {pid}/{step} missing -- skip")
            continue
        out = f"{UCB}/shadow/{pid}_probe{step}.json"
        r = subprocess.run([OPENPI_PY, "-u", "gradient_analysis/probe_ckpt.py",
                            "--ckpt_root", root, "--step", str(step), "--out", out],
                           env=env, capture_output=True, text=True, timeout=1800)
        if r.returncode == 0:
            log(f"shadow: {pid}@{step} -> {json.load(open(out))}")
            if step == 5000:
                shutil.rmtree(f"{root}/{step}", ignore_errors=True)
                log(f"disk: removed {pid}/5000 ckpt after probing")
        else:
            log(f"shadow: probe {pid}@{step} FAILED: {r.stderr[-300:]}")


def main():
    arms_def = json.load(open(f"{UCB}/arms_r3.json"))
    pool_df = pool.build_pool_table(write=False)
    e_features = eval_set.load_manifest()
    b, per_stratum_b, _ = run_race.load_baseline()
    eval_fn = run_race._make_eval_fn()
    cfg_name = pull.train_config_name_for_slot(SLOT)

    for arm in ARMS:
        pid = pull.pull_id_for(arm, J)
        if row_exists(pid):
            log(f"{pid}: already done -- skip")
            continue
        if free_gb() < MIN_FREE_GB:
            log(f"HARD STOP: disk free {free_gb():.0f}GB < {MIN_FREE_GB}GB -- halting queue")
            open(f"{UCB}/HALT_DISK.txt", "w").write(
                f"halted before {pid} at {time.ctime()}, free={free_gb():.0f}GB\n")
            sys.exit(2)
        regions = pd.Series({int(e): arm for e in arms_def.get(arm, [])}, dtype=object)
        regions.index.name = "episode_index"
        log(f"pull start: {pid} slot={SLOT} gpu={GPU} steps={STEPS} candidates={len(regions)}")
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
        try:
            shadow_probe(pid, cfg_name)
        except Exception:
            log(f"shadow FAILED for {pid} (non-fatal)")
        ds = f"/data/xinyua11/ft_arms/ppc2sink_bandit_{arm}_j{J}"
        if os.path.isdir(ds):
            shutil.rmtree(ds, ignore_errors=True)
            log(f"disk: removed dataset {ds} (rebuildable from demo_ids)")
    log("RUNG3 QUEUE COMPLETE")


if __name__ == "__main__":
    main()
