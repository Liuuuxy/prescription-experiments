"""Clean object-test driver — fixed batches, paired seeds (2026-08-12).

QUEUE env = comma list of arm:round tokens, e.g. "hard_obj:110,random_ctrl:110".
Each item: train the arm's FIXED dataset (built by objtest_build.py) at seed
1000+round for 10k steps, eval on frozen E, save result json, DELETE the
checkpoint (fixed batches + proven seed-determinism make it reproducible).
Results: gradient_analysis/objtest/<exp>_result.json. Race ledger untouched
(eval episodes land under distinct policy_ids only).
"""
import json
import os
import shutil
import sys
import time
import traceback

sys.path.insert(0, "/data/xinyua11/robocasa")
os.chdir("/data/xinyua11/robocasa")

from bandit_v1 import pull, eval_set

OUT = os.environ.get("OBJTEST_OUT", "/data/xinyua11/robocasa/gradient_analysis/objtest")
QUEUE = [t for t in os.environ["QUEUE"].split(",") if t]
SLOT = os.environ.get("SLOT", "a")
GPU = int(os.environ.get("GPU", "0"))
PORT = 8130 if SLOT == "a" else 8131
CFG = pull.train_config_name_for_slot(SLOT)          # profile-aware (BANDIT_TASK_PROFILE)
BASELINE = float(os.environ.get("OBJ_BASELINE", "0.5133"))
DST_PREFIX = os.environ.get("OBJ_DST_PREFIX", "ppc2sink_bandit_")
SLOT_PREFIX = os.environ.get("OBJ_SLOT_PREFIX", "/data/xinyua11/ft_arms/ppc2sink_bandit_slot_")


def log(*a):
    print(f"[objtest {time.strftime('%H:%M:%S')}]", *a, flush=True)


def one(arm, j):
    TAG = os.environ.get("OBJ_TAG", "objtest")
    exp = f"{TAG}_{arm}_j{j}"
    res_path = f"{OUT}/{exp}_result.json"
    if os.path.exists(res_path):
        log(f"{exp}: done already -- skip")
        return
    pull.symlink_slot(f"/data/xinyua11/ft_arms/{DST_PREFIX}{TAG}_{arm}",
                      f"{SLOT_PREFIX}{SLOT}")
    cmd = pull.train_cmd(CFG, exp, 1000 + j, num_train_steps=10000)
    ckpt = pull.ckpt_final_dir(CFG, exp, 10000)
    log(f"pull start: {exp} slot={SLOT} gpu={GPU}")
    proc = pull.launch_training(cmd, GPU, f"{OUT}/{exp}_train.log")
    ok = pull.wait_for_checkpoint(proc, ckpt, num_train_steps=10000, log=log)
    if not ok:
        log(f"{exp}: TRAINING FAILED")
        return
    _, sp = pull.launch_server(CFG, str(ckpt), PORT, GPU, f"{OUT}/{exp}_serve.log")
    try:
        up = pull.wait_for_port("127.0.0.1", PORT, proc=sp, log=log)
        if not up:
            log(f"{exp}: SERVER FAILED")
            return
        r = eval_set.eval_checkpoint(PORT, exp, "objtest", exp, workers=4,
                                     resume=True, log=log)
    finally:
        sp.terminate()
    delta = (r["mean"] - BASELINE) * 100
    json.dump({"exp": exp, "arm": arm, "j": j, "sr": r["mean"],
               "delta_pp": delta, "per_stratum": r["per_stratum_mean"]},
              open(res_path, "w"))
    log(f"pull done: {exp} delta={delta:+.2f}pp per_stratum="
        f"{ {k: round(v,3) for k,v in r['per_stratum_mean'].items()} }")
    shutil.rmtree(str(ckpt.parent), ignore_errors=True)
    log(f"disk: removed {exp} checkpoints (reproducible: fixed batch + seed)")


def main():
    for tok in QUEUE:
        arm, j = tok.split(":")
        try:
            one(arm, int(j))
        except Exception:
            log(f"{tok} FAILED\n{traceback.format_exc()[-600:]}")
    log("OBJTEST QUEUE COMPLETE")


if __name__ == "__main__":
    main()
