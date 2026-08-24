"""Extended task-2 DP concordance (2026-08-22, after task-1 base failed its bar).

Queue on GPU0: (1) re-eval DP base at n=150 (tightens the shared reference from
±5 to ±3pp); (2) planted-poison cell — corrupted dataset built by
ppccab_prep_planted (PLANT_J=143), the policy-agnostic anchor: shuffled
actions must crash DP or the surrogate is dead; (3) gc0/gc3/gc4/gc5 cells
(rebuilt from ledger demo_ids) to complete all six arms for the arm-level
rho against the race's paired Delta_pi0. All cells: 100 epochs
(learning-curve validated), ckpt_every=50, eval n=150.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, "/data/xinyua11/robocasa")
os.chdir("/data/xinyua11/robocasa")
assert os.environ.get("BANDIT_TASK_PROFILE") == "ppccab"

from bandit_v1 import pull, ledger

DP = "/data/xinyua11/diffusion_policy"
RPY = "/data/xinyua11/conda/envs/robocasa/bin/python"
EVAL_N = 150


def log(*a):
    print(f"[dpext {time.strftime('%H:%M:%S')}]", *a, flush=True)


def sr_of(out_dir):
    try:
        d = json.load(open(f"{out_dir}/PickPlaceCounterToCabinet/eval_log.json"))
        return [v for k, v in d.items() if "success_rate" in k][0]
    except Exception:
        return None


def dp_env():
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="0", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
               WANDB_MODE="offline", WANDB_DIR="/data/xinyua11/wandb",
               TMPDIR="/data/xinyua11/tmp", HF_HOME="/data/xinyua11/.cache/huggingface")
    return env


def evaluate(ck, out):
    r = subprocess.run([RPY, "run_dp_smoketest.py", "-c", ck, "-t", "PickPlaceCounterToCabinet",
                        "-s", "pretrain", "-n", str(EVAL_N), "-e", "8", "-o", out],
                       cwd=DP, env=dp_env(), capture_output=True, text=True, timeout=4*3600)
    if r.returncode != 0:
        log(f"EVAL FAILED rc={r.returncode}: {r.stderr[-400:]}")
    return sr_of(out)


def train_on(dataset_dir, cfg):
    cfgdir = f"{DP}/diffusion_policy/config/task/robocasa"
    src = open(f"{cfgdir}/ppccab_d0.yaml").read()
    open(f"{cfgdir}/{cfg}.yaml", "w").write(
        src.replace("name: ppccab_d0", f"name: {cfg}")
           .replace("- /data/xinyua11/ft_arms/ppccab_d0", f"- {dataset_dir}"))
    r = subprocess.run([RPY, "train.py", "--config-name=train_diffusion_transformer_bs192",
                        f"task=robocasa/{cfg}", "training.num_epochs=100",
                        "training.checkpoint_every=50", "logging.mode=offline"],
                       cwd=DP, env=dp_env(), capture_output=True, text=True, timeout=8*3600)
    if r.returncode != 0:
        log(f"TRAIN FAILED {cfg}: {r.stdout[-300:]}{r.stderr[-300:]}")
        return None
    runs = sorted(glob.glob(f"{DP}/data/outputs/*/*{cfg}*"), key=os.path.getmtime)
    ck = f"{runs[-1]}/checkpoints/latest.ckpt"
    return ck if os.path.exists(ck) else None


def cell_from_ledger(pid):
    dst = f"/data/xinyua11/ft_arms/ppccab_dpconc_{pid}"
    if not os.path.exists(dst):
        p = ledger.read("pulls")
        row = p[(p.pull_id == pid) & (p.status == "smoke")].iloc[-1]
        demo_ids = [int(x) for x in row["demo_ids"]]
        d0 = pull.load_d0_episode_ids()
        aj, which = pull.write_pull_arms_json(f"dpconc_{pid}", d0, demo_ids)
        pull.run_dataset_build(str(aj), which, dst)
    ck = train_on(dst, f"ppccab_dpconc_{pid}")
    if ck:
        sr = evaluate(ck, f"/data/xinyua11/tmp/dp_conc_{pid}_evaln{EVAL_N}")
        log(f"CELL RESULT {pid}: DP SR(n={EVAL_N}) = {sr}")
    shutil.rmtree(dst, ignore_errors=True)


def main():
    # (1) base reference at n=150
    base_out = f"/data/xinyua11/tmp/dp_ppccab_d0_evaln{EVAL_N}"
    if sr_of(base_out) is None:
        runs = sorted(glob.glob(f"{DP}/data/outputs/*/*_ppccab_d0"), key=os.path.getmtime)
        sr = evaluate(f"{runs[-1]}/checkpoints/latest.ckpt", base_out)
        log(f"BASE RESULT (n={EVAL_N}): DP SR = {sr}")

    # (2) planted poison anchor
    planted_dir = str(pull.dataset_dir_for("planted_bad", 143))
    if not os.path.exists(planted_dir):
        r = subprocess.run([RPY, "-u", "gradient_analysis/ppccab/ppccab_prep_planted.py"],
                           env={**os.environ, "PLANT_J": "143"},
                           capture_output=True, text=True, timeout=3600)
        log(f"planted build rc={r.returncode}: {r.stdout[-200:] if r.returncode else 'OK'}")
    if os.path.exists(planted_dir):
        ck = train_on(planted_dir, "ppccab_dpconc_planted143")
        if ck:
            sr = evaluate(ck, f"/data/xinyua11/tmp/dp_conc_planted143_evaln{EVAL_N}")
            log(f"CELL RESULT planted_bad_j143: DP SR(n={EVAL_N}) = {sr}")

    # (3) remaining gc arms
    for pid in ("random_j141",):
        try:
            cell_from_ledger(pid)
        except Exception as e:
            log(f"{pid} FAILED (fail-forward): {e}")
    log("DPEXT DONE")


if __name__ == "__main__":
    main()
