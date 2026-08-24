"""Overnight concordance driver (GPU0, pilot-gated) — 2026-08-21.

Waits for the DP pilot eval, applies the pre-registered gate (base SR >= 0.15),
then runs concordance cells: rebuild archived prescription datasets from the
ledger's demo_ids (race datasets were deleted; demo_ids are the durable
record), train DP on each (epochs chosen from the epoch-100 learning-curve
answer if available), eval 50 rollouts. Priority: gc2 (known Delta_pi0 −6.3,
the confirmed-harmful arm) then gc1 (+0.74, best arm). Fail-forward per cell.
Planted-poison cell deferred to daytime (needs its own build+corrupt step).
"""
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
CELLS = ["gc2_j140", "gc1_j141"]          # pull_ids whose demo_ids we rebuild
PILOT_EVAL = "/data/xinyua11/tmp/dp_ppccab_d0_eval50/eval_log.json"
EP100_EVAL = "/data/xinyua11/tmp/dp_ppccab_ep100_eval50/eval_log.json"
MIN_SR = 0.15


def log(*a):
    print(f"[dpnight {time.strftime('%H:%M:%S')}]", *a, flush=True)


def read_sr(path):
    try:
        d = json.load(open(path))
        for k in ("success_rate", "mean_success", "sr"):
            if k in d:
                return float(d[k])
        # robocasa eval logs sometimes nest per-env results
        vals = [v for v in d.values() if isinstance(v, (int, float))]
        return float(vals[0]) if vals else None
    except Exception:
        return None


def wait_pilot():
    while not os.path.exists(PILOT_EVAL):
        if not any("run_dp_smoketest" in (open(f"/proc/{p}/cmdline").read() if os.path.exists(f"/proc/{p}/cmdline") else "")
                   for p in os.listdir("/proc") if p.isdigit()) and \
           not any("train_diffusion" in (open(f"/proc/{p}/cmdline").read() if os.path.exists(f"/proc/{p}/cmdline") else "")
                   for p in os.listdir("/proc") if p.isdigit()):
            # neither training nor eval running and no result -> eval chain died
            if not os.path.exists(PILOT_EVAL):
                log("pilot eval chain appears dead with no result -- waiting anyway (fail-forward poll)")
        time.sleep(300)
    return read_sr(PILOT_EVAL)


def run_cell(pid, epochs):
    dst = f"/data/xinyua11/ft_arms/ppccab_dpconc_{pid}"
    if not os.path.exists(dst):
        p = ledger.read("pulls")
        row = p[(p.pull_id == pid) & (p.status == "smoke")].iloc[-1]
        demo_ids = [int(x) for x in row["demo_ids"]]
        d0 = pull.load_d0_episode_ids()
        aj, which = pull.write_pull_arms_json(f"dpconc_{pid}", d0, demo_ids)
        pull.run_dataset_build(str(aj), which, dst)
        log(f"{pid}: dataset rebuilt from ledger demo_ids -> {dst}")
    cfgdir = f"{DP}/diffusion_policy/config/task/robocasa"
    cfg = f"ppccab_dpconc_{pid}"
    src_yaml = open(f"{cfgdir}/ppccab_d0.yaml").read()
    open(f"{cfgdir}/{cfg}.yaml", "w").write(
        src_yaml.replace("name: ppccab_d0", f"name: {cfg}")
                .replace("- /data/xinyua11/ft_arms/ppccab_d0", f"- {dst}"))
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="0", MUJOCO_GL="egl", WANDB_MODE="offline",
               WANDB_DIR="/data/xinyua11/wandb", TMPDIR="/data/xinyua11/tmp",
               HF_HOME="/data/xinyua11/.cache/huggingface")
    log(f"{pid}: training DP ({epochs} epochs)")
    r = subprocess.run([RPY, "train.py", "--config-name=train_diffusion_transformer_bs192",
                        f"task=robocasa/{cfg}", f"training.num_epochs={epochs}",
                        "training.checkpoint_every=50", "logging.mode=offline"],
                       cwd=DP, env=env, capture_output=True, text=True, timeout=8*3600)
    if r.returncode != 0:
        log(f"{pid}: TRAIN FAILED: {r.stdout[-300:]}{r.stderr[-300:]}")
        return
    import glob
    runs = sorted(glob.glob(f"{DP}/data/outputs/*/*{cfg}*"), key=os.path.getmtime)
    ck = f"{runs[-1]}/checkpoints/latest.ckpt"
    out = f"/data/xinyua11/tmp/dp_conc_{pid}_eval50"
    env["PYOPENGL_PLATFORM"] = "egl"
    log(f"{pid}: eval 50 rollouts")
    r = subprocess.run([RPY, "run_dp_smoketest.py", "-c", ck,
                        "-t", "PickPlaceCounterToCabinet", "-s", "pretrain",
                        "-n", "50", "-e", "8", "-o", out],
                       cwd=DP, env=env, capture_output=True, text=True, timeout=3*3600)
    sr = read_sr(f"{out}/PickPlaceCounterToCabinet/eval_log.json")
    log(f"CELL RESULT {pid}: DP SR = {sr}")
    shutil.rmtree(dst, ignore_errors=True)   # rebuildable; save disk


def main():
    sr = wait_pilot()
    ep100 = read_sr(EP100_EVAL)
    log(f"PILOT base SR = {sr}; epoch-100 SR = {ep100}")
    if sr is None or sr < MIN_SR:
        log(f"PILOT FAILED the pre-registered bar (SR {sr} < {MIN_SR}) -- no concordance tonight")
        log("DPNIGHT DONE (pilot-fail branch; trainability probe deferred to daytime decision)")
        return
    epochs = 100 if (ep100 is not None and sr is not None and ep100 >= sr - 0.05) else 200
    log(f"gate PASSED -- concordance cells {CELLS} at {epochs} epochs each")
    for pid in CELLS:
        try:
            run_cell(pid, epochs)
        except Exception as e:
            log(f"{pid} FAILED (fail-forward): {e}")
    log("DPNIGHT DONE")


if __name__ == "__main__":
    main()
