"""Determinism test (owner request, 2026-08-10): rerun random_j106 with
IDENTICAL everything — same 200 demo ids (from the ledger row, not a redraw),
same training seed 1106, same 10k-step recipe, same slot-b config, same GPU1 —
under exp name 'replica_random_j106'. Then compare against the original at
three levels:
  1. weights  : probe_ckpt losses (deterministic fixed-draw computation) vs the
                original's saved shadow probe -> any weight drift shows here
  2. outcome  : eval delta vs the original's -3.56pp
  3. episodes : per-(start,repeat) success agreement, original vs replica
Writes gradient_analysis/replica_test_result.json. Nothing in the race ledger.
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, "/data/xinyua11/robocasa")
os.chdir("/data/xinyua11/robocasa")

import numpy as np
import pandas as pd

from bandit_v1 import pull, ledger, eval_set

EXP = "replica_random_j106"
CFG = pull.train_config_name_for_slot("b")
GPU = 1
SEED = 1106
STEPS = 10000
UCB = "/data/xinyua11/robocasa/gradient_analysis/ucb_robot"
OPENPI_PY = "/data/xinyua11/conda/envs/openpi/bin/python"


def log(*a):
    print(f"[replica {time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    p = ledger.read("pulls")
    orig = p[(p.pull_id == "random_j106") & (p.status == "smoke")].iloc[-1]
    demo_ids = [int(x) for x in orig["demo_ids"]]
    log(f"original delta={orig.delta*100:+.2f}pp, {len(demo_ids)} demo ids recovered")

    d0 = pull.load_d0_episode_ids()
    aj = pull.write_pull_arms_json(EXP, d0, demo_ids, path=f"{UCB}/{EXP}_arms.json")
    dst = f"/data/xinyua11/ft_arms/ppc2sink_bandit_{EXP}"
    if not os.path.exists(dst):
        pull.run_dataset_build(f"{UCB}/{EXP}_arms.json", "base+pull", dst)
    pull.symlink_slot(dst, "/data/xinyua11/ft_arms/ppc2sink_bandit_slot_b")

    cmd = pull.train_cmd(CFG, EXP, SEED, num_train_steps=STEPS)
    ckpt = pull.ckpt_final_dir(CFG, EXP, STEPS)
    proc = pull.launch_training(cmd, GPU, f"{UCB}/{EXP}_train.log")
    ok = pull.wait_for_checkpoint(proc, ckpt, num_train_steps=STEPS, log=log)
    assert ok, "training failed"

    # level 1: weight fingerprint via deterministic probe
    env = {k: v for k, v in os.environ.items() if k != "PYOPENGL_PLATFORM"}
    env.update({"CUDA_VISIBLE_DEVICES": str(GPU), "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.3", "OMP_NUM_THREADS": "4"})
    pj = f"{UCB}/{EXP}_probe.json"
    r = subprocess.run([OPENPI_PY, "-u", "gradient_analysis/probe_ckpt.py",
                        "--ckpt_root", os.path.dirname(str(ckpt)),
                        "--step", str(os.path.basename(str(ckpt))), "--out", pj],
                       env=env, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-300:]
    rep_probe = json.load(open(pj))
    orig_probe = json.load(open(f"{UCB}/shadow/random_j106_probe9999.json"))

    # level 2+3: serve + eval, then per-episode agreement
    _, sp = pull.launch_server(CFG, str(ckpt), 8131, GPU, f"{UCB}/{EXP}_serve.log")
    up = pull.wait_for_port("127.0.0.1", 8131, proc=sp, log=log)
    assert up, "server failed"
    try:
        res = eval_set.eval_checkpoint(8131, EXP, "replica", EXP, workers=4, resume=True, log=log)
    finally:
        sp.terminate()
    delta = (res["mean"] - 0.5133) * 100

    ep = ledger.read("episodes")
    a = ep[ep.policy_id == "random_j106"][["start_id", "repeat_idx", "success"]]
    b = ep[ep.policy_id == EXP][["start_id", "repeat_idx", "success"]]
    m = a.merge(b, on=["start_id", "repeat_idx"], suffixes=("_orig", "_rep"))
    agree = float((m.success_orig == m.success_rep).mean())

    out = {"orig_delta": float(orig.delta * 100), "replica_delta": float(delta),
           "probe_orig": orig_probe, "probe_replica": rep_probe,
           "probe_absdiff": {k: abs(orig_probe[k] - rep_probe[k])
                             for k in ("loss_balanced", "loss_retention")},
           "episode_agreement": agree, "n_episode_pairs": len(m)}
    json.dump(out, open("/data/xinyua11/robocasa/gradient_analysis/replica_test_result.json", "w"))
    log(f"RESULT orig={orig.delta*100:+.2f} replica={delta:+.2f} "
        f"probe_diff={out['probe_absdiff']} episode_agreement={agree:.3f}")
    log("REPLICA TEST COMPLETE")


if __name__ == "__main__":
    main()
