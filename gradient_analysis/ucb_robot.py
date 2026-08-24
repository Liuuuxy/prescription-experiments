"""Robot loss-reward bandit — burst pulls + Phase A mini-calibration.

A burst pull = fresh LoRA fine-tune from pretrain on D0 + B(200) drawn from the
arm, ONLY num_train_steps=1500 (~36 min on one H100), then probe the burst
checkpoint's balanced/retention losses (probe_ckpt.py in the openpi env);
reward = loss_balanced - loss_retention (calibrated composite, inverted-sign
convention: "learn new, don't conform, keep D0").

Phase A (this script's main): 4 arms with KNOWN rollout ground truth
(style_hi, style_lo, mid_band, random) x 2 draws = 8 bursts. GATE: the burst
reward must rank style_lo last and put style_hi/mid_band above random --
otherwise the reward is not trustworthy and Phase B (UCB loop) is off.

Own ledger: gradient_analysis/ucb_robot/ledger.parquet (never the race ledger).
Slot a / GPU0 only. Phase A keeps burst ckpts (~7GB each) for re-probing.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/xinyua11/robocasa")
os.chdir("/data/xinyua11/robocasa")

from bandit_v1 import pull, draw, pool, eval_set

UCB_DIR = "/data/xinyua11/robocasa/gradient_analysis/ucb_robot"
LEDGER = f"{UCB_DIR}/ledger.parquet"
CONFIG = "pi0_ppc2sink_bandit_a"
SLOT_LINK = "/data/xinyua11/ft_arms/ppc2sink_bandit_slot_a"
GPU = 0
B = 200
BURST_STEPS = 1500
OPENPI_PY = "/data/xinyua11/conda/envs/openpi/bin/python"
PHASE_A = [("style_hi", 0), ("style_lo", 0), ("mid_band", 0), ("random", 0),
           ("style_hi", 1), ("style_lo", 1), ("mid_band", 1), ("random", 1)]


def log(*a):
    print(f"[ucb_robot {time.strftime('%H:%M:%S')}]", *a, flush=True)


def load_ctx():
    arms = json.load(open(f"{UCB_DIR}/arms.json"))
    pairs = {}
    for name, ids in arms.items():
        if name == "random":
            continue
        for e in ids:
            pairs.setdefault(int(e), []).append(name)
    # one Series per arm avoids multi-label collisions: build on demand
    pool_df = pool.build_pool_table(write=False)
    e_features = eval_set.load_manifest()
    return arms, pool_df, e_features


def regions_for(arm, arms):
    s = pd.Series({int(e): arm for e in arms[arm]}, dtype=object)
    s.index.name = "episode_index"
    return s


def burst_pull(arm, tag, seed, arms, pool_df, e_features):
    pull_id = f"ucb_{arm}_{tag}"
    rng = np.random.default_rng(900000 + (hash(pull_id) % 100000))
    regions = regions_for(arm, arms) if arm != "random" else pd.Series(dtype=object)
    demo_ids = draw.pull_demos(arm, B, rng, pool_df=pool_df, regions=regions,
                               e_features=e_features)
    d0 = pull.load_d0_episode_ids()
    pull.assemble_episode_ids(d0, demo_ids)
    aj = pull.write_pull_arms_json(pull_id, d0, demo_ids,
                                   path=f"{UCB_DIR}/{pull_id}_arms.json")
    dst = f"/data/xinyua11/ft_arms/ppc2sink_bandit_{pull_id}"
    if not os.path.exists(dst):
        pull.run_dataset_build(aj if isinstance(aj, (str, os.PathLike)) else f"{UCB_DIR}/{pull_id}_arms.json",
                               "base+pull", dst)
    pull.symlink_slot(dst, SLOT_LINK)
    cmd = pull.train_cmd(CONFIG, pull_id, seed, num_train_steps=BURST_STEPS)
    ckpt = pull.ckpt_final_dir(CONFIG, pull_id, BURST_STEPS)
    tlog = f"{UCB_DIR}/{pull_id}_train.log"
    t0 = time.time()
    proc = pull.launch_training(cmd, GPU, tlog)
    ok = pull.wait_for_checkpoint(proc, ckpt, num_train_steps=BURST_STEPS, log=log)
    if not ok:
        log(f"{pull_id}: TRAINING FAILED (see {tlog})")
        return None
    ckpt = str(ckpt)
    root, step = os.path.dirname(ckpt), int(os.path.basename(ckpt))
    pj = f"{UCB_DIR}/{pull_id}_probe.json"
    env = {k: v for k, v in os.environ.items() if k != "PYOPENGL_PLATFORM"}
    env.update({"CUDA_VISIBLE_DEVICES": str(GPU),
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.3",
                "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
    r = subprocess.run([OPENPI_PY, "-u", "gradient_analysis/probe_ckpt.py",
                        "--ckpt_root", root, "--step", str(step), "--out", pj],
                       env=env, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        log(f"{pull_id}: PROBE FAILED: {r.stderr[-400:]}")
        return None
    res = json.load(open(pj))
    row = {"pull_id": pull_id, "arm": arm, "tag": tag, "seed": seed,
           "n_demos": len(demo_ids), **res,
           "minutes": round((time.time() - t0) / 60, 1)}
    rows = []
    if os.path.exists(LEDGER):
        rows = pd.read_parquet(LEDGER).to_dict("records")
    rows.append(row)
    pd.DataFrame(rows).to_parquet(LEDGER)
    log(f"BURST DONE {pull_id}: reward={res['reward']:+.4f} "
        f"(bal {res['loss_balanced']:.4f} ret {res['loss_retention']:.4f}) "
        f"{row['minutes']}min")
    return row


def main():
    arms, pool_df, e_features = load_ctx()
    done = set()
    if os.path.exists(LEDGER):
        done = set(pd.read_parquet(LEDGER).pull_id)
    for arm, s in PHASE_A:
        pid = f"ucb_{arm}_a{s}"
        if pid in done:
            log(f"{pid}: already done -- skipping")
            continue
        burst_pull(arm, f"a{s}", 4000 + s, arms, pool_df, e_features)
    log("PHASE A COMPLETE")
    df = pd.read_parquet(LEDGER)
    df = df[df.tag.str.startswith("a")]
    means = df.groupby("arm")["reward"].mean().sort_values(ascending=False)
    log(f"PHASE A RANKING (burst reward): {dict(means.round(4))}")
    log("ground truth SR order: style_hi(+3.6) ~ mid_band(+3.6) > random(+2.6) > style_lo(-1.3)")


if __name__ == "__main__":
    main()
