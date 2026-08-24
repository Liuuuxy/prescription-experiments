"""Task-2 checkpoint probe (probe_ckpt.py at ppccab bindings) — the cheap
proxy phi for the trust-gated index (2026-08-18 build).

phi convention identical to task-1 calibration: composite reward =
loss_balanced − loss_retention (good models keep D0 loss LOW and generic-pool
loss HIGH). Probe sets are frozen on first run to probe_sets.json:
  balanced : 80 demos, fixed rng 20260818, from the GATED pool (E/D0 gate)
  retention: 80 demos, fixed rng 20260819, from task-2 D0
Usage: ppccab_probe_ckpt.py --ckpt_root <root> --step N --out out.json
(openpi env, BANDIT_TASK_PROFILE=ppccab, one GPU, MEM_FRACTION<=0.3)
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/data/xinyua11/robocasa")
sys.path.insert(0, "/data/xinyua11/robocasa/policy_analysis")
sys.path.insert(0, "/data/xinyua11/robocasa/gradient_analysis/ppccab")
assert os.environ.get("BANDIT_TASK_PROFILE") == "ppccab"

import influence_score as inf  # noqa: E402
from ppccab_sketches import build_dataset_ppccab, gated_pool_episodes  # noqa: E402

import jax  # noqa: E402
from flax import nnx  # noqa: E402

GA = "/data/xinyua11/robocasa/gradient_analysis/ppccab"
SETS = f"{GA}/probe_sets.json"
K_FRAMES = 8
N_DRAWS = 2
N_PER_PROBE = 80
REAL_DIM = 12


def load_or_freeze_sets():
    if os.path.exists(SETS):
        return json.load(open(SETS))
    from bandit_v1 import pull
    gated = gated_pool_episodes()
    bal = sorted(int(x) for x in np.random.default_rng(20260818).choice(gated, N_PER_PROBE, replace=False))
    d0 = pull.load_d0_episode_ids()
    ret = sorted(int(x) for x in np.random.default_rng(20260819).choice(d0, N_PER_PROBE, replace=False))
    sets = {"balanced": bal, "retention": ret}
    json.dump(sets, open(SETS, "w"))
    print(f"[probe] frozen probe sets -> {SETS}", flush=True)
    return sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    probes = load_or_freeze_sets()
    inf.CKPT_BASE = args.ckpt_root
    cfg, raw, tds = build_dataset_ppccab()
    starts = inf.episode_frame_starts(raw)
    model = inf.load_model(cfg, args.step)

    @nnx.jit
    def loss_fn(model, rng, obs, act):
        return inf.masked_flow_loss(model, rng, obs, act, REAL_DIM)

    out = {}
    for pname, eps in probes.items():
        vals = []
        for e in eps:
            if e not in starts:
                continue
            s, L = starts[e]
            o, a = inf.batch_to_inputs(inf.stack_batch(tds, inf.frame_indices(s, L, K_FRAMES)))
            for d in range(N_DRAWS):
                r = jax.random.fold_in(jax.random.key(int(e)), args.step + d)
                vals.append(float(loss_fn(model, r, o, a)))
        out[f"loss_{pname}"] = float(np.mean(vals))
    out["reward"] = out["loss_balanced"] - out["loss_retention"]
    json.dump(out, open(args.out, "w"))
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
