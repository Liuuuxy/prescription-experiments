"""Probe a checkpoint's balanced/retention losses (openpi env CLI).

Usage: probe_ckpt.py --ckpt_root <root-with-step-subdirs> --step N --out out.json
Reward convention (calibrated 2026-08-05, loss_probe_calibration): GOOD models
have HIGH balanced-pool loss and LOW retention (D0) loss; composite reward =
loss_balanced - loss_retention.
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "/data/xinyua11/robocasa")
sys.path.insert(0, "/data/xinyua11/robocasa/policy_analysis")
import influence_score as inf  # noqa: E402

import jax  # noqa: E402
from flax import nnx  # noqa: E402

LISTS = "/data/xinyua11/robocasa/gradient_analysis/demo_lists.json"
K_FRAMES = 8
N_DRAWS = 2
N_PER_PROBE = 80
REAL_DIM = 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lists = json.load(open(LISTS))
    probes = {"balanced": [int(e) for e in lists["gate_out_region"]][:N_PER_PROBE],
              "retention": [int(e) for e in lists["d0_sample"]][:N_PER_PROBE]}

    inf.CKPT_BASE = args.ckpt_root
    cfg, raw, tds = inf.build_dataset(inf.MG_DEFAULT)
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
