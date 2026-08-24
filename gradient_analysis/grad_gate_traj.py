"""Does region-encoding EMERGE along a fine-tune trajectory? (the LESS last-chance test)

Q0 found the tall region is not encoded in the LoRA gradient at pi_0 (the trajectory
START). The one loophole: features formed DURING fine-tuning could make the region
gradient-visible at later checkpoints -- in which case a late-checkpoint trajectory-LESS
would have signal. Sharpest test: the tall-arm model itself (tall_vessel_grasp_fail_j4,
trained on 200 in-region demos) vs a control trajectory (random_j4).

Computes gate-set gradient sketches (same machinery as Q0: lora grads, real_dim=12, K=8
frames, 2048-d JL, proj seed 12345) at steps 5000/10000/19999 of both trajectories.
Gate sets = Q0's 120-vs-120 region gate MINUS any demo in either trajectory's own draw
(a trained-on demo's gradient is absorbed -> would fake a separation).

Output: gradient_analysis/gate_traj/<traj>/<step>_{sketches,norms}.npy + meta.json.
Analysis (CPU, after): analyze_gate_traj.py.

Run (openpi env): CUDA_VISIBLE_DEVICES=0 ... grad_gate_traj.py
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/data/xinyua11/robocasa")
sys.path.insert(0, "/data/xinyua11/robocasa/policy_analysis")
import influence_score as inf  # noqa: E402

import jax  # noqa: E402

GA = "/data/xinyua11/robocasa/gradient_analysis"
OUT = f"{GA}/gate_traj"
TRAJS = {
    "tall_vessel_grasp_fail_j4": "/data/xinyua11/openpi/checkpoints/pi0_ppc2sink_bandit_b/tall_vessel_grasp_fail_j4",
    "random_j4": "/data/xinyua11/openpi/checkpoints/pi0_ppc2sink_bandit_a/random_j4",
}
STEPS = [5000, 10000, 19999]
K_FRAMES = 8
SKETCH_DIM = 2048
NNZ = 4000
PROJ_SEED = 12345


def main():
    os.makedirs(OUT, exist_ok=True)
    lists = json.load(open(f"{GA}/demo_lists.json"))
    exclude = set()
    for t in TRAJS:
        exclude.update(lists["pulls"][t]["demo_ids"])
    gi = [e for e in lists["gate_in_region"] if e not in exclude]
    go = [e for e in lists["gate_out_region"] if e not in exclude]
    eps_all = gi + go
    labs = ["in"] * len(gi) + ["out"] * len(go)
    print(f"[gate_traj] gate after excluding both draws: {len(gi)} in / {len(go)} out", flush=True)

    cfg, raw, tds = inf.build_dataset(inf.MG_DEFAULT)
    starts = inf.episode_frame_starts(raw)
    gfn = inf.make_grad_fn("lora", 12)

    print("[gate_traj] decoding (shared across trajectories) ...", flush=True)
    t0 = time.time()
    cache = {e: inf.stack_batch(tds, inf.frame_indices(*starts[e], K_FRAMES)) for e in eps_all}
    print(f"[gate_traj] decoded {len(cache)} in {(time.time()-t0)/60:.1f} min", flush=True)

    first_root = list(TRAJS.values())[0]
    inf.CKPT_BASE = first_root
    model = inf.load_model(cfg, STEPS[0])
    e0 = eps_all[0]
    o0, a0 = inf.batch_to_inputs(cache[e0])
    Dflat = int(np.asarray(gfn(model, jax.random.key(0), o0, a0)).size)
    idx, sgn = inf.make_sparse_proj(Dflat, SKETCH_DIM, NNZ, seed=PROJ_SEED)
    del model

    for traj, root in TRAJS.items():
        odir = os.path.join(OUT, traj)
        os.makedirs(odir, exist_ok=True)
        json.dump({"episodes": eps_all, "labels": labs, "ckpt_root": root, "steps": STEPS,
                   "k_frames": K_FRAMES, "proj_seed": PROJ_SEED},
                  open(os.path.join(odir, "meta.json"), "w"))
        inf.CKPT_BASE = root
        for step in STEPS:
            p_sk = os.path.join(odir, f"{step}_sketches.npy")
            if os.path.exists(p_sk):
                print(f"[gate_traj] {traj}@{step}: exists, skip", flush=True)
                continue
            t0 = time.time()
            model = inf.load_model(cfg, step)
            S, N = [], []
            for i, e in enumerate(eps_all):
                o, a = inf.batch_to_inputs(cache[e])
                sk, gn = inf.grad_sketch_raw(gfn, model, e, step, o, a, idx, sgn)
                S.append(sk); N.append(gn)
                if (i + 1) % 100 == 0:
                    print(f"    {traj}@{step}: {i+1}/{len(eps_all)}", flush=True)
            np.save(p_sk, np.stack(S).astype(np.float32))
            np.save(os.path.join(odir, f"{step}_norms.npy"), np.asarray(N, dtype=np.float32))
            del model
            print(f"[gate_traj] {traj}@{step} done ({(time.time()-t0)/60:.1f} min)", flush=True)
    print("[gate_traj] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
