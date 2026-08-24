"""Q2: absorption dynamics along each pull's own fine-tune trajectory.

For each pull in q2_jobs.json (tall_j3/j4, gradarm_a/b x j3/j4), at each retained
checkpoint (5000/10000/15000/19999), compute per-demo LoRA-grad norms + 2048-d JL
sketches (same machinery/proj seed as Q0) for three demo sets:
  own  : the pull's exact 200-demo draw (trained on -> should be absorbed)
  ctrl : 100 fixed pool demos in NO pull's draw (never trained -> should stay high)
  d0   : 60 D0 demos (in every fine-tune's mix -> should stay absorbed)

Separates "didn't learn the demos" from "learned the demos, demos lack the skill":
if own-norms collapse toward the D0 level while the pull's target stratum stayed
flat (tall: -2.7/0.0pp), the second explanation is nailed. gradarm_a_j3 (delta
-2.7pp) additionally probes interference on a harmful pull.

Output: gradient_analysis/q2_absorption/<pull>/<step>_{norms,sketches}.npy + meta.json
Resume-safe: skips (pull, step) outputs that already exist.

Run (openpi env), split by slot:
  CUDA_VISIBLE_DEVICES=0 ... q2_absorption.py --pulls tall_vessel_grasp_fail_j3,gradarm_a_j3,gradarm_a_j4
  CUDA_VISIBLE_DEVICES=1 ... q2_absorption.py --pulls tall_vessel_grasp_fail_j4,gradarm_b_j3,gradarm_b_j4
"""
import argparse
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
OUT = f"{GA}/q2_absorption"
K_FRAMES = 8
SKETCH_DIM = 2048
NNZ = 4000
PROJ_SEED = 12345
N_CTRL = 100
N_D0 = 60
CTRL_SEED = 20260803


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pulls", required=True, help="comma-separated pull_ids from q2_jobs.json")
    args = ap.parse_args()

    jobs = json.load(open(f"{GA}/q2_jobs.json"))
    lists = json.load(open(f"{GA}/demo_lists.json"))
    pulls = [p.strip() for p in args.pulls.split(",")]
    for p in pulls:
        assert p in jobs, p

    # fixed control: pool demos in NO pull's draw (same list for every trajectory)
    all_draws = set()
    for spec in jobs.values():
        all_draws.update(spec["demo_ids"])
    pool_eps = sorted(int(e) for e in lists["region_assignment"])
    ctrl_pool = [e for e in pool_eps if e not in all_draws]
    rng = np.random.RandomState(CTRL_SEED)
    ctrl = sorted(int(x) for x in rng.choice(ctrl_pool, size=N_CTRL, replace=False))
    d0 = lists["d0_sample"][:N_D0]

    cfg, raw, tds = inf.build_dataset(inf.MG_DEFAULT)
    starts = inf.episode_frame_starts(raw)
    gfn = inf.make_grad_fn("lora", 12)

    # fixed projection: probe flat dim once with the first pull's first ckpt
    first = jobs[pulls[0]]
    inf.CKPT_BASE = first["ckpt_root"]
    model = inf.load_model(cfg, first["steps"][0])
    e0 = first["demo_ids"][0]
    s, L = starts[e0]
    o0, a0 = inf.batch_to_inputs(inf.stack_batch(tds, inf.frame_indices(s, L, K_FRAMES)))
    Dflat = int(np.asarray(gfn(model, jax.random.key(0), o0, a0)).size)
    idx, sgn = inf.make_sparse_proj(Dflat, SKETCH_DIM, NNZ, seed=PROJ_SEED)
    del model
    print(f"[q2] flat dim={Dflat}; ctrl={len(ctrl)} d0={len(d0)}", flush=True)

    for pid in pulls:
        spec = jobs[pid]
        own = spec["demo_ids"]
        sets = [("own", own), ("ctrl", ctrl), ("d0", d0)]
        eps_all = [e for _, es in sets for e in es]
        tags_all = [t for t, es in sets for _ in es]
        odir = os.path.join(OUT, pid)
        os.makedirs(odir, exist_ok=True)
        json.dump({"episodes": eps_all, "sets": tags_all, "delta": spec["delta"],
                   "ckpt_root": spec["ckpt_root"], "steps": spec["steps"],
                   "k_frames": K_FRAMES, "proj_seed": PROJ_SEED},
                  open(os.path.join(odir, "meta.json"), "w"))

        print(f"[q2] {pid}: decoding {len(eps_all)} demos ...", flush=True)
        t0 = time.time()
        cache = {}
        for e in dict.fromkeys(eps_all):  # decode once per unique episode
            s, L = starts[e]
            cache[e] = inf.stack_batch(tds, inf.frame_indices(s, L, K_FRAMES))
        print(f"[q2] {pid}: decoded in {(time.time()-t0)/60:.1f} min", flush=True)

        inf.CKPT_BASE = spec["ckpt_root"]
        for step in spec["steps"]:
            np_norm = os.path.join(odir, f"{step}_norms.npy")
            np_sk = os.path.join(odir, f"{step}_sketches.npy")
            if os.path.exists(np_norm) and os.path.exists(np_sk):
                print(f"[q2] {pid}@{step}: exists, skip", flush=True)
                continue
            t0 = time.time()
            model = inf.load_model(cfg, step)
            S, N = [], []
            for i, e in enumerate(eps_all):
                o, a = inf.batch_to_inputs(cache[e])
                sk, gn = inf.grad_sketch_raw(gfn, model, e, step, o, a, idx, sgn)
                S.append(sk); N.append(gn)
                if (i + 1) % 120 == 0:
                    print(f"    {pid}@{step}: {i+1}/{len(eps_all)}", flush=True)
            np.save(np_sk, np.stack(S).astype(np.float32))
            np.save(np_norm, np.asarray(N, dtype=np.float32))
            del model
            own_n = np.asarray(N[: len(own)]); ctrl_n = np.asarray(N[len(own): len(own) + len(ctrl)])
            d0_n = np.asarray(N[len(own) + len(ctrl):])
            print(f"[q2] {pid}@{step} done ({(time.time()-t0)/60:.1f} min) | mean|g|: "
                  f"own {own_n.mean():.3f}  ctrl {ctrl_n.mean():.3f}  d0 {d0_n.mean():.3f}", flush=True)
    print("[q2] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
