"""Task-2 per-demo LoRA-gradient JL sketches (PickPlaceCounterToCabinet).

pool_sketches.py's exact protocol at task-2 bindings, per the 2026-08-18
pre-registration in weakregion/REWARD_DECISION.md: gradient of the pi0
flow-matching loss at ppccab_base@9999, LoRA params, K=8 frames, 1 noise draw
(fold_in(episode, ckpt_step)), 2048-d sparse-JL nnz 4000 PROJ_SEED 12345,
fidelity gate corr>0.9 or STOP. Work list = the GATED pool (same E/D0
contamination gate objtest_build applies) — episodes that could never be
drawn into a pull don't need sketches.

Split across GPUs by env: SHARD="0/2" sketches episodes[0::2] (etc.); each
shard writes its own OUT_DIR suffix and the cluster step merges.

Run (openpi env, needs a free GPU):
  CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.18 OMP_NUM_THREADS=4 BANDIT_TASK_PROFILE=ppccab \
  /data/xinyua11/conda/envs/openpi/bin/python gradient_analysis/ppccab/ppccab_sketches.py [--probe]
"""
import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/data/xinyua11/robocasa")
sys.path.insert(0, "/data/xinyua11/robocasa/policy_analysis")

assert os.environ.get("BANDIT_TASK_PROFILE") == "ppccab", "set BANDIT_TASK_PROFILE=ppccab"

import influence_score as inf  # noqa: E402
import jax  # noqa: E402

CKPT_ROOT = "/data/xinyua11/openpi/checkpoints/pi0_ppccab_bandit_a/ppccab_base"
CKPT_STEP = 9999
BASE_CONFIG = "pi0_ppccab_bandit_a"
MG_DIR = ("/data/xinyua11/robocasa_pkg/datasets/v1.0/pretrain/atomic/"
          "PickPlaceCounterToCabinet/20250819/mg/demo/2025-08-20-21-56-25/lerobot")
TASK = "PickPlaceCounterToCabinet"
HORIZON = 750
K_FRAMES = 8
REAL_DIM = 12
SKETCH_DIM = 2048
NNZ = 4000
PROJ_SEED = 12345
SAVE_EVERY = 40
SHARD = os.environ.get("SHARD", "0/1")           # "i/n" -> episodes[i::n]
OUT_BASE = "/data/xinyua11/robocasa/gradient_analysis/ppccab/sketches_ppccabbase_9999"


def build_dataset_ppccab():
    """influence_score.build_dataset with task-2 config/dir/task/horizon."""
    inf.BASE_CONFIG = BASE_CONFIG
    cfg = inf._config.get_config(BASE_CONFIG)
    data = dataclasses.replace(
        cfg.data,
        data_dirs=[{"path": MG_DIR, "horizon": HORIZON, "filter_key": None,
                    "task": TASK, "split": "pretrain", "source": "mimicgen"}],
    )
    cfg = dataclasses.replace(cfg, data=data)
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    raw = inf._data_loader.create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model)
    tds = inf._data_loader.transform_dataset(raw, data_config)
    return cfg, raw, tds


def gated_pool_episodes():
    from bandit_v1 import pool, draw, eval_set, config
    W = pool.build_pool_table(write=False)
    d0_df = W[W.in_d0]
    W = W[~W.in_d0].copy()
    ef = eval_set.load_manifest()
    conf = draw._conflict_mask(W, ef, config.EPS_XY) | draw._conflict_mask(W, d0_df, config.EPS_XY)
    return sorted(int(x) for x in W[~conf].episode_index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--fidelity_n", type=int, default=10)
    args = ap.parse_args()

    si, sn = (int(x) for x in SHARD.split("/"))
    out_dir = OUT_BASE if sn == 1 else f"{OUT_BASE}_shard{si}of{sn}"
    os.makedirs(out_dir, exist_ok=True)

    pool_eps = gated_pool_episodes()
    worklist = pool_eps[si::sn]
    print(f"[ppccab_sketch] gated pool={len(pool_eps)} shard {SHARD} -> {len(worklist)} episodes "
          f"-> {out_dir}", flush=True)

    inf.CKPT_BASE = CKPT_ROOT
    cfg, raw, tds = build_dataset_ppccab()
    starts = inf.episode_frame_starts(raw)
    missing = [e for e in worklist if e not in starts]
    if missing:
        raise RuntimeError(f"{len(missing)} episodes missing from dataset, e.g. {missing[:5]}")

    gfn = inf.make_grad_fn("lora", REAL_DIM)
    t0 = time.time()
    model = inf.load_model(cfg, CKPT_STEP)
    print(f"[ppccab_sketch] loaded {CKPT_ROOT}/{CKPT_STEP} in {time.time()-t0:.0f}s", flush=True)

    e0 = worklist[0]
    s, L = starts[e0]
    obs0, act0 = inf.batch_to_inputs(inf.stack_batch(tds, inf.frame_indices(s, L, K_FRAMES)))
    t0 = time.time()
    g0 = np.asarray(gfn(model, jax.random.key(0), obs0, act0))
    Dflat = int(g0.size)
    print(f"[ppccab_sketch] flat grad dim={Dflat} norm={np.linalg.norm(g0):.4f} "
          f"(first grad {time.time()-t0:.0f}s incl compile)", flush=True)
    idx, sgn = inf.make_sparse_proj(Dflat, SKETCH_DIM, NNZ, seed=PROJ_SEED)

    if args.probe:
        sk, n = inf.grad_sketch_raw(gfn, model, e0, CKPT_STEP, obs0, act0, idx, sgn)
        print(f"[probe] ep {e0}: sketch shape={sk.shape} norm={n:.4f} | OK", flush=True)
        return

    import jax.numpy as jnp
    fid_eps = worklist[: args.fidelity_n]
    full, sk = [], []
    for e in fid_eps:
        s, L = starts[e]
        o, a = inf.batch_to_inputs(inf.stack_batch(tds, inf.frame_indices(s, L, K_FRAMES)))
        r = jax.random.fold_in(jax.random.key(int(e)), CKPT_STEP)
        g = gfn(model, r, o, a)
        g = g / (jnp.linalg.norm(g) + 1e-12)
        full.append(np.asarray(g)); sk.append(np.asarray(inf._sketch(g, idx, sgn)))
    full = np.stack(full); sk_m = np.stack(sk)
    tri = np.triu_indices(len(fid_eps), 1)
    tc = (full @ full.T)[tri]
    scn = sk_m / (np.linalg.norm(sk_m, axis=1, keepdims=True) + 1e-12)
    sc = (scn @ scn.T)[tri]
    corr = float(np.corrcoef(tc, sc)[0, 1])
    print(f"[ppccab_sketch] FIDELITY corr={corr:.3f} mae={np.abs(tc-sc).mean():.3f} "
          f"({'PASS' if corr > 0.9 else 'FAIL -- STOP'})", flush=True)
    del full
    if corr <= 0.9:
        raise RuntimeError("sketch fidelity check failed; do not proceed")

    ep_path = os.path.join(out_dir, "episodes.json")
    sk_path = os.path.join(out_dir, "sketches.npy")
    nm_path = os.path.join(out_dir, "norms.npy")
    done_eps, S, N = [], [], []
    if os.path.exists(ep_path):
        prev = json.load(open(ep_path))
        if prev.get("proj_seed") == PROJ_SEED and prev.get("ckpt_step") == CKPT_STEP:
            done_eps = prev["episodes"]
            S = list(np.load(sk_path)[: len(done_eps)])
            N = list(np.load(nm_path)[: len(done_eps)])
            print(f"[ppccab_sketch] resuming: {len(done_eps)} done", flush=True)
    done_set = set(done_eps)
    todo = [e for e in worklist if e not in done_set]

    def save():
        np.save(sk_path, np.stack(S).astype(np.float32))
        np.save(nm_path, np.asarray(N, dtype=np.float32))
        json.dump({"episodes": done_eps, "ckpt_root": CKPT_ROOT, "ckpt_step": CKPT_STEP,
                   "k_frames": K_FRAMES, "real_dim": REAL_DIM, "sketch_dim": SKETCH_DIM,
                   "nnz": NNZ, "proj_seed": PROJ_SEED, "fidelity_corr": corr,
                   "shard": SHARD, "n_total_planned": len(worklist)},
                  open(ep_path, "w"))

    t_start = time.time()
    for i, e in enumerate(todo):
        s, L = starts[e]
        o, a = inf.batch_to_inputs(inf.stack_batch(tds, inf.frame_indices(s, L, K_FRAMES)))
        rs, gn = inf.grad_sketch_raw(gfn, model, e, CKPT_STEP, o, a, idx, sgn)
        S.append(rs); N.append(gn); done_eps.append(int(e))
        if (i + 1) % SAVE_EVERY == 0 or (i + 1) == len(todo):
            save()
            rate = (i + 1) / (time.time() - t_start)
            print(f"[ppccab_sketch] {i+1}/{len(todo)} ({rate*60:.1f}/min, "
                  f"eta {(len(todo)-i-1)/max(rate,1e-9)/60:.0f} min)", flush=True)
    print("PPCCAB SKETCHES COMPLETE", flush=True)


if __name__ == "__main__":
    main()
