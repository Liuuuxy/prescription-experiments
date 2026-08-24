"""Per-demo LoRA-gradient JL sketches for the FULL W pool (gradient-cluster arms).

Extends grad_sketches.py (same conventions, same projection -> rows concatenate
with sketches_pi0base_19999/) to every demo in demo_lists.json's
region_assignment (9,485 pool demos), skipping any episode already sketched in
the Q0 output dir. Purpose: k-means arms in gradient space for the
"well-defined arms" test (2026-08-01) -- race gradient-cluster arms vs the
frozen race's random control at paired seeds.

Output: gradient_analysis/sketches_pi0base_19999_pool/   (NEW dir; the Q0
artifact dir stays byte-stable)
  sketches.npy  [n, 2048] raw (un-normalized) sketches, float32
  norms.npy     [n] full-gradient L2 norms
  episodes.json ordered episode ids + region tag + provenance

Run (openpi env):
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.18 OMP_NUM_THREADS=4 \
  /data/xinyua11/conda/envs/openpi/bin/python gradient_analysis/pool_sketches.py [--probe]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/data/xinyua11/robocasa")
sys.path.insert(0, "/data/xinyua11/robocasa/policy_analysis")
import influence_score as inf  # noqa: E402  (the validated machinery)

import jax  # noqa: E402

CKPT_ROOT = "/data/xinyua11/openpi/checkpoints/pi0_ppc2sink_pi0base/pi0_v1"
CKPT_STEP = 19999
LISTS = "/data/xinyua11/robocasa/gradient_analysis/demo_lists.json"
Q0_DIR = "/data/xinyua11/robocasa/gradient_analysis/sketches_pi0base_19999"
OUT_DIR = "/data/xinyua11/robocasa/gradient_analysis/sketches_pi0base_19999_pool"
K_FRAMES = 8
REAL_DIM = 12
SKETCH_DIM = 2048
NNZ = 4000
PROJ_SEED = 12345  # identical projection -> rows join the Q0 sketches
SAVE_EVERY = 40


def build_worklist():
    """All pool episodes (region_assignment) not already sketched by Q0."""
    lists = json.load(open(LISTS))
    region = {int(e): r for e, r in lists["region_assignment"].items()}
    done_q0 = set()
    q0_ep = os.path.join(Q0_DIR, "episodes.json")
    if os.path.exists(q0_ep):
        prev = json.load(open(q0_ep))
        if prev.get("proj_seed") == PROJ_SEED and prev.get("ckpt_step") == CKPT_STEP:
            done_q0 = set(int(e) for e in prev["episodes"])
    todo = sorted(e for e in region if e not in done_q0)
    return todo, region, done_q0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="1 demo end-to-end sanity, then exit")
    ap.add_argument("--fidelity_n", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    worklist, region, done_q0 = build_worklist()
    print(f"[pool_sketch] pool={len(region)} already-in-Q0={len(done_q0 & set(region))} "
          f"todo-here={len(worklist)}", flush=True)

    inf.CKPT_BASE = CKPT_ROOT  # module global read by inf.load_model
    cfg, raw, tds = inf.build_dataset(inf.MG_DEFAULT)
    starts = inf.episode_frame_starts(raw)
    missing = [e for e in worklist if e not in starts]
    if missing:
        raise RuntimeError(f"{len(missing)} episodes not in dataset, e.g. {missing[:5]}")

    gfn = inf.make_grad_fn("lora", REAL_DIM)
    t0 = time.time()
    model = inf.load_model(cfg, CKPT_STEP)
    print(f"[pool_sketch] loaded {CKPT_ROOT}/{CKPT_STEP} in {time.time()-t0:.0f}s", flush=True)

    e0 = worklist[0]
    s, L = starts[e0]
    obs0, act0 = inf.batch_to_inputs(inf.stack_batch(tds, inf.frame_indices(s, L, K_FRAMES)))
    t0 = time.time()
    g0 = np.asarray(gfn(model, jax.random.key(0), obs0, act0))
    Dflat = int(g0.size)
    print(f"[pool_sketch] flat grad dim={Dflat} norm={np.linalg.norm(g0):.4f} "
          f"(first grad {time.time()-t0:.0f}s incl compile)", flush=True)
    idx, sgn = inf.make_sparse_proj(Dflat, SKETCH_DIM, NNZ, seed=PROJ_SEED)

    if args.probe:
        sk, n = inf.grad_sketch_raw(gfn, model, e0, CKPT_STEP, obs0, act0, idx, sgn)
        print(f"[probe] ep {e0}: sketch shape={sk.shape} norm={n:.4f} | OK", flush=True)
        return

    # sketch-fidelity self-check (same protocol as grad_sketches.py)
    import jax.numpy as jnp
    fid_eps = worklist[:args.fidelity_n]
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
    print(f"[pool_sketch] FIDELITY true-cos vs sketch-cos: corr={corr:.3f} "
          f"mae={np.abs(tc-sc).mean():.3f} ({'PASS' if corr > 0.9 else 'FAIL -- STOP'})", flush=True)
    del full
    if corr <= 0.9:
        raise RuntimeError("sketch fidelity check failed; do not proceed")

    # resume support (identical protocol)
    ep_path = os.path.join(OUT_DIR, "episodes.json")
    sk_path = os.path.join(OUT_DIR, "sketches.npy")
    nm_path = os.path.join(OUT_DIR, "norms.npy")
    done_eps, S, N = [], [], []
    if os.path.exists(ep_path):
        prev = json.load(open(ep_path))
        if prev.get("proj_seed") == PROJ_SEED and prev.get("ckpt_step") == CKPT_STEP:
            done_eps = prev["episodes"]
            S = list(np.load(sk_path)[: len(done_eps)])
            N = list(np.load(nm_path)[: len(done_eps)])
            print(f"[pool_sketch] resuming: {len(done_eps)} already done", flush=True)
    done_set = set(done_eps)
    todo = [e for e in worklist if e not in done_set]

    def save():
        np.save(sk_path, np.stack(S).astype(np.float32))
        np.save(nm_path, np.asarray(N, dtype=np.float32))
        json.dump({"episodes": done_eps,
                   "tags": {str(e): [region[e]] for e in done_eps},
                   "ckpt_root": CKPT_ROOT, "ckpt_step": CKPT_STEP, "k_frames": K_FRAMES,
                   "real_dim": REAL_DIM, "sketch_dim": SKETCH_DIM, "nnz": NNZ,
                   "proj_seed": PROJ_SEED, "fidelity_corr": corr,
                   "n_total_planned": len(worklist)},
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
            print(f"[pool_sketch] {i+1}/{len(todo)} ({rate*60:.1f} demos/min, "
                  f"eta {(len(todo)-i-1)/max(rate,1e-9)/60:.0f} min)", flush=True)
    print(f"[pool_sketch] DONE -> {OUT_DIR} ({len(done_eps)} demos)", flush=True)


if __name__ == "__main__":
    main()
