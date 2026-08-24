"""Per-demo LoRA-gradient JL sketches at the bandit's pi_0 branching checkpoint.

Q0 (encoding gate) + Q1 (treatment-set) inputs for the gradient analysis
(2026-07-30). Reuses policy_analysis/influence_score.py machinery verbatim:
lora-adapter gradients (~50M dims), masked flow loss (real_dim=12), K=8 frames
per demo, sparse JL sketch (dim 2048, nnz 4000; fidelity self-checked here the
same way run_sketch validates it -- want true-cos vs sketch-cos corr > 0.9).

Demo sets (from gradient_analysis/demo_lists.json, built by prep_demo_lists.py):
  gate_in    : 120 tall_vessel_grasp_fail-region demos (Q0 positive class)
  gate_out   : 120 out-of-region demos, stratified over the other arms (Q0 negatives)
  pull sets  : the exact 200-demo draws of tall_j3/j4, random_j3/j4, mid_j3/j4, easy_j3/j4

Checkpoint: pi0_ppc2sink_pi0base/pi0_v1/19999 -- the weights every pull's
fine-tune starts from (the natural reference; no warmup surrogate needed).

Output: gradient_analysis/sketches_pi0base_19999/
  sketches.npy  [n, 2048] raw (un-normalized) sketches, float32
  norms.npy     [n] full-gradient L2 norms
  episodes.json ordered episode ids + set membership + progress marker
Incremental: saves every SAVE_EVERY demos; resumes from episodes.json.

Race-safety (GRADIENT_ANALYSIS_HANDOFF.md): runs on CUDA_VISIBLE_DEVICES=1 with
XLA_PYTHON_CLIENT_PREALLOCATE=false + MEM_FRACTION<=0.18 (GPU1 is below the
race's 70GB launch threshold anyway -- the race cannot claim it); never touches
the ledger; thread caps set by the launcher.

Run (openpi env):
  CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.18 OMP_NUM_THREADS=4 \
  /data/xinyua11/conda/envs/openpi/bin/python gradient_analysis/grad_sketches.py [--probe]
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
OUT_DIR = "/data/xinyua11/robocasa/gradient_analysis/sketches_pi0base_19999"
K_FRAMES = 8
REAL_DIM = 12
SKETCH_DIM = 2048
NNZ = 4000
PROJ_SEED = 12345  # same seed as run_sketch -> comparable across sessions
SAVE_EVERY = 40
PULL_SETS = ["tall_vessel_grasp_fail_j3", "tall_vessel_grasp_fail_j4",
             "random_j3", "random_j4", "mid_band_j3", "mid_band_j4",
             "easy_band_j3", "easy_band_j4"]


def build_worklist(lists):
    """Ordered (episode, tags) worklist: gate sets first, then pull sets.
    An episode can belong to several sets; computed once, tagged with all."""
    tags = {}
    def add(eps, tag):
        for e in eps:
            tags.setdefault(int(e), []).append(tag)
    add(lists["gate_in_region"], "gate_in")
    add(lists["gate_out_region"], "gate_out")
    add(lists.get("d0_sample", []), "d0_sample")
    for ps in PULL_SETS:
        if ps in lists["pulls"]:
            add(lists["pulls"][ps]["demo_ids"], ps)
    # order: gate episodes first (so Q0 completes earliest), then the rest
    gate = [e for e in tags if any(t.startswith("gate") for t in tags[e])]
    rest = [e for e in tags if e not in set(gate)]
    return sorted(gate) + sorted(rest), tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="1 demo end-to-end sanity, then exit")
    ap.add_argument("--fidelity_n", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    lists = json.load(open(LISTS))
    worklist, tags = build_worklist(lists)
    print(f"[sketch] worklist={len(worklist)} unique demos "
          f"(gate {sum(1 for e in worklist if any(t.startswith('gate') for t in tags[e]))}, "
          f"pull-only rest)", flush=True)

    inf.CKPT_BASE = CKPT_ROOT  # module global read by inf.load_model
    cfg, raw, tds = inf.build_dataset(inf.MG_DEFAULT)
    starts = inf.episode_frame_starts(raw)
    missing = [e for e in worklist if e not in starts]
    if missing:
        raise RuntimeError(f"{len(missing)} episodes not in dataset, e.g. {missing[:5]}")

    gfn = inf.make_grad_fn("lora", REAL_DIM)
    t0 = time.time()
    model = inf.load_model(cfg, CKPT_STEP)
    print(f"[sketch] loaded {CKPT_ROOT}/{CKPT_STEP} in {time.time()-t0:.0f}s", flush=True)

    # flat dim + fixed projection (same seed as prior RC-LESS runs)
    e0 = worklist[0]
    s, L = starts[e0]
    obs0, act0 = inf.batch_to_inputs(inf.stack_batch(tds, inf.frame_indices(s, L, K_FRAMES)))
    t0 = time.time()
    g0 = np.asarray(gfn(model, jax.random.key(0), obs0, act0))
    Dflat = int(g0.size)
    print(f"[sketch] flat grad dim={Dflat} norm={np.linalg.norm(g0):.4f} "
          f"(first grad {time.time()-t0:.0f}s incl compile)", flush=True)
    idx, sgn = inf.make_sparse_proj(Dflat, SKETCH_DIM, NNZ, seed=PROJ_SEED)

    if args.probe:
        sk, n = inf.grad_sketch_raw(gfn, model, e0, CKPT_STEP, obs0, act0, idx, sgn)
        print(f"[probe] ep {e0}: sketch shape={sk.shape} norm={n:.4f} | OK", flush=True)
        return

    # sketch-fidelity self-check: true-cos vs sketch-cos on first fidelity_n demos
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
    print(f"[sketch] FIDELITY true-cos vs sketch-cos: corr={corr:.3f} "
          f"mae={np.abs(tc-sc).mean():.3f} ({'PASS' if corr > 0.9 else 'FAIL -- STOP'})", flush=True)
    del full
    if corr <= 0.9:
        raise RuntimeError("sketch fidelity check failed; do not proceed")

    # resume support
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
            print(f"[sketch] resuming: {len(done_eps)} already done", flush=True)
    done_set = set(done_eps)
    todo = [e for e in worklist if e not in done_set]

    def save():
        np.save(sk_path, np.stack(S).astype(np.float32))
        np.save(nm_path, np.asarray(N, dtype=np.float32))
        json.dump({"episodes": done_eps, "tags": {str(e): tags[e] for e in done_eps},
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
            print(f"[sketch] {i+1}/{len(todo)} ({rate*60:.1f} demos/min, "
                  f"eta {(len(todo)-i-1)/max(rate,1e-9)/60:.0f} min)", flush=True)
    print(f"[sketch] DONE -> {OUT_DIR} ({len(done_eps)} demos)", flush=True)


if __name__ == "__main__":
    main()
