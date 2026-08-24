"""Theory demo, stage 1: JL sketches of realized LoRA weight-updates.

For every pull with a retained final checkpoint (theory_jobs.json), compute
Delta-theta = LoRA(final) - LoRA(pi_0), flattened in a fixed deterministic leaf
order, and sketch it with THE SAME sparse JL projection (dim 2048, nnz 4000,
seed 12345) used for all per-demo gradient sketches -- so weight updates and
batch gradients live in one comparable 2048-d space.

Also sketches step-5000 deltas for a 6-pull subset (first-order decay check).

CPU-only (JAX_PLATFORMS=cpu): orbax restore + numpy. Skips missing checkpoints
and already-done entries. Output: gradient_analysis/theory_weights/
  <pull>__<step>.npz  {sketch[2048], norm, dim}
  pi0_flat_norm.json

Run (openpi env):
  JAX_PLATFORMS=cpu /data/xinyua11/conda/envs/openpi/bin/python -u \
    gradient_analysis/theory_weights.py
"""
import gc
import json
import os

import numpy as np

GA = "/data/xinyua11/robocasa/gradient_analysis"
OUT = f"{GA}/theory_weights"
PI0 = "/data/xinyua11/openpi/checkpoints/pi0_ppc2sink_pi0base/pi0_v1/19999/params"
EXPECT_DIM = 49_987_584  # flat LoRA dim of the per-demo gradient sketches
SKETCH_DIM, NNZ, SEED = 2048, 4000, 12345
EARLY_SUBSET = ["tall_vessel_grasp_fail_j3", "random_j3", "gradarm_a_j3",
                "gradarm_b_j3", "style_hi_j3", "style_lo_j3"]


def make_sparse_proj(dim_in, dim_out, nnz, seed):
    """Identical to policy_analysis/influence_score.py make_sparse_proj (numpy)."""
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, dim_in, size=(dim_out, nnz)).astype(np.int64)
    sgn = (rng.randint(0, 2, size=(dim_out, nnz)).astype(np.float32) * 2.0 - 1.0)
    return idx, sgn


def lora_flat(params_path):
    """Restore a checkpoint and flatten its LoRA leaves in sorted-path order."""
    from openpi.models import model as _model
    params = _model.restore_params(params_path)
    import jax
    leaves = jax.tree_util.tree_flatten_with_path(params)[0]
    lora = [(jax.tree_util.keystr(kp), np.asarray(v, dtype=np.float32).ravel())
            for kp, v in leaves if "lora" in jax.tree_util.keystr(kp).lower()]
    lora.sort(key=lambda t: t[0])
    flat = np.concatenate([v for _, v in lora])
    del params, leaves, lora
    gc.collect()
    return flat


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = json.load(open(f"{GA}/theory_jobs.json"))

    print("[w] restoring pi_0 ...", flush=True)
    flat0 = lora_flat(PI0)
    assert flat0.size == EXPECT_DIM, f"LoRA flat dim {flat0.size} != expected {EXPECT_DIM}"
    json.dump({"pi0_lora_norm": float(np.linalg.norm(flat0)), "dim": int(flat0.size)},
              open(f"{OUT}/pi0_flat_norm.json", "w"))
    idx, sgn = make_sparse_proj(flat0.size, SKETCH_DIM, NNZ, SEED)
    print(f"[w] pi_0 LoRA dim {flat0.size}, |theta0| {np.linalg.norm(flat0):.2f}", flush=True)

    work = []
    for pid, spec in sorted(jobs.items()):
        steps = spec.get("steps", [])
        final = 19999 if (not steps or 19999 in steps) else max(steps)
        work.append((pid, final, f"{spec['ckpt_root']}/{final}/params"))
        if pid in EARLY_SUBSET and 5000 in steps:
            work.append((pid, 5000, f"{spec['ckpt_root']}/5000/params"))

    for i, (pid, step, path) in enumerate(work):
        out = f"{OUT}/{pid}__{step}.npz"
        if os.path.exists(out):
            print(f"[w] {pid}@{step}: exists, skip", flush=True)
            continue
        if not os.path.isdir(path):
            print(f"[w] {pid}@{step}: MISSING ckpt ({path}), skip", flush=True)
            continue
        try:
            flat = lora_flat(path)
        except Exception as e:  # noqa: BLE001 -- log-and-continue over 45 restores
            print(f"[w] {pid}@{step}: restore FAILED ({type(e).__name__}: {e}), skip", flush=True)
            continue
        d = flat - flat0
        del flat
        sk = (d[idx] * sgn).sum(axis=1)
        np.savez(out, sketch=sk.astype(np.float32), norm=float(np.linalg.norm(d)))
        print(f"[w] {i+1}/{len(work)} {pid}@{step}: |dtheta| {np.linalg.norm(d):.3f}", flush=True)
        del d
        gc.collect()
    print("[w] DONE", flush=True)


if __name__ == "__main__":
    main()
