"""Forward-pass-only per-demo flow-matching loss evaluator for pi0 bandit checkpoints.

NO fine-tuning, NO gradients. For each (checkpoint, demo-set) job it computes a
[n_demos, n_draws] matrix of pi0 flow-matching losses, masked to the 12 real action
dims (same masking as policy_analysis/influence_score.py so numbers are comparable
with the gradient work).

COMMON RANDOM NUMBERS: the (preprocess, noise, time) rng for demo `ep`, draw `m` is
    jax.random.fold_in(jax.random.fold_in(jax.random.key(SEED_BASE), ep), m)
which depends ONLY on (ep, m) -- never on the checkpoint. So L(z;A) - L(z;B) is a
paired difference over identical noise/time samples, killing the dominant variance
term. This is what makes an M=4..8 draw estimate stable.

Jobs file: [{"ckpt": "<...>/params", "demos": "<set-key>", "out": "<name>.npz"}, ...]
Sets file: {"<set-key>": [episode_index, ...], ...}
Decoding happens ONCE for the union of all sets; models are loaded/freed per job.
Jobs whose out file already exists are skipped (crash-safe resume).

Run (openpi env, tiny JAX memory footprint, per the GPU rules):
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.12 OMP_NUM_THREADS=4 \
  /data/xinyua11/conda/envs/openpi/bin/python loss_eval.py --jobs jobs.json --sets sets.json --out_dir lossmat
"""
import argparse
import dataclasses
import gc
import json
import os
import time

import numpy as np

import jax
import jax.numpy as jnp
import flax.nnx as nnx

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.models.model as _model
import openpi.models.pi0 as _pi0

POOL_LEROBOT = ("/data/xinyua11/robocasa_pkg/datasets/v1.0/pretrain/atomic/"
                "PickPlaceCounterToSink/20250819/mg/demo/2025-08-20-22-32-27/lerobot")
BASE_CONFIG = "pi0_ppc2sink_bandit_a"   # arch + data transforms only; params loaded from --ckpt
SEED_BASE = 777000


# --------------------------------------------------------------------------- data
def build_dataset(mg_dir):
    cfg = _config.get_config(BASE_CONFIG)
    data = dataclasses.replace(
        cfg.data,
        data_dirs=[{"path": mg_dir, "horizon": 600, "filter_key": None,
                    "task": "PickPlaceCounterToSink", "split": "pretrain", "source": "mimicgen"}],
    )
    cfg = dataclasses.replace(cfg, data=data)
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    raw = _data_loader.create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model)
    tds = _data_loader.transform_dataset(raw, data_config)
    return cfg, raw, tds


def episode_frame_starts(raw):
    tids = [int(x) for x in raw.trajectory_ids]
    tlens = [int(x) for x in raw.trajectory_lengths]
    starts = np.concatenate([[0], np.cumsum(tlens)[:-1]]).astype(np.int64)
    return {tid: (int(starts[p]), int(tlens[p])) for p, tid in enumerate(tids)}


def frame_indices(start, length, k):
    if length <= k:
        offs = list(range(length))
    else:
        offs = np.linspace(0, length - 1, k).round().astype(int).tolist()
    return [start + o for o in offs]


def stack_batch(tds, gidxs):
    samples = [tds[g] for g in gidxs]
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *samples)


def batch_to_inputs(batch):
    img = {}
    for k, v in batch["image"].items():
        a = np.asarray(v)
        if a.dtype == np.uint8:
            a = a.astype(np.float32) / 255.0 * 2.0 - 1.0
        img[k] = jnp.asarray(a)
    obs = _model.Observation(
        images=img,
        image_masks={k: jnp.asarray(v) for k, v in batch["image_mask"].items()},
        state=jnp.asarray(batch["state"]),
        tokenized_prompt=jnp.asarray(batch["tokenized_prompt"]) if "tokenized_prompt" in batch else None,
        tokenized_prompt_mask=jnp.asarray(batch["tokenized_prompt_mask"]) if "tokenized_prompt_mask" in batch else None,
    )
    return obs, jnp.asarray(batch["actions"])


# --------------------------------------------------------------------------- loss
def masked_flow_loss(model, rng, observation, actions, real_dim):
    """pi0 flow-matching MSE restricted to the first `real_dim` action dims.
    Verbatim from policy_analysis/influence_score.py (keeps numbers comparable)."""
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = _model.preprocess_observation(preprocess_rng, observation, train=False)
    batch_shape = actions.shape[:-2]
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    time_expanded = time[..., None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(observation, x_t, time)
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask = _pi0.make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1
    (_, suffix_out), _ = model.PaliGemma.llm(
        [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
    )
    v_t = model.action_out_proj(suffix_out[:, -model.action_horizon:])
    err = (v_t - u_t)[..., :real_dim]
    return jnp.mean(jnp.square(err))


def make_loss_fn(graphdef, real_dim):
    """jax.jit over the model STATE only, returning a scalar.

    NOT nnx.jit(model -> ...): nnx.jit threads the model through as both input AND
    output, so the jitted call's io footprint is 2x the params (~13 GB here) and blows
    the 0.12 memory cap. Splitting once and passing only `state` (params in, scalar out)
    halves it. `graphdef` is identical across checkpoints, so this compiles once.
    """
    @jax.jit
    def lfn(state, rng, obs, act):
        model = nnx.merge(graphdef, state)
        return masked_flow_loss(model, rng, obs, act, real_dim)
    return lfn


def crn_key(ep, m):
    """Common random numbers: depends only on (episode, draw index)."""
    return jax.random.fold_in(jax.random.fold_in(jax.random.key(SEED_BASE), int(ep)), int(m))


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--sets", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--k", type=int, default=8, help="frames per demo")
    ap.add_argument("--draws", type=int, default=8, help="independent (noise,time) draws per demo")
    ap.add_argument("--real_dim", type=int, default=12)
    ap.add_argument("--mg_dir", default=POOL_LEROBOT)
    ap.add_argument("--dtype", default="bfloat16",
                    help="param load dtype. Checkpoints are fp32 (~15GB) which does not fit the "
                         "0.12 JAX memory cap; the model's own compute dtype is bfloat16 anyway "
                         "(Pi0Config.dtype), so bf16 params are the natural choice. float16 is "
                         "available as a different-rounding control for the quantization check.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sets = {k: [int(x) for x in v] for k, v in json.load(open(args.sets)).items()}
    jobs = json.load(open(args.jobs))
    todo = [j for j in jobs if not os.path.exists(os.path.join(args.out_dir, j["out"]))]
    print(f"[loss_eval] jobs={len(jobs)} todo={len(todo)} sets={ {k: len(v) for k, v in sets.items()} }", flush=True)
    if not todo:
        print("[loss_eval] nothing to do", flush=True)
        return

    need = sorted({e for j in todo for e in sets[j["demos"]]})
    t0 = time.time()
    cfg, raw, tds = build_dataset(args.mg_dir)
    starts = episode_frame_starts(raw)
    print(f"[loss_eval] dataset built ({time.time()-t0:.0f}s), episodes={len(starts)}, "
          f"decoding {len(need)} demos x k={args.k} ...", flush=True)

    cache = {}
    t0 = time.time()
    for i, ep in enumerate(need):
        s, L = starts[ep]
        cache[ep] = stack_batch(tds, frame_indices(s, L, args.k))
        if (i + 1) % 100 == 0:
            print(f"    decoded {i+1}/{len(need)} ({time.time()-t0:.0f}s)", flush=True)
    b0 = cache[need[0]]
    nbytes = sum(np.asarray(v).nbytes for v in b0["image"].values())
    print(f"[loss_eval] decode done {time.time()-t0:.0f}s | cams={list(b0['image'])} "
          f"shape={ {k: np.asarray(v).shape for k, v in b0['image'].items()} } "
          f"act={np.asarray(b0['actions']).shape} | ~{nbytes/1e6:.1f} MB/demo -> "
          f"{len(need)*nbytes/1e9:.2f} GB cache", flush=True)

    lfn = None
    state = None
    for jn, job in enumerate(todo):
        eps = sets[job["demos"]]
        outp = os.path.join(args.out_dir, job["out"])
        del state                      # free the previous checkpoint BEFORE loading the next
        state = None
        gc.collect()
        t0 = time.time()
        params = _model.restore_params(f"{job['ckpt']}", dtype=jnp.dtype(args.dtype))
        model = cfg.model.load(params)
        del params
        graphdef, state = nnx.split(model)
        del model
        gc.collect()
        if lfn is None:
            lfn = make_loss_fn(graphdef, args.real_dim)
        tload = time.time() - t0
        t0 = time.time()
        M = np.zeros((len(eps), args.draws), np.float32)
        for i, ep in enumerate(eps):
            obs, act = batch_to_inputs(cache[ep])
            for m in range(args.draws):
                M[i, m] = float(lfn(state, crn_key(ep, m), obs, act))
        tcomp = time.time() - t0
        np.savez(outp, loss=M, episodes=np.asarray(eps, np.int64),
                 ckpt=job["ckpt"], demos=job["demos"], k=args.k, draws=args.draws,
                 dtype=args.dtype)
        print(f"[loss_eval] {jn+1}/{len(todo)} {job['out']}: load {tload:.0f}s compute {tcomp:.0f}s "
              f"({len(eps)} demos x {args.draws} draws) mean={M.mean():.5f}", flush=True)

    print("[loss_eval] DONE", flush=True)


if __name__ == "__main__":
    main()
