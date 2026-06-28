"""Influence-based pool selection for pi0 (LESS / TracIn), adapted to openpi.

Scores each MimicGen-pool demo by how much fine-tuning on it moves pi0 in a direction
that reduces loss on pi0's high-failure categories (the "targeted-10"), then selects the
top-N -> weakregion/influence_core.json {"core_episodes": [...]}.

Algorithm (LESS):
  * Reference checkpoints: average influence across a NEUTRAL reference -- the base-only LoRA
    warmup from pretrained pi0 (2k/4k/6k), the shared common prefix of every arm's fine-tune
    trajectory. NOT the core checkpoints (those bake in the failure-targeted selection -> biased).
  * Per-demo gradient g_t(z) = normalize( grad of the pi0 flow-matching loss w.r.t. the chosen
    trainable params at checkpoint t, over K frames sampled from demo z ). g(z) = mean_t g_t(z).
  * Failure direction g_val: D_val = held-out pool demos in the targeted-10 categories (held out
    from the candidate pool). Two variants (`--gval`):
      plain    : g_val = normalize(mean over D_val of g)            (textbook LESS)
      contrast : g_val = normalize(mean_hard - mean_ref)            (DEFAULT)
        where mean_ref is the mean gradient of a random pool sample. Subtracting the reference
        cancels the common-mode "generic pick-place" gradient that dominates every demo, leaving
        the failure-SPECIFIC direction -- and better matches the goal (demos that help the weak
        region beyond generic help).
  * Score: influence(z) = <g(z), g_val>. Select top-N.

Which gradient (`--mode_grad`): with both gemma variants LoRA, the trainable set is the LoRA
adapters + the ENTIRE SigLIP vision tower + the action-expert heads
(freeze_filter = All(.*llm.*, Not(.*lora.*))) -- ~hundreds of M params, so a faithful fixed
Gaussian projection matrix cannot be materialized; instead this scorer streams (build g_val first,
dot each candidate on-device, store only scalars -> no projection needed).
  lastlayer : grad of action_out_proj only (~33k dims). Cheap, but EMPIRICALLY WEAK here -- the
              action readout is grasp/motion-dominated and carries no object identity (smoke
              hard-vs-easy AUC ~0.44, no separation).
  lora      : grad of the LoRA adapters (~50M dims, the params actually updated in fine-tuning).
              DEFAULT. With --gval contrast this separates (smoke AUC ~0.63).
Robocasa pads the real 12-dim action to 32; --real_dim 12 masks the loss to real dims so the
zero-padded (noise-target) columns don't pollute the gradient.

Modes:
  probe : load 1 checkpoint, 1 episode -> print loss + grad shape/norm (integration sanity).
  smoke : build g_val from D_val, then score N_hard targeted + N_easy non-targeted held-out demos;
          report hard-vs-easy separation (plain vs contrast). Gate: hard > easy, AUC > 0.6.
  full  : score the whole candidate pool at all 4 checkpoints -> influence_core.json (top-N).

Run in the openpi env (GPU), e.g.:
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
  MUJOCO_GL=egl python policy_analysis/influence_score.py full \
    --mode_grad lora --gval contrast --real_dim 12 --k 8 --n_ref 300 --n_select 200 \
    --out weakregion/influence_core.json
"""
import argparse
import dataclasses
import json
import os
import re
import time
from collections import defaultdict

import numpy as np

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.models.model as _model
import openpi.models.pi0 as _pi0
import openpi.shared.nnx_utils as nnx_utils

REAL_ACTION_DIM = 12  # robocasa action dim; pi0 pads to action_dim=32 (RobocasaOutputs slices [:, :12])

TARGETED = ["juice", "spray", "pitcher", "canned_food", "soap_dispenser",
            "tupperware", "cheese_grater", "ice_cube", "cream_cheese_stick", "jar"]
# Reference checkpoints to differentiate at. MUST be NEUTRAL w.r.t. which 200 we select:
# the base-only LoRA warmup from pretrained pi0 (the shared common prefix of every arm's
# trajectory, in the correct fine-tuning regime). Do NOT use pi0_ppc2sink_core checkpoints --
# those already fine-tuned on a candidate selection (failure-targeted), so failure-region demos
# are absorbed there (low loss -> tiny/noisy normalized grads) and the score is biased.
CKPT_STEPS = [2000, 4000, 6000]
MG_DEFAULT = ("/data/xinyua11/robocasa_pkg/datasets/v1.0/pretrain/atomic/"
              "PickPlaceCounterToSink/20250819/mg/demo/2025-08-20-22-32-27/lerobot")
CKPT_BASE = "/data/xinyua11/openpi/checkpoints/pi0_ppc2sink_basewarmup/warmup_v1"
BASE_CONFIG = "pi0_ppc2sink_core"  # only for model arch + data transforms (same LoRA arch); ckpts loaded from CKPT_BASE


# ----------------------------------------------------------------------------- data / categories
def episode_categories(mg_dir):
    """episode_index -> normalized object category (same parse as build_arms.py)."""
    out = {}
    for line in open(os.path.join(mg_dir, "meta", "episodes.jsonl")):
        e = json.loads(line)
        obj = None
        for t in e.get("tasks", []):
            m = re.match(r"Pick the (.+?) from the counter", t)
            if m:
                obj = str(m.group(1)).strip().lower().replace(" ", "_")
                break
        out[e["episode_index"]] = obj
    return out


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
    """episode_index -> (global_start_frame, length). Global index ordering matches all_steps."""
    tids = [int(x) for x in raw.trajectory_ids]
    tlens = [int(x) for x in raw.trajectory_lengths]
    starts = np.concatenate([[0], np.cumsum(tlens)[:-1]]).astype(np.int64)
    return {tid: (int(starts[p]), int(tlens[p])) for p, tid in enumerate(tids)}


def frame_indices(start, length, k):
    """K frames evenly spaced across [0, length-1] (global indices)."""
    if length <= k:
        offs = list(range(length))
    else:
        offs = np.linspace(0, length - 1, k).round().astype(int).tolist()
    return [start + o for o in offs]


def stack_batch(tds, gidxs):
    """Decode+transform K frames for one demo, stack to a numpy batch dict (images kept uint8)."""
    samples = [tds[g] for g in gidxs]
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *samples)


def batch_to_inputs(batch):
    """numpy batch -> (Observation, actions) as jax arrays WITHOUT mutating `batch`."""
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
    act = jnp.asarray(batch["actions"])
    return obs, act


# ----------------------------------------------------------------------------- model / gradients
def load_model(cfg, step):
    params = _model.restore_params(f"{CKPT_BASE}/{step}/params")  # keeps checkpoint dtypes
    return cfg.model.load(params)


def masked_flow_loss(model, rng, observation, actions, real_dim):
    """pi0 flow-matching MSE restricted to the first `real_dim` action dims (drop zero-padding).

    Replicates Pi0.compute_loss (pi0.py:189) but slices the (v_t - u_t) residual to real dims so
    the ~20 zero-padded action columns (whose target is pure noise) don't swamp the gradient.
    """
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


def make_grad_fn(mode, real_dim=0):
    if mode == "lastlayer":
        filt = nnx.All(nnx.Param, nnx_utils.PathRegex(r".*action_out_proj.*"))
    elif mode == "heads":  # small action-expert heads (no llm, no vision tower)
        filt = nnx.All(nnx.Param,
                       nnx.Not(nnx_utils.PathRegex(r".*llm.*")),
                       nnx.Not(nnx_utils.PathRegex(r".*img.*")))
    elif mode == "lora":  # the LoRA adapters that actually get updated during fine-tuning
        filt = nnx.All(nnx.Param, nnx_utils.PathRegex(r".*lora.*"))
    elif mode == "lora_heads":  # LoRA adapters + action-expert output heads
        filt = nnx.All(nnx.Param,
                       nnx.Any(nnx_utils.PathRegex(r".*lora.*"),
                               nnx.All(nnx.Not(nnx_utils.PathRegex(r".*llm.*")),
                                       nnx.Not(nnx_utils.PathRegex(r".*img.*")))))
    else:
        raise ValueError(mode)

    if real_dim and real_dim > 0:
        def loss_fn(model, rng, obs, act):
            return masked_flow_loss(model, rng, obs, act, real_dim)
    else:
        def loss_fn(model, rng, obs, act):
            return jnp.mean(model.compute_loss(rng, obs, act, train=False))

    diff = nnx.DiffState(0, filt)

    @nnx.jit
    def gfn(model, rng, obs, act):
        grads = nnx.grad(loss_fn, argnums=diff)(model, rng, obs, act)
        leaves = [l.value if hasattr(l, "value") else l for l in jax.tree.leaves(grads)]
        return jnp.concatenate([jnp.ravel(l).astype(jnp.float32) for l in leaves])

    return gfn


def grad_unit(gfn, model, ep, step, obs, act):
    """Normalized gradient vector for one demo at one checkpoint (numpy float32)."""
    rng = jax.random.fold_in(jax.random.key(int(ep)), int(step))
    g = np.asarray(gfn(model, rng, obs, act))
    n = float(np.linalg.norm(g))
    return g / n if n > 0 else g


# ----------------------------------------------------------------------------- splits
def make_splits(mg_dir, base_file, seed, dval_mode="targeted", n_val_per_cat=8, n_val=120):
    """Build D_val, the candidate pool (categorized - base - D_val), and the category map.

    dval_mode:
      targeted : D_val = held-out demos from the targeted-10 failure categories
                 (FAILURE-region influence; up to n_val_per_cat per category).
      balanced : D_val = CLASS-BALANCED held-out sample across ALL object categories -- up to
                 n_val_per_cat per category, capped by availability (VALUE influence). Mirrors the
                 balanced/stratified eval (per-category average), NOT the pool frequency.
      uniform  : D_val = a frequency-uniform random sample of n_val held-out demos over the whole
                 pool (over-weights common categories; kept for ablation, not the default value run).
    """
    ep_cat = episode_categories(mg_dir)
    base = set()
    if base_file and os.path.exists(base_file):
        b = json.load(open(base_file))
        base = set(b["base_episodes"] if isinstance(b, dict) else b)
    rng = np.random.RandomState(seed)
    pool = sorted(ep for ep, c in ep_cat.items() if c is not None and ep not in base)
    by_cat = defaultdict(list)
    for ep in pool:
        by_cat[ep_cat[ep]].append(ep)
    if dval_mode == "uniform":
        dval = sorted(int(x) for x in rng.choice(pool, size=min(n_val, len(pool)), replace=False))
    elif dval_mode == "balanced":
        dval = []
        for c in sorted(by_cat):  # ALL categories, equal-ish, capped by availability
            avail = sorted(by_cat[c])
            dval += list(rng.choice(avail, size=min(n_val_per_cat, len(avail)), replace=False))
        dval = sorted(int(x) for x in dval)
    elif dval_mode == "targeted":
        dval = []
        for c in TARGETED:
            avail = sorted(by_cat.get(c, []))
            k = min(n_val_per_cat, max(1, len(avail) // 3))
            dval += list(rng.choice(avail, size=min(k, len(avail)), replace=False))
        dval = sorted(int(x) for x in dval)
    else:
        raise ValueError(dval_mode)
    dval_set = set(dval)
    candidates = [ep for ep in pool if ep not in dval_set]
    return ep_cat, base, dval, candidates


# ----------------------------------------------------------------------------- runners
def run_probe(args):
    cfg, raw, tds = build_dataset(args.mg_dir)
    starts = episode_frame_starts(raw)
    ep = sorted(starts)[0]
    s, L = starts[ep]
    print(f"[probe] dataset len={len(raw)} episodes={len(starts)} | ep {ep} start={s} len={L}", flush=True)
    gidxs = frame_indices(s, L, args.k)
    t0 = time.time()
    batch = stack_batch(tds, gidxs)
    print(f"[probe] decoded {args.k} frames in {time.time()-t0:.2f}s | keys={list(batch)} "
          f"img_dtype={batch['image'][list(batch['image'])[0]].dtype} "
          f"img_shape={batch['image'][list(batch['image'])[0]].shape} act_shape={batch['actions'].shape}", flush=True)
    obs, act = batch_to_inputs(batch)
    t0 = time.time()
    model = load_model(cfg, CKPT_STEPS[0])
    print(f"[probe] loaded ckpt {CKPT_STEPS[0]} in {time.time()-t0:.1f}s", flush=True)
    gfn = make_grad_fn(args.mode_grad, args.real_dim)
    t0 = time.time()
    g = np.asarray(gfn(model, jax.random.key(0), obs, act))
    print(f"[probe] grad[{args.mode_grad}] dim={g.size} norm={np.linalg.norm(g):.4f} "
          f"(first call {time.time()-t0:.1f}s incl compile)", flush=True)
    t0 = time.time()
    _ = np.asarray(gfn(model, jax.random.key(1), obs, act))
    print(f"[probe] second grad call {time.time()-t0:.3f}s | OK", flush=True)


def _score_set(gfn, models, steps, cache):
    """g(z)=mean_t normalize(grad_t); returns {ep: numpy vec} averaged over checkpoints.
    cache holds CPU numpy batches; inputs are moved to device per use."""
    acc = {ep: None for ep in cache}
    for step, model in zip(steps, models):
        for ep, batch in cache.items():
            obs, act = batch_to_inputs(batch)
            g = grad_unit(gfn, model, ep, step, obs, act)
            acc[ep] = g if acc[ep] is None else acc[ep] + g
    return {ep: v / len(steps) for ep, v in acc.items()}


def _decode_cache(tds, starts, eps, k):
    """Decode K frames/demo into CPU numpy batches (uint8 images), reused across checkpoints."""
    cache = {}
    for i, ep in enumerate(eps):
        s, L = starts[ep]
        cache[ep] = stack_batch(tds, frame_indices(s, L, k))
        if (i + 1) % 200 == 0:
            print(f"    decoded {i+1}/{len(eps)}", flush=True)
    return cache


def _auc_report(tag, hs, es):
    auc = np.mean([h > e for h in hs for e in es])
    print(f"[smoke:{tag}] hard={hs.mean():+.4f}±{hs.std():.4f}  easy={es.mean():+.4f}±{es.std():.4f}  "
          f"Δ={hs.mean()-es.mean():+.4f}  AUC(hard>easy)={auc:.3f}  "
          f"{'PASS' if (hs.mean()>es.mean() and auc>0.6) else 'WEAK'}", flush=True)
    return auc


def run_smoke(args):
    cfg, raw, tds = build_dataset(args.mg_dir)
    starts = episode_frame_starts(raw)
    ep_cat, base, dval, candidates = make_splits(args.mg_dir, args.base_file, args.seed, args.dval, args.n_val_per_cat, args.n_val)
    rng = np.random.RandomState(args.seed + 1)
    hard_all = [e for e in candidates if ep_cat[e] in TARGETED]
    easy_all = [e for e in candidates if ep_cat[e] not in TARGETED]
    hard_test = sorted(rng.choice(hard_all, size=min(args.n_test, len(hard_all)), replace=False).tolist())
    easy_test = sorted(rng.choice(easy_all, size=min(args.n_test, len(easy_all)), replace=False).tolist())
    # reference set for the common-mode: random pool demos disjoint from val/test
    used = set(dval) | set(hard_test) | set(easy_test)
    ref_pool = [e for e in candidates if e not in used]
    ref = sorted(rng.choice(ref_pool, size=min(args.n_ref, len(ref_pool)), replace=False).tolist())
    steps = [int(s) for s in args.steps.split(",")]
    print(f"[smoke] D_val={len(dval)} ref={len(ref)} hard_test={len(hard_test)} easy_test={len(easy_test)} "
          f"ckpts={steps} k={args.k} mode={args.mode_grad}", flush=True)

    print("[smoke] decoding ...", flush=True)
    val_cache = _decode_cache(tds, starts, dval, args.k)
    ref_cache = _decode_cache(tds, starts, ref, args.k)
    hard_cache = _decode_cache(tds, starts, hard_test, args.k)
    easy_cache = _decode_cache(tds, starts, easy_test, args.k)

    gfn = make_grad_fn(args.mode_grad, args.real_dim)
    models = []
    for s in steps:
        t0 = time.time()
        models.append(load_model(cfg, s))
        print(f"[smoke] loaded ckpt {s} ({time.time()-t0:.1f}s)", flush=True)

    val_vecs = _score_set(gfn, models, steps, val_cache)
    ref_vecs = _score_set(gfn, models, steps, ref_cache)
    hard_vecs = _score_set(gfn, models, steps, hard_cache)
    easy_vecs = _score_set(gfn, models, steps, easy_cache)

    mean_hard = np.mean(np.stack(list(val_vecs.values())), axis=0)
    mean_ref = np.mean(np.stack(list(ref_vecs.values())), axis=0)
    H = np.stack([hard_vecs[e] for e in hard_test])
    E = np.stack([easy_vecs[e] for e in easy_test])

    # (1) plain LESS: g_val = normalize(mean_hard)
    g_plain = mean_hard / (np.linalg.norm(mean_hard) + 1e-12)
    _auc_report("plain", H @ g_plain, E @ g_plain)

    # (2) contrast: g_val = normalize(mean_hard - mean_ref) -> failure-SPECIFIC direction
    gc = mean_hard - mean_ref
    gc = gc / (np.linalg.norm(gc) + 1e-12)
    _auc_report("contrast", H @ gc, E @ gc)

    # (3) contrast + center the demo grads by the common mode too
    _auc_report("contrast+center", (H - mean_ref) @ gc, (E - mean_ref) @ gc)


def _unit(g):
    """L2-normalize on device."""
    n = jnp.linalg.norm(g)
    return g / jnp.where(n > 0, n, 1.0)


def _accumulate_mean(gfn, model, step, cache):
    """Sum of unit grads over a cache at one checkpoint (device array)."""
    acc = None
    for ep, batch in cache.items():
        obs, act = batch_to_inputs(batch)
        rng = jax.random.fold_in(jax.random.key(int(ep)), int(step))
        u = _unit(gfn(model, rng, obs, act))
        acc = u if acc is None else acc + u
    return acc


def run_full(args):
    cfg, raw, tds = build_dataset(args.mg_dir)
    starts = episode_frame_starts(raw)
    ep_cat, base, dval, candidates = make_splits(args.mg_dir, args.base_file, args.seed, args.dval, args.n_val_per_cat, args.n_val)
    steps = [int(s) for s in args.steps.split(",")]
    rng_np = np.random.RandomState(args.seed + 7)
    use_ref = args.gval == "contrast"  # plain g_val (VALUE influence) doesn't need a reference
    ref = (sorted(rng_np.choice([e for e in candidates if e not in set(dval)],
                                size=min(args.n_ref, len(candidates)), replace=False).tolist())
           if use_ref else [])
    if args.limit:
        candidates = candidates[:args.limit]
    print(f"[full] mode={args.mode_grad} gval={args.gval} dval={args.dval} D_val={len(dval)} ref={len(ref)} "
          f"candidates={len(candidates)} ckpts={steps} k={args.k} real_dim={args.real_dim}", flush=True)

    print("[full] decoding D_val + ref ...", flush=True)
    val_cache = _decode_cache(tds, starts, dval, args.k)
    ref_cache = _decode_cache(tds, starts, ref, args.k) if use_ref else {}
    print("[full] decoding candidate pool (slow) ...", flush=True)
    cand_cache = _decode_cache(tds, starts, candidates, args.k)

    gfn = make_grad_fn(args.mode_grad, args.real_dim)

    # ---- pass 1: build g_val direction (D_val mean; + reference mean for contrast) ----
    acc_hard = acc_ref = None
    for s in steps:
        t0 = time.time()
        model = load_model(cfg, s)
        ah = _accumulate_mean(gfn, model, s, val_cache)
        acc_hard = ah if acc_hard is None else acc_hard + ah
        if use_ref:
            ar = _accumulate_mean(gfn, model, s, ref_cache)
            acc_ref = ar if acc_ref is None else acc_ref + ar
        del model
        print(f"[full] pass1 ckpt {s} ({time.time()-t0:.0f}s)", flush=True)
    mean_hard = acc_hard / (len(steps) * len(dval))
    g_val = (mean_hard - acc_ref / (len(steps) * len(ref))) if use_ref else mean_hard
    g_val = _unit(g_val)
    g_val = jax.device_put(g_val)  # keep resident
    print(f"[full] g_val built (dim={g_val.size}); scoring candidates ...", flush=True)

    # ---- pass 2: score each candidate = (1/T) sum_t <unit grad_t(z), g_val> ----
    score_sum = defaultdict(float)
    for s in steps:
        t0 = time.time()
        model = load_model(cfg, s)
        for i, (ep, batch) in enumerate(cand_cache.items()):
            obs, act = batch_to_inputs(batch)
            rng = jax.random.fold_in(jax.random.key(int(ep)), int(s))
            u = _unit(gfn(model, rng, obs, act))
            score_sum[ep] += float(jnp.vdot(u, g_val))
            if (i + 1) % 500 == 0:
                print(f"    pass2 ckpt {s}: {i+1}/{len(candidates)} ({time.time()-t0:.0f}s)", flush=True)
        del model
        # crash-safety: dump partial scores after each checkpoint
        json.dump({str(e): score_sum[e] for e in candidates}, open(args.out + ".partial", "w"))
        print(f"[full] pass2 ckpt {s} done ({time.time()-t0:.0f}s)", flush=True)

    scores = {ep: score_sum[ep] / len(steps) for ep in candidates}
    order = sorted(candidates, key=lambda e: -scores[e])
    selected = sorted(int(e) for e in order[:args.n_select])
    sel_mix = defaultdict(int)
    for e in selected:
        sel_mix[ep_cat[e]] += 1
    out = {
        "method": f"influence_{args.mode_grad}_{args.gval}",
        "n_select": len(selected),
        "ckpt_steps": steps,
        "k_frames": args.k,
        "real_dim": args.real_dim,
        "gval": args.gval,
        "dval": args.dval,
        "ckpt_base": CKPT_BASE,
        "n_dval": len(dval),
        "n_ref": len(ref),
        "dval_episodes": dval,
        "core_episodes": selected,
        "score_range": [scores[order[-1]], scores[order[0]]],
        "selected_targeted_frac": sum(1 for e in selected if ep_cat[e] in TARGETED) / len(selected),
        "selected_mix": dict(sorted(sel_mix.items(), key=lambda x: -x[1])),
        "all_scores": {str(e): scores[e] for e in candidates},
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n[full] wrote {args.out}: top {len(selected)}; "
          f"targeted_frac={out['selected_targeted_frac']:.3f} score_range={out['score_range']}")
    print(f"[full] selected mix (top 15): {dict(list(out['selected_mix'].items())[:15])}")


# ----------------------------------------------------------------------------- value smoke
def _grad_filter(mode):
    if mode == "lastlayer":
        return nnx.All(nnx.Param, nnx_utils.PathRegex(r".*action_out_proj.*"))
    if mode == "lora":
        return nnx.All(nnx.Param, nnx_utils.PathRegex(r".*lora.*"))
    if mode == "heads":
        return nnx.All(nnx.Param, nnx.Not(nnx_utils.PathRegex(r".*llm.*")), nnx.Not(nnx_utils.PathRegex(r".*img.*")))
    if mode == "lora_heads":
        return nnx.All(nnx.Param, nnx.Any(nnx_utils.PathRegex(r".*lora.*"),
                       nnx.All(nnx.Not(nnx_utils.PathRegex(r".*llm.*")), nnx.Not(nnx_utils.PathRegex(r".*img.*")))))
    raise ValueError(mode)


def _scalar_loss(real_dim):
    if real_dim and real_dim > 0:
        return lambda model, rng, obs, act: masked_flow_loss(model, rng, obs, act, real_dim)
    return lambda model, rng, obs, act: jnp.mean(model.compute_loss(rng, obs, act, train=False))


def run_vsmoke(args):
    """Value-influence sanity: do the TOP-scored demos, applied as a 1-step SGD update, reduce
    D_val loss MORE than random demos? Validates the influence sign/scale + the pipeline. D_val is
    a uniform held-out sample (the test distribution), g_val = plain mean (VALUE, not failure)."""
    cfg, raw, tds = build_dataset(args.mg_dir)
    starts = episode_frame_starts(raw)
    ep_cat, base, dval, candidates = make_splits(args.mg_dir, args.base_file, args.seed,
                                                 args.dval, args.n_val_per_cat, args.n_val)
    rng = np.random.RandomState(args.seed + 3)
    cand_sample = sorted(rng.choice(candidates, size=min(args.vsmoke_cand, len(candidates)), replace=False).tolist())
    step = args.vsmoke_ckpt
    print(f"[vsmoke] dval({args.dval})={len(dval)} cand_sample={len(cand_sample)} ckpt={step} "
          f"group_k={args.group_k} lr={args.lr} mode={args.mode_grad}", flush=True)

    val_cache = _decode_cache(tds, starts, dval, args.k)
    cand_cache = _decode_cache(tds, starts, cand_sample, args.k)

    gfn = make_grad_fn(args.mode_grad, args.real_dim)              # flat unit grads -> scoring
    filt = _grad_filter(args.mode_grad)
    loss_fn = _scalar_loss(args.real_dim)
    diff = nnx.DiffState(0, filt)

    @nnx.jit
    def grad_state(model, rng, obs, act):                         # nnx grad State -> 1-step update
        return nnx.grad(loss_fn, argnums=diff)(model, rng, obs, act)

    @nnx.jit
    def loss_at(model, rng, obs, act):
        return loss_fn(model, rng, obs, act)

    # 1) score candidates by value = <unit grad(z), g_val>, g_val = plain mean over D_val (1 ckpt)
    model = load_model(cfg, step)
    val_vecs = {}
    for ep, batch in val_cache.items():
        obs, act = batch_to_inputs(batch)
        val_vecs[ep] = grad_unit(gfn, model, ep, step, obs, act)
    g_val = np.mean(np.stack(list(val_vecs.values())), axis=0)
    g_val = g_val / (np.linalg.norm(g_val) + 1e-12)
    scores = {}
    for ep, batch in cand_cache.items():
        obs, act = batch_to_inputs(batch)
        scores[ep] = float(grad_unit(gfn, model, ep, step, obs, act) @ g_val)
    order = sorted(cand_sample, key=lambda e: -scores[e])
    top_group = order[:args.group_k]
    bot_group = order[-args.group_k:]

    # baseline D_val loss
    def dval_loss(m):
        tot = 0.0
        for ep, batch in val_cache.items():
            obs, act = batch_to_inputs(batch)
            tot += float(loss_at(m, jax.random.fold_in(jax.random.key(int(ep)), 999), obs, act))
        return tot / len(val_cache)

    base_loss = dval_loss(model)
    print(f"[vsmoke] base D_val loss = {base_loss:.5f}", flush=True)

    # group mean gradient (batched, 2 frames/demo) -> SGD step -> measure D_val loss delta.
    # normalize=True scales the group gradient to unit norm first, so EVERY group takes the same
    # step size -> the test isolates DIRECTION (what the cosine score selects on), removing the
    # confound that high-cosine demos can have small gradient norms (tiny raw updates).
    def group_update_delta(eps, normalize):
        gi = []
        for ep in eps:
            s, L = starts[ep]
            gi += frame_indices(s, L, 2)               # 2 frames/demo keeps the batch modest
        batch = stack_batch(tds, gi)
        obs, act = batch_to_inputs(batch)
        m = load_model(cfg, step)                      # fresh copy
        g = grad_state(m, jax.random.key(int(eps[0])), obs, act)
        if normalize:
            gnorm = optax.global_norm(g)
            g = jax.tree.map(lambda x: x / (gnorm + 1e-12), g)
        params = nnx.state(m, filt)
        tx = optax.sgd(args.lr)
        updates, _ = tx.update(g, tx.init(params), params)
        nnx.update(m, optax.apply_updates(params, updates))
        return dval_loss(m) - base_loss

    for normalize in (True, False):
        tag = "unit-norm step (DIRECTION test)" if normalize else "raw 1-step update (sign+scale)"
        d_top = group_update_delta(top_group, normalize)
        d_bot = group_update_delta(bot_group, normalize)
        d_rand = np.array([group_update_delta(rng.choice(cand_sample, size=args.group_k, replace=False).tolist(),
                                              normalize) for _ in range(args.n_rand_groups)])
        better = (d_top < d_rand.mean()) and (d_top < d_bot)
        print(f"\n[vsmoke] ΔD_val loss -- {tag} (negative = improvement):")
        print(f"  TOP-{args.group_k} value : {d_top:+.6f}")
        print(f"  BOTTOM-{args.group_k}    : {d_bot:+.6f}")
        print(f"  random groups     : mean {d_rand.mean():+.6f}  (min {d_rand.min():+.6f}, max {d_rand.max():+.6f})")
        print(f"  -> {'PASS' if better else 'WEAK'} (top better than random AND than bottom)")


def main():
    global CKPT_BASE  # declared up-front: --ckpt_base default reads it, then we rebind it below
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["probe", "smoke", "vsmoke", "full"])
    ap.add_argument("--mg_dir", default=MG_DEFAULT)
    ap.add_argument("--base_file", default="weakregion/arms.json")
    ap.add_argument("--mode_grad", choices=["lastlayer", "heads", "lora", "lora_heads"], default="lastlayer")
    ap.add_argument("--real_dim", type=int, default=REAL_ACTION_DIM,
                    help="mask flow loss to first N action dims (0=all 32, incl. zero-padding)")
    ap.add_argument("--k", type=int, default=8, help="frames sampled per demo")
    ap.add_argument("--ckpt_base", default=CKPT_BASE,
                    help="dir holding the NEUTRAL reference checkpoints (<step>/params); default = base warmup")
    ap.add_argument("--steps", default=",".join(str(s) for s in CKPT_STEPS))
    ap.add_argument("--dval", choices=["targeted", "balanced", "uniform"], default="targeted",
                    help="D_val: targeted=failure categories (failure influence); "
                         "balanced=class-balanced over ALL categories (VALUE influence, matches "
                         "stratified eval); uniform=frequency-uniform sample (ablation)")
    ap.add_argument("--n_val_per_cat", type=int, default=8, help="dval=targeted: held-out demos/category")
    ap.add_argument("--n_val", type=int, default=120, help="dval=uniform: held-out demos in D_val")
    ap.add_argument("--n_test", type=int, default=100, help="smoke: hard/easy test demos each")
    ap.add_argument("--n_ref", type=int, default=100, help="random reference demos (common-mode)")
    ap.add_argument("--gval", choices=["plain", "contrast"], default="contrast",
                    help="full: g_val direction (contrast = mean_hard - mean_ref, cancels common mode; "
                         "plain = mean over D_val, for VALUE influence)")
    ap.add_argument("--n_select", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="full: cap candidates (debug)")
    ap.add_argument("--out", default="weakregion/influence_core.json")
    # vsmoke (value 1-step loss-reduction test) knobs:
    ap.add_argument("--vsmoke_ckpt", type=int, default=10000, help="vsmoke: checkpoint to update")
    ap.add_argument("--vsmoke_cand", type=int, default=300, help="vsmoke: candidates to score")
    ap.add_argument("--group_k", type=int, default=25, help="vsmoke: demos per update group")
    ap.add_argument("--lr", type=float, default=0.05, help="vsmoke: 1-step SGD lr")
    ap.add_argument("--n_rand_groups", type=int, default=5, help="vsmoke: random groups to compare")
    args = ap.parse_args()
    CKPT_BASE = args.ckpt_base  # load_model() reads this module global (declared global at top)
    print(f"[ref] differentiating at checkpoints: {CKPT_BASE} steps={args.steps}", flush=True)
    if args.mode == "probe":
        run_probe(args)
    elif args.mode == "smoke":
        run_smoke(args)
    elif args.mode == "vsmoke":
        run_vsmoke(args)
    else:
        run_full(args)


if __name__ == "__main__":
    main()
