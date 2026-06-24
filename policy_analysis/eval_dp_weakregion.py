"""Evaluate a trained Diffusion Policy on SPECIFIC robocasa seeds, with per-episode
init-state logging -> weak-region (tall vs short object) breakdown.

Why this exists: the DP fork's eval_robocasa labels rollouts with cosmetic seeds
(test_start_seed+i) but actually seeds the env ONCE at creation, so it can't target
our frozen stratified eval_set. This drives ONE env per seed via gym.make(seed=S)
(identical scene to build_eval_set / our pi0 weak-region harness) while reusing the
DP fork's RobomimicImageWrapper+MultiStepWrapper so the policy's obs/action handling
is byte-identical to the real eval.

Run in the robocasa env, MUJOCO_GL=egl:
  MUJOCO_GL=egl python policy_analysis/eval_dp_weakregion.py \
      -c checkpoints/dp_h100/seed0/epoch=0400-...ckpt \
      --eval_set phase1_data/eval_set.json --out weakregion/dp_seed0_ep400
"""
import argparse
import json
import os
import numpy as np
import torch
import dill
import gymnasium as gym
import robocasa  # noqa: F401 (registers envs)
import hydra

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.common.pytorch_util import dict_apply
from robomimic.utils.lang_utils import LangEncoder
from robocasa.utils.dataset_registry_utils import get_task_horizon

HEIGHT_THRESH = 0.10  # tall (weak region) if obj_height > thresh; from the n=150 analysis


def load_policy(ckpt, device):
    payload = torch.load(open(ckpt, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir="/tmp/dp_eval_ws")
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(device)
    policy.eval()
    return policy, cfg


def obj_descriptor(rs):
    ep = rs.get_ep_meta()
    cat = next((c.get("info", {}).get("cat") for c in ep.get("object_cfgs", [])
                if c.get("name") == "obj"), "unknown")
    try:
        o = rs.objects["obj"]
        h = float(o.top_offset[2] - o.bottom_offset[2])
    except Exception:
        h = None
    return cat, h, ep.get("layout_id"), ep.get("style_id")


def rollout_seed(policy, cfg, task, split, seed, device, horizon, shared_lang_encoder, k_unc=0):
    base = gym.make(f"robocasa/{task}", split=split, seed=seed)
    wrapped = RobomimicImageWrapper(
        env=base, shape_meta=cfg.task.shape_meta, init_state=None,
        render_obs_key=cfg.task.env_runner.render_obs_key)
    wrapped.lang_encoder = shared_lang_encoder  # reuse one encoder across seeds
    env = MultiStepWrapper(
        wrapped, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps,
        max_episode_steps=horizon)

    obs = env.reset()
    cat, h, lay, sty = obj_descriptor(base.unwrapped)

    success = False
    uncertainty = None  # K-sample action-variance at the FIRST state (epistemic; AUC-0.77 signal)
    steps = 0
    done = False
    first = True
    while not done and steps < horizon:
        obs_dict = dict_apply(obs, lambda x: torch.from_numpy(x[None]).to(
            device=device, dtype=torch.float32))
        with torch.no_grad():
            if first and k_unc:
                samples = [policy.predict_action(obs_dict)["action"].detach().cpu().numpy()
                           for _ in range(k_unc)]
                uncertainty = float(np.stack(samples, 0).std(axis=0).mean())
                action_dict = {"action": torch.from_numpy(samples[-1]).to(device)}
            else:
                action_dict = policy.predict_action(obs_dict)
        first = False
        action = action_dict["action"][0].detach().cpu().numpy()  # (n_action_steps, adim)
        obs, reward, done, info = env.step(action)
        steps += cfg.n_action_steps
        if float(np.asarray(reward).max()) > 0:
            success = True
    env.close()
    return dict(seed=int(seed), object_category=cat,
                obj_height=(round(h, 4) if h is not None else None),
                layout_id=lay, style_id=sty, success=bool(success),
                uncertainty=(round(uncertainty, 5) if uncertainty is not None else None))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--checkpoint", required=True)
    p.add_argument("-t", "--task", default="PickPlaceCounterToSink")
    p.add_argument("-s", "--split", default="pretrain")
    p.add_argument("--eval_set", default="phase1_data/eval_set.json")
    p.add_argument("--pool_start", type=int, default=None,
                   help="contiguous gym.make seed pool: run seeds [pool_start, pool_start+pool_n)")
    p.add_argument("--pool_n", type=int, default=None)
    p.add_argument("--out", default="weakregion/dp_eval")
    p.add_argument("-d", "--device", default="cuda:0")
    p.add_argument("-k", "--k_unc", type=int, default=0,
                   help="K action samples at first state -> per-episode uncertainty (0=off)")
    p.add_argument("--limit", type=int, default=None, help="only first N seeds (smoke test)")
    args = p.parse_args()

    if args.pool_n is not None:
        # contiguous gym.make pool (same scenes pi0 n=150 saw, when start=0) — region
        # tag is assigned post-hoc from the OBSERVED height, not eval_set (seeds differ).
        start = args.pool_start or 0
        seeds = [(s, None) for s in range(start, start + args.pool_n)]
    else:
        ev = json.load(open(args.eval_set))
        seeds = [(r["seed"], None) for r in ev["weak_seeds"] + ev["other_seeds"]]
        if args.limit:
            seeds = [(r["seed"], None) for r in
                     ev["weak_seeds"][:args.limit] + ev["other_seeds"][:args.limit]]

    device = torch.device(args.device)
    policy, cfg = load_policy(args.checkpoint, device)
    horizon = int(get_task_horizon(task=args.task))
    lang_encoder = LangEncoder("cpu")  # one shared encoder

    os.makedirs(args.out, exist_ok=True)
    results = []
    for i, (seed, tag) in enumerate(seeds):
        rec = rollout_seed(policy, cfg, args.task, args.split, seed, device,
                           horizon, lang_encoder, k_unc=args.k_unc)
        # region from OBSERVED height (eval_set tags don't match gym.make scenes)
        rec["region"] = "tall" if (rec["obj_height"] or 0) > HEIGHT_THRESH else "short"
        results.append(rec)
        print(f"[{i+1}/{len(seeds)}] seed={seed} {tag} obj={rec['object_category']} "
              f"h={rec['obj_height']} -> {'SUCCESS' if rec['success'] else 'fail'}", flush=True)
        json.dump({"checkpoint": args.checkpoint, "task": args.task, "n": len(results),
                   "results": results}, open(os.path.join(args.out, "dp_eval.json"), "w"), indent=2)

    # weak-region summary
    def rate(tag):
        rs = [r["success"] for r in results if r["region"] == tag]
        return (sum(rs), len(rs), (sum(rs) / len(rs) if rs else float("nan")))
    ts, tn, tr = rate("tall")
    ss, sn, sr = rate("short")
    alls = sum(r["success"] for r in results)
    print(f"\n==== DP weak-region eval ====")
    print(f"overall: {alls}/{len(results)} = {alls/len(results):.1%}")
    print(f"tall  (weak): {ts}/{tn} = {tr:.1%}")
    print(f"short:        {ss}/{sn} = {sr:.1%}")


if __name__ == "__main__":
    main()
