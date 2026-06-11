"""Probe: does DP's action-sampling variance (epistemic uncertainty) predict failure?

The core-algorithm question: the acquisition function's third term is student
epistemic uncertainty. Diffusion policies are generative — sampling K action
chunks at the same observation and measuring their spread is the natural
uncertainty estimate. This probe wraps the policy used by the official eval so
that at the START of each episode it draws K extra samples and records their
variance; per-episode success then comes from the eval_log. If high start-state
variance predicts failure (AUC >> 0.5), the uncertainty term is validated.

Run in the robocasa env on a free GPU:
  MUJOCO_GL=egl python probe_dp_uncertainty.py -c <ckpt> -n 50
"""
import argparse
import json
import math
import os

import numpy as np
import torch

from eval_robocasa import eval_task  # noqa: F401  (env setup path)
import eval_robocasa


class UncertaintyProbeWrapper:
    """Wraps the DP policy; on each episode's FIRST predict_action call, draws
    K extra samples and records their std. Episode boundaries are detected by
    call count: with max_steps=H and n_action_steps=A, there are
    ceil(H/A) calls per episode (n_envs=1)."""

    def __init__(self, policy, calls_per_episode, k=8):
        self._p = policy
        self._cpe = calls_per_episode
        self._k = k
        self._calls = 0
        self.episode_uncertainty = []

    def __getattr__(self, name):  # delegate everything else
        return getattr(self._p, name)

    def reset(self):
        return self._p.reset()

    def predict_action(self, obs_dict):
        if self._calls % self._cpe == 0:  # episode start
            samples = []
            with torch.no_grad():
                for _ in range(self._k):
                    out = self._p.predict_action(obs_dict)
                    samples.append(out["action"].detach().cpu().numpy())
            A = np.stack(samples, 0)  # (K, B, T, D)
            # mean over batch/time/dims of the per-coordinate std across samples
            unc = float(A.std(axis=0).mean())
            self.episode_uncertainty.append(unc)
            self._calls += 1
            return out  # use the last sample as the executed action
        self._calls += 1
        return self._p.predict_action(obs_dict)


def auc(scores, y):
    s, y = np.asarray(scores, float), np.asarray(y, float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--checkpoint", required=True)
    p.add_argument("-t", "--task", default="PickPlaceCounterToSink")
    p.add_argument("-s", "--split", default="pretrain")
    p.add_argument("-n", "--num_rollouts", type=int, default=50)
    p.add_argument("-k", "--k_samples", type=int, default=8)
    p.add_argument("-o", "--output_dir",
                   default="/home/asurite.ad.asu.edu/xinyua11/robocasa_experiments/weakregion/dp_uncertainty_probe")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # patch eval_robocasa's workspace pipeline to wrap the policy.
    # eval_task loads the policy then builds env_runner; easiest interception:
    # monkeypatch hydra instantiate of the runner? Simpler: replicate eval_task
    # body here with the wrapper inserted.
    import copy
    import dill
    import hydra
    import pathlib
    from omegaconf import OmegaConf
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    payload = torch.load(open(args.checkpoint, "rb"), pickle_module=dill)
    cfg = copy.deepcopy(OmegaConf.to_container(payload["cfg"]))
    cfg["task"]["env_runner"]["env_kwargs"] = {
        "split": args.split, "seed": 1111111, "env_name": args.task}
    cfg = OmegaConf.create(cfg)
    horizon = int(get_task_horizon(args.task))
    cfg.task.env_runner.n_train = 0
    cfg.task.env_runner.n_test = args.num_rollouts
    cfg.task.env_runner.max_steps = horizon
    cfg.task.env_runner.n_envs = 1

    out_dir = os.path.join(args.output_dir, args.task)
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=out_dir)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(torch.device("cuda:0"))
    policy.eval()

    n_action_steps = int(cfg.n_action_steps)
    calls_per_episode = math.ceil(horizon / n_action_steps)
    print(f"horizon={horizon} n_action_steps={n_action_steps} "
          f"-> {calls_per_episode} predict calls/episode; K={args.k_samples}")

    wrapped = UncertaintyProbeWrapper(policy, calls_per_episode, k=args.k_samples)
    env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=out_dir)
    runner_log = env_runner.run(wrapped)

    # per-episode success from the runner log
    succ = [float(v) > 0 for k, v in sorted(runner_log.items())
            if k.startswith("test/sim_max_reward_")]
    unc = wrapped.episode_uncertainty[:len(succ)]
    print(f"\nepisodes: {len(succ)} | success rate: {np.mean(succ):.1%}")
    print(f"uncertainty: mean {np.mean(unc):.4f} range [{min(unc):.4f}, {max(unc):.4f}]")
    # failure prediction: higher uncertainty -> failure? score = unc, y = fail
    a = auc(unc, [0 if s else 1 for s in succ])
    print(f"\nAUC(start-state action-variance -> FAILURE): {a:.3f}")
    print("  ~0.5 = uncertainty does NOT predict failure; >0.65 = validated signal")
    json.dump({"task": args.task, "n": len(succ), "k": args.k_samples,
               "success": succ, "uncertainty": unc, "auc_fail": a},
              open(os.path.join(args.output_dir, "probe_results.json"), "w"), indent=2)
    print(f"Wrote {args.output_dir}/probe_results.json")
    env_runner.close() if hasattr(env_runner, "close") else None


if __name__ == "__main__":
    main()
