---
name: dp-model-smoketest-status
description: "How to run a real Diffusion Policy checkpoint locally on the 4070 box — WORKING end-to-end; install + 5 compat patches porting gym 0.21->0.26"
metadata: 
  node_type: memory
  type: project
  originSessionId: 4f31bea1-2bdd-4e89-9f93-63656e3f0e12
---

Real-model smoke test of Diffusion Policy on this 4070 box (2026-06-09), task PickPlaceCounterToSink: **WORKING END-TO-END.** Official `eval_robocasa.eval_task` ran 2x 600-step rollouts at ~14 it/s, wrote eval_log.json + 2 videos to `~/robocasa_experiments/checkpoints/smoketest_evals/PickPlaceCounterToSink/`. Result: success_rate 0.0 (0/2 — but 2 rollouts is far too few to be a real number; need ~50+). Pipeline verified; performance number NOT.

Run command: `cd ~/diffusion_policy && MUJOCO_GL=egl PYOPENGL_PLATFORM=egl <robocasa-env-python> run_dp_smoketest.py -c ~/robocasa_experiments/checkpoints/dp_pretrain_human300_ep500.ckpt -t PickPlaceCounterToSink -s pretrain -n 2 -e 1`.

What was set up (all into the existing `robocasa` conda env, torch 2.7.1 / numpy 2.2.5 kept intact throughout):
- Cloned `robocasa-benchmark/diffusion_policy` -> `~/diffusion_policy` (empty setup.py, installed editable).
- Cloned `robocasa-benchmark/robomimic` -> `~/robomimic` (fork that adds `VisualCoreLanguageConditioned`; HEAD = "changes for dp to work with robocasa365"). Installed editable `--no-deps`. Stock pip robomimic 0.3.0 LACKS that class — must use this fork.
- pip-added: hydra-core, omegaconf, zarr, numcodecs, matplotlib, egl_probe, accelerate, transformers, tokenizers. None downgraded torch/numpy.
- Downloaded the published DP checkpoint (~1.7GB) to `~/robocasa_experiments/checkpoints/dp_pretrain_human300_ep500.ckpt` (HF robocasa/robocasa365_checkpoints, diffusion_policy/...pretrain_human300/...epoch=0500...ckpt).
- Driver: `~/diffusion_policy/run_dp_smoketest.py` calls eval_robocasa.eval_task for a SINGLE task (eval_task takes one `task` arg). Run with MUJOCO_GL=egl.

Compatibility PATCHES applied (modern diffusers/gym vs the fork's old pins) — all in the cloned forks:
1. `~/diffusion_policy/diffusion_policy/model/common/lr_scheduler.py` — `Union,Optional` from typing, not diffusers.optimization (diffusers 0.38 dropped them).
2. `~/robomimic/robomimic/utils/torch_utils.py` — same typing import fix.
3. `~/diffusion_policy/diffusion_policy/env_runner/robomimic_image_runner.py` — AsyncVectorEnv(..., shared_memory=False) (custom obs space incompatible with shared_memory=True).
4. `~/diffusion_policy/diffusion_policy/gym_util/async_vector_env.py` — added `reset(self, seed=None, options=None)` override that calls reset_async()+reset_wait() directly, bypassing gym>=0.26 base reset() which forwards seed/options into the old-API methods.
5. same file — flipped `concatenate(...)` arg order at 2 call sites to gym 0.26's `(space, items, out)`. (create_empty_array was already correct; step() needed no change — gym 0.26 base step() is a passthrough to the 4-tuple step_wait.)

Key insight: gym 0.26.2 coexists fine with numpy 2.x; the port was a finite, contained set of API-shim patches (5), NOT a numpy wall. The DP policy object exposes `.predict_action(obs_dict)` if ever bypassing the runner.

Next for real numbers: run with -n 50 (single task) here, or many tasks on the H100. To get the project's per-bucket weak-region analysis on a real model, still need to join eval episodes with init-state metadata ([[policy-analysis-harness]], [[first-experiment-pickplace-sink]]). See [[robocasa-compute-and-training-stack]].
