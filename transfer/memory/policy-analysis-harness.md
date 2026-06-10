---
name: policy-analysis-harness
description: Location and design of the quant+qual policy-analysis harness (project step 5.4)
metadata: 
  node_type: memory
  type: project
  originSessionId: 4f31bea1-2bdd-4e89-9f93-63656e3f0e12
---

Built the **policy-analysis harness** (project step 5.4: quant+qual model triage + weak-region detection) at `~/robocasa_experiments/policy_analysis/` (sibling dir, kept OUT of the upstream robocasa clone to keep it clean).

Files: `rollout_eval.py` (run policy, log per-episode success + metadata, save bucketed videos, write metrics.json + report.txt), `analysis.py` (pure aggregation: overall + by object category / spatial region / layout / style, Wilson CI, weakest-bucket detection — simulator-independent so it's unit-testable), `policies.py` (`RandomPolicy` + `CheckpointPolicyStub` adapter template for DP/GR00T/pi0), `test_analysis.py` (5 tests, all pass), `README.md`.

Key API facts learned (verified 2026-06-08): build on `robocasa.utils.env_utils.create_env` (robosuite env, has camera obs) NOT the gym wrapper (state-only obs, useless for vision policies). `env.action_spec` only works AFTER a `reset()` (robots instantiated then). Target manipulated object is named `"obj"`; category via `env.get_ep_meta()["object_cfgs"][*]["info"]["cat"]`; init pos via `env.sim.data.body_xpos[env.obj_body_id["obj"]]`; success via `env._check_success()`. Run with `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`.

Status: verified end-to-end on this 4070 box with RandomPolicy (0% success as expected). Next: implement the CheckpointPolicyStub on the H100 to point it at real DP/GR00T/pi0 checkpoints. This feeds the targeted-vs-random experiment ([[first-experiment-pickplace-sink]]) — the weak-region output IS the targeting signal. Note the official DP fork already gives overall success+videos via eval_robocasa.py; this harness adds the per-bucket breakdown. See [[robocasa-compute-and-training-stack]].
