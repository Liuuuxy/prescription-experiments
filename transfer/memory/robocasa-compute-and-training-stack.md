---
name: robocasa-compute-and-training-stack
description: Compute constraints and the official RoboCasa diffusion-policy training/eval stack
metadata: 
  node_type: memory
  type: project
  originSessionId: 4f31bea1-2bdd-4e89-9f93-63656e3f0e12
---

**Working setup:** user travels with only a MacBook, so needs a stable remote machine. **Decided (2026-06): the remote H100 is the training host** (datasets, checkpoints, DP training live there); this 4070 box is dev + headless eval only. Requirements: stable remote access, working RoboCasa env, GPU for diffusion-policy training, **headless eval that saves videos/metrics (no GUI)**, and a full pipeline: train → eval → failure analysis → targeted data selection → retrain → compare.

**This dev machine (host xinyua11):** RTX 4070, **12 GB** VRAM. RoboCasa's Diffusion Policy fork recommends **24 GB+ for training (48 GB+ preferred)**; the default config is `train_diffusion_transformer_bs192` (batch 192) which will OOM at 12 GB. Inference/eval only needs 8 GB. So: this box is fine for **dev/debug + evaluation**, but real training belongs on the **H100/Kochi**.

**Official training stack** (docs/benchmarking/policy_learning_algorithms.md): RoboCasa benchmarks Diffusion Policy, Openpi (pi0), GR00T — all as forks under github.com/robocasa-benchmark. For this project use the **Diffusion Policy** fork: github.com/robocasa-benchmark/diffusion_policy (`train.py`, `eval_robocasa.py`, `scripts/get_eval_stats.py`).

**Key RoboCasa scripts:** `download_datasets.py` (fetch demo datasets), `collect_demos.py` (collect/generate demos, MimicGen path), `download_kitchen_assets.py` (scenes/objects, ~10GB+).

Env install details for this machine: robocasa conda env at miniconda3/envs/robocasa (py3.11), robosuite master cloned to ~/robosuite. **Install VERIFIED 2026-06-08** — headless EGL rollout on PickPlaceCounterToSink created the env + saved a video (`MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`). Use the env's pip directly (miniconda3/envs/robocasa/bin/pip); `conda run -n robocasa pip` mis-resolves to system pip. Asset gotcha: `download_kitchen_assets` stalled on the 5.7GB aigen pack; recovered by fetching only the missing `objs_lw` (lightwheel objects, 792MB) via `--type objs_lw`. aigen objects were skipped (not needed for objaverse-based tasks). For targeted-demo generation (Mechanism A) you'll need to install **mimicgen** later. See [[project-goal-targeted-data-il]] and [[first-experiment-pickplace-sink]].
