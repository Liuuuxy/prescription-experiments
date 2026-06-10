---
name: expert-data-generation-loop
description: "VALIDATED keystone loop — generate trainable LeRobot data from an expert policy (pi0/GR00T), no human demos needed"
metadata: 
  node_type: memory
  type: project
  originSessionId: 4f31bea1-2bdd-4e89-9f93-63656e3f0e12
---

**The data->train loop is closed and validated (2026-06-10).** This unblocks the whole targeted-vs-random experiment ([[first-experiment-pickplace-sink]]) without any human demos.

Pipeline:
1. **Generate states+actions with an expert policy**: `~/openpi/examples/robocasa/record_pi0_demos.py` drives the gym env with pi0 (or adapt for GR00T), records mujoco states+actions of SUCCESSFUL episodes -> robomimic hdf5 (states/actions/model_file/ep_meta + env_args). Run in openpi_env with pi0 server up, MUJOCO_GL=egl. (Recorder does NOT save images — only states.)
2. **Render obs + convert to LeRobot in ONE step**: `robocasa/scripts/dataset_scripts/convert_hdf5_lerobot.py --raw_dataset_path <hdf5> --camera_names robot0_agentview_left robot0_agentview_right robot0_eye_in_hand --camera_height 256 --camera_width 256`. It recreates the env from env_args, replays states (`env.reset_to`), RENDERS the cameras internally, and writes a LeRobot dataset to `<hdf5_parent>/lerobot/`. Run in the **robocasa** env (has lerobot 0.3.3). Use `MUJOCO_GL=osmesa` (CPU render) if the GPU is busy with a server — slow but non-disruptive; `egl` if GPU is free (fast).

VALIDATED: 5 pi0 PickPlaceCounterToSink demos -> LeRobot with 5 episodes / 1415 frames / 20fps / 256x256, features = 3 camera videos + observation.state + action + annotations + next.reward/done. Decoded frame: mean 127, std 58, full 0-255 range = real (non-blank) render. Output at `~/robocasa_experiments/mimicgen_src/lerobot/`.

Implication: pi0 (58%) / GR00T (66%) are expert teachers -> generate arbitrary trainable data in any region. For the targeted arm, constrain init states to a weak region (extend record_pi0_demos). For training DP on this LeRobot data: H100 (the diffusion_policy fork trains on LeRobot task soups). Note `convert_hdf5_lerobot` also needs the per-task geom patches if it calls get_datagen_info (add_datagen_info path); the basic obs render path used here did not. See [[robocasa-compute-and-training-stack]], [[policy-analysis-harness]].
