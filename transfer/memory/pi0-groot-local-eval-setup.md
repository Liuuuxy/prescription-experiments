---
name: pi0-groot-local-eval-setup
description: "Local setup to run pi0 (openpi) and GR00T checkpoints on the 4070 box — envs, checkpoints, patches, run commands"
metadata: 
  node_type: memory
  type: project
  originSessionId: 4f31bea1-2bdd-4e89-9f93-63656e3f0e12
---

Running the other two RoboCasa benchmark policies locally on the 4070 box (2026-06-09), task PickPlaceCounterToSink. Companion to [[dp-model-smoketest-status]]. DP 50-rollout result was **10% (5/50)**.

## pi0 (openpi fork) — SERVER WORKING, client in progress
- Cloned `robocasa-benchmark/openpi` -> `~/openpi`. Conda env **`openpi_env`** (python 3.11). `pip install -e .` + `pip install -e packages/openpi-client/`. jax[cuda12]==0.5.3 on GPU.
- **openpi's config.py imports robocasa at module level**, so the server env also needs robocasa. Installed robosuite(master)+robocasa editable into openpi_env with `CC=gcc` (evdev compile). Pulled numpy 2.2.5 + protobuf 3.19.6 (protobuf conflict warnings are non-fatal). Also had to `pip install h5py lxml` (and reinstall robocasa fully — an interrupted install left deps missing).
- **chex gotcha**: `pip install chex` pulled jax 0.10.1, breaking openpi's jax==0.5.3. Fix: `pip install "jax[cuda12]==0.5.3"` then `pip install --no-deps chex==0.1.88`.
- Checkpoint (~12GB) at `~/robocasa_experiments/checkpoints/pi0_ckpt/pi0/pi0_robocasa_pretrain_human300/multitask_learning/75000` (has params/ + assets/norm_stats.json). Config name: **`pi0_robocasa_pretrain_human300`**.
- **PATCH** (the collaborator's norm-stats fix): `~/openpi/src/openpi/groot_utils/groot_openpi_dataset.py` `_load_norm_stats_from_groot_mixture_dataset` — wrap per-dataset load in try/except FileNotFoundError, return None when training stats absent. Then create_trained_policy falls back to checkpoint `assets/norm_stats.json`. (Needed because no training datasets are downloaded.)
- Server (run in openpi_env): `cd ~/openpi && XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl python scripts/serve_policy.py --port 8000 policy:checkpoint --policy.config pi0_robocasa_pretrain_human300 --policy.dir <CKPT>`. **Set XLA_PYTHON_CLIENT_PREALLOCATE=false** or JAX grabs ~9.8GB (75%) and starves the client's MuJoCo EGL renderer.
- Client (same env): `cd ~/openpi/examples/robocasa && MUJOCO_GL=egl python run_pi0_client.py -t PickPlaceCounterToSink -n 2` (driver I wrote, calls main.eval_env for one task). replan_steps=5. Team reported pi0 = SUCCESS (1/1) — the only model that worked.

## pi0 RESULT: n=50 = **58% (29/50)** on PickPlaceCounterToSink (vs DP 10%). pi0 is the strong policy.

## mimicgen (5.3) — **FULL PIPELINE NOW WORKS on v1.0.1** (was fully blocked). First result: gen success 0/20 with pi0 sources on PickPlaceCounterToSink.
Working pipeline (all in mimicgen_env unless noted):
1. Record pi0 source demos: `~/openpi/examples/robocasa/record_pi0_demos.py` (run in **openpi_env** with pi0 server up, MUJOCO_GL=egl) — drives the gym env like the pi0 client, records mujoco states+actions of SUCCESSFUL episodes, writes robomimic hdf5 (data group + per-demo model_file/ep_meta/states/actions) + convert_to_robomimic_format. Output `~/robocasa_experiments/mimicgen_src/PickPlaceCounterToSink_pi0_src.hdf5` (5 demos). Solves the source-demo wall (v1.0.1 LeRobot lacks states).
2. `prepare_src_dataset.py --dataset <hdf5> --env_interface MG_PnPCounterToSink --env_interface_type robosuite --output <prepared.hdf5>` (NOTE: interface_type is **robosuite** not robocasa; the robocasa interfaces subclass RobosuiteInterface).
3. GEOM PATCH: `~/mimicgen/.../env_interfaces/robocasa/single_stage/mg_pnp.py` MG_PnPCounterToSink.get_object_poses — sink geom `_bottom` is gone in v1.0.1; fall back `reg_basin_left`->`reg_basin`->`g0`.
4. `generate_config_templates.py` → makes `~/mimicgen/mimicgen/exps/templates/robocasa/single_stage/kitchen_pnp/PnPCounterToSink.json`. Copy + set source.dataset_path=<prepared>, source.filter_key=None, generation.path=<out>, generation.num_trials.
5. `generate_dataset.py --config <cfg>` (MUJOCO_GL=egl). Reports gen success rate. (Failure-video playback at end errors on action_limits — cosmetic, generation is fine.)

RESULTS (5.3 characterization, pi0 sources): **PickPlaceCounterToSink 0/20** (complex: grasp+transport+place, long 214-358-step closed-loop trajectories replay poorly open-loop; matches team's 0/10 keyboard-source) vs **CloseDrawer 0.40 (8/20)** (simple: push, no grasp, short 167-209 steps). CONCLUSION: MimicGen reliability depends strongly on task complexity — viable for simple single-stage tasks, NOT multi-stage pick-place. Each task needs its own geom patch (sink: `_bottom`->`reg_basin`; drawer: handle_name `_handle`->`_reg_main`/`_g0`, in MG_OpenDrawer/MG_CloseDrawer). Implication for [[first-experiment-pickplace-sink]]: MimicGen can't generate targeted data for the hard PickPlace task — use the pi0-recorder with controlled init states instead (the recorder already produces valid source/training demos).

## mimicgen (5.3) — OLD: ENV CONFLICT FIXED; deeper v0.2->v1.0.1 port remains
- Dedicated env **mimicgen_env** (py3.11): robosuite(master)+robomimic(`ARISE-Initiative/robomimic -b robocasa` at ~/robomimic_robocasa)+mimicgen(`NVlabs/mimicgen -b experimental/robocasa` at ~/mimicgen)+robocasa, ALL import together under numpy 2.2.5 / torch 2.7.1 / robosuite 1.5.2. The root blocker WAS: mimicgen setup pins numpy==1.23.2 vs robocasa pins numpy==2.2.5 — resolved by installing robocasa last (numpy->2.2.5); the pins are cosmetic, code runs on numpy 2. `single_arm_env` import error is HARMLESS (only mimicgen's old robosuite demo envs Stack/Threading, not robocasa). prepare_src_dataset.py loads; mimicgen robocasa PnP interface (mimicgen/env_interfaces/robocasa/single_stage/mg_pnp.py) imports.
- REMAINING to actually generate (deeper v0.2-era-mimicgen vs v1.0.1-robocasa gaps): (1) **data format**: v1.0.1 ships **LeRobot** datasets but mimicgen prepare_src_dataset wants **robomimic hdf5** → need conversion or hdf5 source; (2) **geom names**: collaborator patched _bottom->_reg_basin_left in MG_PnPCounterToSink (v0.2->v1.0.1 geom renames) + sink-geom fallback; (3) task config json (e.g. PickPlaceCounterToSink.json) may need generating from configs/robocasa/*.py. This is a genuine port, not a quick fix.

## GR00T (Isaac-GR00T fork) — WORKING. **n=50 = 66% (33/50)** on PickPlaceCounterToSink — strongest of the three (GR00T 66% > pi0 58% > DP 10%, all n=50). (n=2 smoke was 50%.) Run via `~/Isaac-GR00T/scripts/run_groot_smoketest.py` in groot_env, `USE_TF=0 TRANSFORMERS_NO_TF=1 MUJOCO_GL=egl`. Extra fixes beyond the OLD section: robocasa install into groot_env forced torch->2.7.1+cu126 (so flash-attn needed the **cu12torch2.7cxx11abiTRUE** wheel via `--force-reinstall --no-deps` — same 2.7.4.post1 version tag as the torch2.5 wheel so plain install is skipped); `opencv-python-headless>=4.10` + `protobuf==3.20.3` for numpy2/transformers; `USE_TF=0` to skip the numpy-1-built tensorflow/ml_dtypes that transformers auto-imports. Must kill pi0 server first (GPU). GR00T used ~8.5GB GPU.

## GR00T — OLD prep notes (pre-run)
- groot_env (py3.10): `pip install -e .[base]` -> torch 2.5.1+cu124. flash-attn: prebuilt wheel `flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310` from Dao-AILab releases (no nvcc to build). gr00t + flash_attn import OK.
- Checkpoint `~/robocasa_experiments/checkpoints/groot_ckpt/gr00t_n1-5/foundation_model_learning/pretraining/checkpoint-80000` (has experiment_cfg/metadata.json; embodiment_tag=`new_embodiment`, data_config=`panda_omron`).
- Driver `~/Isaac-GR00T/scripts/run_groot_smoketest.py` (injects 1-task set, server-thread+client). Run in groot_env, MUJOCO_GL=egl. **Must kill pi0 server first to free GPU** (GR00T ~2.7B needs ~6GB + mujoco).
- OLD section below (pre-flash-attn):
- Cloned `robocasa-benchmark/Isaac-GR00T` -> `~/Isaac-GR00T` (shallow clone left .github/media as `D` in git status; core code present). Checkpoint (~16GB) downloaded to `~/robocasa_experiments/checkpoints/groot_ckpt/gr00t_n1-5/foundation_model_learning/pretraining/` (incl. checkpoint-80000 + experiment_cfg).
- Plan: separate conda env **groot_env** (py3.10), torch 2.7.x, **prebuilt flash-attn wheel** matching torch2.7+cu126+cxx11abiTRUE+py310 (building needs nvcc, not installed). Server/client zmq: `scripts/run_eval.py` (--model_path, --task_set, --split, --port 5555, --n_episodes, --n_envs). Team reported GR00T = 0/1.

See [[robocasa-compute-and-training-stack]], [[policy-analysis-harness]], results at `~/robocasa_experiments/results_log.md`.
