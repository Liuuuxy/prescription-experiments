---
name: h100-setup-complete
description: "H100 box DP-training stack is installed + smoke-test VERIFIED; exact paths, env, what's deferred, and how to run training/eval here"
metadata: 
  node_type: memory
  type: project
  originSessionId: f67d51b8-3da3-4024-9353-4295971251e5
---

The 2×H100-NVL (95GB each) box is set up for Diffusion-Policy training/eval, per [[robocasa-compute-and-training-stack]] and the H100 handoff. Done 2026-06-11. **NOT a shared filesystem with the 4070** — fresh build; everything under `/data/xinyua11` (the 4070's `~/...` paths do NOT exist here).

**`/home` has a 20GB cap** — all caches routed to /data via ~/.bashrc: `HF_HOME`, `PIP_CACHE_DIR`, `TMPDIR`, `TORCH_HOME`, `WANDB_DIR` all under `/data/xinyua11/...`. (These exports only apply to future login shells, not the Claude harness's non-interactive shells — set them inline when scripting heavy installs/downloads.) /home sits at 13GB; /data has ~1.5TB free.

**conda env:** `/data/xinyua11/conda/envs/robocasa` (py3.11). Use the env python directly: `/data/xinyua11/conda/envs/robocasa/bin/python` (or `conda activate robocasa`). torch 2.7.1+cu126 (cuda OK, 2 GPUs), numpy 2.2.5, mujoco 3.3.1 — pins intact.

**Repos (all `/data/xinyua11/`):**
- `robosuite` (master, `pip install -e`)
- `robocasa_pkg` — the robocasa **package** fork. NOTE the name: `/data/xinyua11/robocasa` is the **experiments bundle** (this git repo / the cwd / memory project `-data-xinyua11-robocasa`), so the package had to clone elsewhere. `python -m robocasa.scripts...` still works (module name unaffected).
- `diffusion_policy` (robocasa-benchmark fork, patched ×3 + `run_dp_smoketest.py` placed)
- `robomimic` (robocasa-benchmark fork w/ `VisualCoreLanguageConditioned`, `-e --no-deps`, patched ×1). The pip "robomimic 0.3.0 requires numpy==1.23.2/torch==2.0.1..." conflict warnings are EXPECTED/harmless (installed --no-deps).
- `mimicgen` (experimental/robocasa branch, patched ×2) — cloned but NOT pip-installed (data-gen, not needed for training).
The 5 gym/diffusers compat patches from [[dp-model-smoketest-status]] are applied here too (transfer/patches/, via git apply).

**Artifacts (re-downloaded, not transferred):**
- assets: `/data/xinyua11/robocasa_pkg/robocasa/models/assets` (~11GB; skipped `objs_aigen` 5.7GB pack that stalls — got tex/tex_generative/fixtures_lw/objs_objaverse/objs_lw).
- dataset: `/data/xinyua11/robocasa_pkg/datasets/v1.0/pretrain/atomic/PickPlaceCounterToSink/20250819/lerobot` (human pretrain, the only task downloaded).
- DP checkpoint: `/data/xinyua11/robocasa/checkpoints/dp_pretrain_human300_ep500.ckpt` (1.7GB, HF robocasa/robocasa365_checkpoints). `checkpoints/` is gitignored in the bundle.

**Smoke test VERIFIED (2026-06-11):** `cd /data/xinyua11/diffusion_policy && MUJOCO_GL=egl PYOPENGL_PLATFORM=egl <env python> run_dp_smoketest.py -c <ckpt> -t PickPlaceCounterToSink -s pretrain -n 2 -e 1` ran end-to-end at ~8.3 it/s, wrote eval_log.json + 2 mp4s. success_rate 0/2 (NOT a real number — pipeline verified, not performance; need n≥50). BrokenPipeError/leaked-semaphore at shutdown = benign AsyncVectorEnv teardown.

**To TRAIN:** `cd /data/xinyua11/diffusion_policy && conda activate robocasa && MUJOCO_GL=egl python train.py --config-name=train_diffusion_transformer_bs192 task=robocasa/<soup>`. Soup configs in `diffusion_policy/config/task/robocasa/`. Dataset paths auto-resolve via `DATASET_SOUP_REGISTRY` to `/data/xinyua11/robocasa_pkg/datasets/...` (macros.DATASET_BASE_PATH=None falls back to the right default — no override needed). CAVEAT: the `pretrain_human300` soup lists 300 demos across MANY tasks; only PickPlaceCounterToSink is on disk → build a single-task PickPlaceCounterToSink soup for the experiment baseline (see [[first-experiment-pickplace-sink]]). Configs also carry one stale hardcoded `/mnt/amlfs-01/...` env_runner.dataset_path — irrelevant unless you turn on the env_runner during train.

**Deferred (handoff says H100 = training only):** openpi (pi0) + Isaac-GR00T data-gen stacks NOT installed here; their standalone scripts (run_pi0_client.py, record_pi0_demos.py, run_groot_smoketest.py) sit in `transfer/new_files/` for when targeted-data generation is needed. See [[expert-data-generation-loop]], [[pi0-groot-local-eval-setup]].

---
**BASELINE TRAINING — BUILT + VALIDATED, run HELD pending free GPUs (2026-06-11).**
- Extra dep needed beyond the install list: **`numpydantic`** (`pip install numpydantic`) — robocasa's `groot_utils/schema.py` imports it, and DP's `lerobot_dataset.py` imports that. Without it, hydra fails with `Error locating target ...LerobotCotrainingDataset`. Now installed.
- Single-task config: `diffusion_policy/config/task/robocasa/pickplacesink_human.yaml` (copy of pretrain_human300.yaml but `dataset_paths: [<PickPlaceCounterToSink lerobot dir>]` instead of `dataset_soup`; 108 human demos, filter_key=None=all). The loader only consumes path/filter_key/ds_weight per entry; horizon comes from config level — single-task works, ds_weight auto = 1.0.
- Online `env_runner` is NOT instantiated when `training.rollout_every` is null (startup instantiation is commented out), so the stale `/mnt/amlfs-01/...` env_runner.dataset_path is harmless during training. Eval offline afterward with `eval_robocasa`/`run_dp_smoketest.py`.
- Checkpointing VERIFIED: writes `latest.ckpt` + topk `epoch=NNNN-test_mean_score=-1.000.ckpt` (1.7GB each) at every `checkpoint_every` (default 100). TopKCheckpointManager defaults the missing `test_mean_score` to -1.0 (no online rollout) — no crash.
- Training validated end-to-end (loss decreasing, optimizer stepping) at bs32; FULL bs192 blocked only by GPU contention.
- **GPU contention:** box is SHARED — user ssagar6 runs VLLM EngineCore ~78GB on BOTH H100-NVLs (often 100% util), leaving ~12-14GB free. DP bs192 wants ~24GB+. So real runs are held.
- **PLAN (user-chosen): 3 seeds x 500 epochs, bs192.** Launcher: `bash /data/xinyua11/robocasa/launch_dp_baseline.sh <seed> <gpu_index> [epochs=500]` — refuses to start unless target GPU has >=40GB free (MINFREE_MIB). Outputs/checkpoints -> `/data/xinyua11/dp_runs/pickplacesink_human/seed<N>/`. Then offline-eval each to get sigma_seed (the dominant unknown, see [[first-experiment-pickplace-sink]] power analysis).

**CRITICAL DATALOADER FIXES (2026-06-20/21) — without these, training is unusable:**
1. **Default config is wildly data-bound.** `train_diffusion_transformer_bs192` ships `num_workers` high but `persistent_workers=False`, and DP's `LerobotDataset` decoded mp4 frames on the fly every sample -> GPU 0% util, ~9 min/epoch (~3 days/seed).
2. **Fix = use the in-RAM cached frames + null the lang-encoder.** Edited `diffusion_policy/dataset/lerobot_dataset.py`: `LerobotDataset` now inherits `CachedLeRobotSingleDataset` (decodes ALL frames into RAM once at init, ~13GB for this task, ~170s; box has 1TB RAM) with `video_backend="pyav"` (base default 'opencv' is NOT implemented in get_all_frames; decord is unsupported). AND in `LerobotCotrainingDataset.__init__`, after building datasets, set each `_d._lang_encoder=None` + gc + empty_cache. **Why the null is mandatory:** the dataset keeps a live CUDA lang-encoder (`del_lang_encoder_after_init=False`); forking DataLoader workers after CUDA init then DEADLOCKS at the first batch (workers 0% CPU, Sl state, hangs forever at "epoch 0 0%"). Embeddings are precomputed (numpy) so the encoder is unused in __getitem__. Saved as `transfer/patches/diffusion_policy_cached_loader.patch`.
3. **Result: ~3 min/epoch, GPU 90-100% util (compute-bound), ~24h/seed.** Launcher now uses `num_workers=16 persistent_workers=True +prefetch_factor=4` (note `+` prefix: prefetch_factor isn't in the struct).
4. Single-task config `pickplacesink_human.yaml` has only 117 batches/epoch but `max_train_steps=500`, so it cycles the data ~4x/epoch ("Creating new train dataloader iterator" is normal). 500 epochs x 500 steps on 108 demos = heavy; rely on the per-100-epoch checkpoints + pick best at eval (overfit guard).

**RUN STATUS (2026-06-21):**
- **seed 0:** non-cached run trained 3 days to epoch 471 then was accidentally killed by a pkill; checkpoints to **epoch 400 + latest survived**, preserved at `dp_runs/pickplacesink_human/seed0_noncached_partial_ep471/`. Usable (epoch 400 ~converged).
- **seed 1 (GPU0) + seed 2 (GPU1):** launched cached, both compute-bound, ETA ~24h to epoch 500. `dp_runs/pickplacesink_human/seed{1,2}/`.
- Compare all 3 at a common epoch (>=400, all have 100/200/300/400 ckpts). NEXT: offline-eval (run_dp_smoketest.py / eval_robocasa, n>=50) -> per-episode success -> sigma_seed + DP weak-region (first DP weak-region data for the project).
