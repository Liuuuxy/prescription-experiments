# H100 Progress Report — back-handoff to the dev box

Written 2026-06-22 **from the H100** (reverse of `H100_HANDOFF.md`, which was written from the 4070 → H100). This documents everything done on the H100 so a session on the other machine knows the state. Companion memory: `transfer/memory/h100-setup-complete.md` (synced) — read it too.

**Box:** 2× H100-NVL (95 GB each), **shared** (other users' VLLM jobs frequently saturate both GPUs — the launcher guards against this). **`/home` has a 20 GB cap** → everything lives under `/data/xinyua11`, caches (HF/pip/tmp/torch/wandb) routed off home via `~/.bashrc`.

---

## 1. Setup — DONE and smoke-tested
- **Not a shared filesystem with the 4070.** Fresh build; the 4070's `~/...` paths do **not** exist here. Everything is under `/data/xinyua11/`.
- **conda env:** `/data/xinyua11/conda/envs/robocasa` (py3.11). torch 2.7.1+cu126, numpy 2.2.5, mujoco 3.3.1 (pins intact). Use the env python directly: `/data/xinyua11/conda/envs/robocasa/bin/python`.
- **Repos (all `/data/xinyua11/`):** `robosuite` (master), `robocasa_pkg` (the robocasa **package** — note: `/data/xinyua11/robocasa` is the **experiments bundle / this git repo**, so the package had to clone to `robocasa_pkg`), `diffusion_policy` (patched), `robomimic` (fork, patched, `-e --no-deps`), `mimicgen` (cloned, not installed).
- **Patches applied:** the original 5 gym/diffusers shims (`transfer/patches/`), **plus a new one** (see §2): `transfer/patches/diffusion_policy_cached_loader.patch`.
- **Extra dep** not in the original list: **`numpydantic`** (robocasa groot schema needs it; without it hydra can't load `LerobotCotrainingDataset`).
- **Artifacts (re-downloaded, ~13 GB):** kitchen assets at `robocasa_pkg/robocasa/models/assets` (skipped the `objs_aigen` pack that stalls); PickPlaceCounterToSink human-pretrain LeRobot data at `robocasa_pkg/datasets/v1.0/pretrain/atomic/PickPlaceCounterToSink/20250819/lerobot` (**108 demos**); published DP checkpoint at `robocasa/checkpoints/dp_pretrain_human300_ep500.ckpt` (1.7 GB).

## 2. CRITICAL: dataloader was unusably slow — fixed
- **Symptom:** default `train_diffusion_transformer_bs192` trained at ~9 min/epoch with GPU at 0% util → ~3 days/seed. DP's `LerobotDataset` decodes mp4 frames **on the fly every sample**.
- **Fix (in `diffusion_policy/dataset/lerobot_dataset.py`, saved as `transfer/patches/diffusion_policy_cached_loader.patch`):**
  1. `LerobotDataset` now inherits **`CachedLeRobotSingleDataset`** → decodes all frames into RAM once at init (~13 GB, ~170 s; box has 1 TB RAM), then array-indexes. Pass `video_backend="pyav"` (base default `"opencv"` is **not** implemented in `get_all_frames`; `decord` is unsupported).
  2. In `LerobotCotrainingDataset.__init__`, after building datasets, **null each `_d._lang_encoder`** (+ gc + empty_cache). **Mandatory:** the dataset keeps a live CUDA lang-encoder (`del_lang_encoder_after_init=False`); forking DataLoader workers after CUDA init **deadlocks at the first batch** (workers 0% CPU / `Sl`, hangs forever at "epoch 0 0%"). Embeddings are precomputed, so the encoder is unused in `__getitem__`.
- **Result: ~3 min/epoch, GPU 90–100% util (compute-bound), ~24 h/seed.**
- Launcher uses `num_workers=16 persistent_workers=True +prefetch_factor=4` (note the `+` — prefetch_factor isn't in the config struct).

## 3. Experiment config + scripts (in this repo)
- **Single-task baseline config:** `diffusion_policy/.../config/task/robocasa/pickplacesink_human.yaml` — copy of `pretrain_human300.yaml` but uses `dataset_paths: [<the 108-demo lerobot dir>]` instead of `dataset_soup` (the soup expects 300 demos across many tasks; only PickPlaceCounterToSink is on disk). Dataset paths auto-resolve via the registry (no `DATASET_BASE_PATH` override needed).
- **`launch_dp_baseline.sh <seed> <gpu> [epochs=500]`** — guarded launcher (refuses to start unless target GPU has ≥40 GB free). Outputs → `/data/xinyua11/dp_runs/pickplacesink_human/seed<N>/`.
- **`eval_baseline_seeds.sh [epoch=0400] [n=50]`** — evals seed0/1/2 at a common epoch across both GPUs.
- Online `env_runner` eval is OFF during training (`rollout_every=null`), so its stale `/mnt/amlfs-01/...` path is harmless. Eval is done offline with `run_dp_smoketest.py` / `eval_robocasa`.

## 4. Training run status (2026-06-22)
- **seed 0:** trained 3 days (non-cached, pre-fix) to epoch 471, then **accidentally killed** by a cleanup `pkill`. Checkpoints through **epoch 400 + latest survived**, preserved at `dp_runs/pickplacesink_human/seed0_noncached_partial_ep471/` (epoch 400 ~converged). Usable.
- **seed 1 (GPU 0) + seed 2 (GPU 1):** launched cached, **training now, ~epoch 25/500**, ETA ~22 h to epoch 500. `dp_runs/pickplacesink_human/seed{1,2}/`.
- A background watcher (`tmp/wait_for_seeds.sh`) will fire when both finish and trigger the full 3-seed eval automatically.

## 5. First real eval result — READ THE CAVEAT
- **Our seed-0 baseline @ epoch 400, n=50 = 2% success (1/50)** on PickPlaceCounterToSink (`dp_runs/pickplacesink_human/eval/seed0_ep400/eval_log.json`).
- **This is NOT comparable to the "DP 10%" in memory.** That 10% (and the n=2 H100 smoke test) used the **published** checkpoint `robocasa/checkpoints/dp_pretrain_human300_ep500.ckpt` (authors' model, trained on the full **300-demo multi-task** human soup). The 2% is **our single-task 108-demo** model. Same eval harness/protocol (`run_dp_smoketest.py`→`eval_robocasa`, `split=pretrain`, test seeds 100000+); **different model**, and n=50 vs n=2.
- 2% is low but plausible for a 108-demo single-task baseline; epoch 400 of this config (250k steps over 108 demos) may be **overfit**. To interpret: eval earlier epochs (200/300) of seed 0, eval the published ckpt here at n=50 as an apples-to-apples reference, and use seeds 1/2 for the across-seed spread (`sigma_seed`).

## 6. Next steps (per the experiment plan in `H100_HANDOFF.md` §3 + `first-experiment-pickplace-sink` memory)
1. When seeds 1/2 finish: eval all 3 at a common epoch (≥400) → **`sigma_seed`** (the dominant unknown from the power analysis) + first DP **weak-region** data.
2. Build the **stratified held-out eval set** (fixed region-binned seeds) — still a TODO.
3. Then the targeted-vs-random retrain arms (same dataloader fix applies — every future training run benefits).
