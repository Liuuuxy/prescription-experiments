# H100 Handoff — RoboCasa targeted-data IL experiment

Written 2026-06-10 from the 4070 dev box. Goal of the H100 work: **train diffusion policies** for the targeted-vs-random data-efficiency experiment. All dev/eval/data-generation infra already works on the 4070 box; the H100 is for **training only** (the DP fork recommends 24GB+; 4070's 12GB can't train the default config).

If a fresh Claude Code session starts on the H100 and `~` is the **same shared filesystem**, it inherits all of this automatically (memory at `~/.claude/projects/-home-asurite-ad-asu-edu-xinyua11-robocasa/memory/`, the repos, the datasets). Read that memory first — especially `dp-model-smoketest-status`, `expert-data-generation-loop`, `pi0-groot-local-eval-setup`, `first-experiment-pickplace-sink`.

---

## 0. Project goal (one line)
Test whether adding **targeted** demos from a policy's **failure regions** improves a diffusion policy more **per-demo** than the same number of **random** demos. (Active learning for IL; the weak-region detector is the acquisition function.)

## 1. What already works on the 4070 (reuse, don't rebuild)
- **3 policies eval'd on PickPlaceCounterToSink (pretrain, n=50):** GR00T **66%**, pi0 **58%**, DP **10%**. (`~/robocasa_experiments/results_log.md`, `model_triage.md`)
- **Weak-region detector** (`~/robocasa_experiments/policy_analysis/analyze_pi0_weakregions.py`): per-episode success joined with object category / position-rel-to-robot / layout / style + failure-mode tag (no-grasp / grasped-no-transport / reached-sink-no-place). Output: `~/robocasa_experiments/weakregion/pi0_PickPlaceCounterToSink/`.
  - **KEY FINDING (pi0, n=50):** **96% of failures are `fail_no_grasp`** — pi0 fails at the GRASP, not transport/placement. Object-type driven: 0% on tall/cylindrical/awkward objects (juice, jugs, bottles, pitcher, glass_cup, rolling_pin, reamer, blender_jug, cheese_grater, squash…), 100% on compact food (orange, tomato, peach, carrot, eggplant, lemon, steak…). Worst regions: far/lateral (mid-left 29%, far-right 33%, far-center 38%) vs near-left 89%.
  - **=> Concrete targeting signal for the experiment:** the "targeted" arm should be **grasp demos for tall/awkward objects at far/lateral positions**. The "random" arm = demos from the full distribution. Same budget; compare improvement-per-demo.
- **Expert→trainable-data loop (VALIDATED):** `record_pi0_demos.py` (pi0→robomimic hdf5 w/ states) → `robocasa/scripts/dataset_scripts/convert_hdf5_lerobot.py` (renders obs + writes LeRobot). See memory `expert-data-generation-loop`. Example output: `~/robocasa_experiments/mimicgen_src/lerobot/` (5 eps, 1415 frames).
- **mimicgen** works on simple tasks (CloseDrawer 40%) but **fails** on PickPlaceCounterToSink (0/20) — so use the **pi0/GR00T expert loop**, not mimicgen, to generate targeted data for the hard task.

## 2. DP training setup on the H100 (the actual new work)
Repo: `~/diffusion_policy` (fork `robocasa-benchmark/diffusion_policy`). **It needs the robocasa-benchmark robomimic fork** (`~/robomimic`, has `VisualCoreLanguageConditioned`) — NOT pip robomimic.

Install into a py3.11 env that also has robocasa (see memory `dp-model-smoketest-status` for the exact sequence). Pip-add: hydra-core, omegaconf, zarr, numcodecs, matplotlib, egl_probe, accelerate, transformers. **5 compatibility patches already applied in `~/diffusion_policy` + `~/robomimic`** (will be present if `~` is shared; otherwise re-apply — all documented in `dp-model-smoketest-status`):
1. `diffusion_policy/model/common/lr_scheduler.py` — `Union,Optional` from typing.
2. `robomimic/utils/torch_utils.py` — same typing fix.
3. `diffusion_policy/env_runner/robomimic_image_runner.py` — `AsyncVectorEnv(..., shared_memory=False)`.
4. `diffusion_policy/gym_util/async_vector_env.py` — `reset()` override (gym 0.26 seed bypass).
5. same file — `concatenate()` arg order → `(space, items, out)`.
(These were for EVAL/gym-0.26; training may surface more — patch as needed.)

**Train** (DP fork): `python train.py --config-name=train_diffusion_transformer_bs192 task=robocasa/<dataset-soup>`. Training reads **LeRobot** task soups. **Eval**: `python eval_robocasa.py --checkpoint <ckpt> --task_set <set> --split <split>` (works on the 4070 too — inference only needs 8GB). Stats: `scripts/get_eval_stats.py`.

GPU on this box during eval: pi0 ~6GB, GR00T ~8.5GB, DP small. On the H100 all fit easily.

## 3. The experiment (concrete steps)
1. **Baseline dataset**: pick N (e.g. 100) demos for PickPlaceCounterToSink. Either download the official human/mg LeRobot data (`python -m robocasa.scripts.download_datasets --tasks PickPlaceCounterToSink --source human --split pretrain`) OR generate them with the expert loop (pi0). Train baseline DP.
2. **Eval baseline** on a **fixed, region-stratified held-out set** (TODO: build — fixed seeds binned by object/position/scene so before/after is clean). Run the weak-region detector on the baseline DP.
3. **Identify weak region(s)** from the detector output.
4. **Two retrain arms, same demo budget M:**
   - *random*: M extra demos sampled from the full init-state distribution.
   - *targeted*: M extra demos with init states **constrained to the weak region** (extend `record_pi0_demos.py` to constrain the object type/position sampler; then convert to LeRobot and merge into the soup).
5. **Compare** improvement-per-demo, and whether targeted helps the weak region **without hurting** others (that's why the eval set must be stratified).

## 4. Local pieces still to build before/independent of training (can do on 4070)
- **Controlled-init-state targeted generator** (extend recorder) — the targeted arm's data.
- **Stratified held-out eval set** (fixed region-binned seeds).
- **GR00T cross-policy weak-region** (do pi0 + GR00T agree on failure regions? = universally-hard targets).

## 5. If `~` is NOT shared — copy to the H100
- `~/.claude/projects/-home-asurite-ad-asu-edu-xinyua11-robocasa/memory/` (the memory — most important)
- repos: `~/diffusion_policy`, `~/robomimic` (patched), `~/robocasa`, `~/robosuite`, (`~/openpi`, `~/Isaac-GR00T` if generating data there)
- `~/robocasa_experiments/` (results, the validated LeRobot dataset, recorded demos, this doc)
- checkpoints: `~/robocasa_experiments/checkpoints/` (DP 1.7GB, pi0 12GB, GR00T 16GB) — or re-download on the H100.
