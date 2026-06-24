# H100 task: pi0 LoRA fine-tune — targeted (core) vs random arms

Goal: prove the core-algorithm-selected data beats random. Baseline = pretrained pi0
(~55% on PickPlaceCounterToSink). Fine-tune pi0 (LoRA) on **base+core** vs **base+random**
(identical recipe, only the +200 differs), eval both on a held-out region-stratified
set, compare. Data selection is **done** (`weakregion/arms.json`, `subsample_plan.json`).

> Honest status: everything except the pi0 *training* itself was built and verified on
> the 4070 (it can't train a 3B VLA). The configs below are adapted from openpi's working
> `pi0_robocasa_finetune_*` + `pi0_libero_low_mem_finetune` templates but are **untested on
> a training GPU** — run the smoke step first and expect to fix norm-stats / weight-loader paths.

---

## 1. Prereqs (rebuild the data locally — small, deterministic)
```sh
git pull                                   # gets build_lerobot_subset.py, build_arms.py, arms.json
# download the 10k MimicGen pool (~6 GB)
python -m robocasa.scripts.download_datasets --tasks PickPlaceCounterToSink --source mimicgen --split pretrain
MG=<.../v1.0/pretrain/atomic/PickPlaceCounterToSink/20250819/mg/demo/.../lerobot>
# rebuild the two fine-tune datasets from the index lists (identical to the 4070 build)
python policy_analysis/build_lerobot_subset.py --src "$MG" --arms weakregion/arms.json --which base+core   --dst /data/xinyua11/ft_arms/ppc2sink_base_core
python policy_analysis/build_lerobot_subset.py --src "$MG" --arms weakregion/arms.json --which base+random --dst /data/xinyua11/ft_arms/ppc2sink_base_random
```
Each = 600 demos (400 shared base + 200 arm). Verify they load:
`python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; print(LeRobotDataset('x', root='/data/xinyua11/ft_arms/ppc2sink_base_core').num_episodes)"`

## 2. Add the two configs to `openpi/src/openpi/training/config.py` `_CONFIGS`
```python
# --- ppc2sink targeted-vs-random LoRA fine-tunes (identical recipe, only data_dirs differ) ---
TrainConfig(
    name="pi0_ppc2sink_core",
    model=pi0_config.Pi0Config(max_token_len=96,
        paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
    data=LeRobotRobocasaDataConfig(
        data_dirs=["/data/xinyua11/ft_arms/ppc2sink_base_core"]),
    # fine-tune FROM the robocasa-pretrained pi0 (the 55% baseline), not pi0_base:
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "/data/xinyua11/.../pi0_robocasa_pretrain_human300/multitask_learning/75000/params"),
    freeze_filter=pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora").get_freeze_filter(),
    ema_decay=None,            # off for LoRA
    num_train_steps=20000, save_interval=5000, keep_period=5000,
    batch_size=32, num_workers=4,
),
TrainConfig(
    name="pi0_ppc2sink_random",   # IDENTICAL except data_dirs
    model=pi0_config.Pi0Config(max_token_len=96,
        paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
    data=LeRobotRobocasaDataConfig(
        data_dirs=["/data/xinyua11/ft_arms/ppc2sink_base_random"]),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "/data/xinyua11/.../pi0_robocasa_pretrain_human300/multitask_learning/75000/params"),
    freeze_filter=pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora").get_freeze_filter(),
    ema_decay=None,
    num_train_steps=20000, save_interval=5000, keep_period=5000,
    batch_size=32, num_workers=4,
),
```
Keep `num_train_steps`, `batch_size`, LoRA, seed **identical** across the two — only `data_dirs` differs.

## 3. Norm stats + train (smoke first!)
```sh
# norm stats per config (Groot loader reads them from the data dir; compute if missing)
python scripts/compute_norm_stats.py pi0_ppc2sink_core
python scripts/compute_norm_stats.py pi0_ppc2sink_random
# SMOKE: 50 steps to shake out norm-stats / weight-loader / LoRA issues before the real run
python scripts/train.py pi0_ppc2sink_core --exp_name smoke --num_train_steps 50
# real runs (both GPUs in parallel)
python scripts/train.py pi0_ppc2sink_core   --exp_name core_v1
python scripts/train.py pi0_ppc2sink_random --exp_name random_v1
```

## 4. Eval — baseline + core + random on a HELD-OUT stratified set
Eval uses the pi0 server + the per-category harness on fresh `gym.make` seeds (disjoint
from the n=500 student eval which used 0–499). Run for each of the 3 checkpoints:
```sh
# serve the checkpoint (XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl) on :8000, then:
MUJOCO_GL=egl python policy_analysis/analyze_pi0_weakregions.py \
    --n 300 --seed 100000 --out_dir weakregion/eval_<baseline|core|random>
```
- **baseline** = the un-fine-tuned pi0 (the 55% reference).
- Report **overall** success and **per-category**, focusing on the 10 targeted categories
  (juice, spray, pitcher, canned_food, soap_dispenser, tupperware, cheese_grater, ice_cube,
  cream_cheese_stick, jar).

## 5. Win condition
**core > random**, especially on the targeted categories, **without hurting the others**
(the coverage check). Also report per-demo improvement (Δsuccess / 200 demos).

## Known unknowns to expect
- **Norm stats:** the mg subset has LeRobot `stats.json` but the Groot data config expects
  groot-style norm stats — `compute_norm_stats.py` should produce what's needed; verify.
- **Weight loader:** loading robocasa-pi0 (non-LoRA) params into a `gemma_*_lora` model — openpi
  does this for `pi0_base`; should work from the robocasa ckpt too, but confirm no missing-key errors.
- **max_token_len / action_dim** must match the pretrained pi0 (robocasa configs use 96).
- If LoRA-from-robocasa-ckpt misbehaves, fall back to full fine-tune (drop the `_lora` variants +
  `freeze_filter` + set a low LR) — still valid as long as both arms use the identical recipe.
