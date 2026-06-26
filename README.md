# Targeted-Data Imitation Learning on RoboCasa

Research bundle testing a single question:

> Per demonstration, does training data **targeted at a policy's failure regions** improve a
> robot policy more than the **same number of randomly sampled** demonstrations?

Task: RoboCasa `PickPlaceCounterToSink`. Student policy: **π0** (openpi, robocasa-pretrained
`pi0_robocasa_pretrain_human300` @ step 75000, ~55% baseline), LoRA fine-tuned. The
demonstration pool is RoboCasa MimicGen: **9,885 demos across 79 object categories**.

## Method

1. **Weak-region detection.** Run the student on many held-out scenes and estimate the
   per-object-category failure rate. π0 fails predominantly at the *grasp* phase, concentrated
   on a stable set of object categories.
2. **Selection.** Pick a fixed **200-demo budget** from the pool by some strategy (the *core*
   arm), plus an equal-size uniform-random draw (the *control*).
3. **Fine-tune.** LoRA-fine-tune π0 on `base(400 shared) + arm(200)` with an **identical recipe**
   — only the 200 selected demos differ — for 20k steps.
4. **Evaluate.** Serve each fine-tuned policy and roll out on held-out scenes; compare success
   per object category, with emphasis on the targeted (highest-failure) categories.

The invariant across every arm: **same base, same recipe, same norm stats — only the 200
selected demos change.** This isolates the effect of *which* data was selected.

## Selection methods (the arms)

| arm | how the 200 are chosen | script |
|---|---|---|
| `random` | uniform over the pool (control) | `policy_analysis/build_arms.py` |
| `core` | P(fail) heuristic — concentrate on the **top-10** failure categories | `acquisition.py` → `build_arms.py` |
| `coverage` | P(fail) spread across the **top-25** failure categories | `acquisition_subsample.py` → `build_arms.py` |
| `retrieval` | CLIP/DINOv2 visual similarity to the scenes the policy fails in | `score_retrieval.py` / `score_pool.py` |
| `submod` | submodular facility-location coverage of the failure set (DINOv2) | `score_pool.py` |
| `influence` | **LESS / TracIn** gradient influence on the failure region | `influence_score.py` |

Each scorer emits an episode-index list (`weakregion/*_core.json`); `build_arms.py` turns it into
arm/base lists (`weakregion/arms_*.json`) and `build_lerobot_subset.py` materializes a standalone
LeRobot fine-tune dataset.

## Influence selection (LESS / TracIn)

`policy_analysis/influence_score.py` scores each pool demo by how much fine-tuning on it moves π0
in a direction that reduces loss on its high-failure categories — a gradient-based, principled
alternative to the category heuristic.

- **Gradient** per demo `z` at training checkpoint `t`: the (normalized) gradient of π0's
  flow-matching loss w.r.t. the LoRA adapters, averaged over the trained checkpoints
  (5k/10k/15k/19999) and over `K` frames per demo. (Last-layer-only `action_out_proj` gradients
  were tried first but carry no object identity — they are grasp/motion-dominated — so the LoRA
  adapters, the params actually updated during fine-tuning, are used.)
- **Failure direction** `g_val = normalize(mean_hard − mean_ref)`: the mean gradient over a
  held-out set of failure-category demos, *minus* the mean gradient of a random reference sample.
  The subtraction cancels the common-mode "generic pick-place" gradient that dominates every demo
  and leaves the failure-*specific* direction.
- **Score** `influence(z) = <g(z), g_val>`; select the top-200.
- **Tractability.** With both gemma variants LoRA, the trainable set is large (LoRA + the full
  vision tower + heads), so a fixed Gaussian projection cannot be materialized. The scorer instead
  *streams*: it builds `g_val` first, then dots each candidate's gradient on-device and keeps only
  the scalar — no projection, bounded memory.
- **Validation gate.** A held-out hard-vs-easy separation check (`smoke` mode) must show targeted
  demos scoring above non-targeted ones before scaling to the full pool.

Run incrementally: `probe` (integration sanity) → `smoke` (separation gate) → `full` (score pool).

## Results (π0 LoRA, n=300 held-out rollouts per arm)

| arm | overall | targeted-10 categories | non-targeted (~69 cats) |
|---|---|---|---|
| baseline (no fine-tune) | 58.0% | 33.3% | 61.7% |
| **core** (top-10) | **59.3%** | **43.2%** | 61.6% |
| random | 56.3% | 28.2% | 60.5% |
| coverage (top-25) | 51.7% | 42.9% | 52.8% |

Headlines: **core > random on the targeted categories by +15 pp** (43.2% vs 28.2%) while
preserving the non-targeted majority — a directional win for failure-targeting (underpowered at
n=300). **Spreading the budget across the top-25 (coverage) hurts**: it keeps the targeted benefit
but causes collateral forgetting on the non-targeted majority (52.8% vs core's 61.6%, p<0.05),
sinking overall to the worst of the four. **Budget concentration on the top failure modes beats
broad coverage.** The retrieval / submodular / influence arms extend this comparison with
feature- and gradient-based selection. See [weakregion/RESULTS_REPORT.md](weakregion/RESULTS_REPORT.md)
and [weakregion/CORE_ALGORITHM_REPORT.md](weakregion/CORE_ALGORITHM_REPORT.md) for the full
per-category tables and statistical caveats.

## Repo layout

```
policy_analysis/        selection + analysis scripts
  analyze_pi0_weakregions.py   student weak-region eval; per-category success; stratified eval
  acquisition*.py              P(fail) failure-rate estimation + budget allocation
  build_arms.py                selection list -> core/random/base episode lists (arms_*.json)
  build_lerobot_subset.py      episode lists -> standalone LeRobot fine-tune dataset
  score_retrieval.py           CLIP failure-retrieval selection
  score_pool.py                retrieval + submodular-coverage selection (DINOv2/CLIP)
  influence_score.py           LESS / TracIn gradient-influence selection
weakregion/             selection outputs, eval JSON, and reports
  subsample_plan.json          P(fail) per-category budget allocation
  arms*.json                   per-arm core/random/base episode lists
  {retrieval,submod,influence}_core.json   per-method selected episode lists
  eval_*/                       per-arm rollout results (analyze_pi0_weakregions.py output)
  RESULTS_REPORT.md             curated results + caveats
transfer/               standalone runner scripts + patches for the training box
```

Large/regenerable artifacts (checkpoints, LeRobot datasets, demo pools, embeddings) are
git-ignored and re-fetched/rebuilt on the training box; see `H100_HANDOFF.md`.

## Reproducing an arm end-to-end

```bash
MG=<path to the MimicGen LeRobot pool>

# 1) select 200 demos (example: influence). Other scorers: score_retrieval.py / score_pool.py
python policy_analysis/influence_score.py full --mode_grad lora --gval contrast \
    --real_dim 12 --k 8 --n_ref 300 --n_select 200 --out weakregion/influence_core.json

# 2) turn the selection into arm/base lists (reuse the exact base from arms.json)
python policy_analysis/build_arms.py --core_file weakregion/influence_core.json \
    --mg_dir $MG --base_file weakregion/arms.json --out weakregion/arms_influence.json

# 3) materialize the 600-demo LeRobot fine-tune set
python policy_analysis/build_lerobot_subset.py --src $MG \
    --arms weakregion/arms_influence.json --which base+core --dst <ft dir>

# 4) LoRA fine-tune (openpi TrainConfig, identical recipe; only data_dirs differ), then
# 5) serve + evaluate with analyze_pi0_weakregions.py and compare per category.
```

The π0 fine-tune lives in a separate openpi checkout (`TrainConfig`s `pi0_ppc2sink_{core,random,
coverage,influence}`, identical except `data_dirs`). The weak-region detection methodology also
replicates across policies (GR00T, Diffusion Policy) — the tall-object grasp weakness is shared,
i.e. a data gap rather than an embodiment limit.
