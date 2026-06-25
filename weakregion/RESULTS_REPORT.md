# Results Log — Targeted vs Random Data (pi0 LoRA), PickPlaceCounterToSink

**Log entry:** 2026-06-24 · H100 box · run by Xinyuan
**Experiment:** targeted (core) vs random data selection — pi0 LoRA fine-tune, per-category eval
**Student policy:** pi0 (openpi, robocasa-pretrained `pi0_robocasa_pretrain_human300`, step 75000, ~55% baseline)
**TL;DR:** core **59.3%** vs random **56.3%** overall; on the 10 targeted categories core **43.2%** vs random **28.2%** (+15 pp), coverage preserved — **directional win, underpowered at n=300.**

---

## 1. Hypothesis & method
Adding training demos **where the policy fails** improves it more — per demo — than adding the **same number of random** demos. The core algorithm: run the student on many scenes, estimate per-**object-category** failure rate (Wilson lower-bound), allocate a fixed demo budget to the highest-failure categories (capped by pool availability), subsample those as the **core** arm; an equal-size uniform-random draw is the **random** control. Then **LoRA-fine-tune pi0** on `base+core` vs `base+random` with an **identical recipe** (only the data differs) and compare per-category on a held-out set.

## 2. Setup
- **Pool:** RoboCasa MimicGen, **9,885 demos** across 79 object categories.
- **Student weak-region eval:** pi0 on 500 scenes → **54.8%**, per-category failure rates → drove the allocation.
- **Three fine-tune datasets, 600 demos each** = **400 shared base** + 200 arm:
  - **core** = 200 demos concentrated on the 10 highest-failure categories.
  - **random** = 200 demos uniform over the pool (lands on common/easy objects).
- **Training:** pi0 LoRA (gemma_2b_lora + gemma_300m_lora), max_token_len=96, **20,000 steps, batch 32, ema off**, fine-tuned FROM the 55% robocasa-pretrained pi0. Identical for both arms; only `data_dirs` differ. ~8 h each (parallel, one per GPU).
- **Eval:** each model served (openpi websocket) and run for **n=300 rollouts** on held-out `gym.make` seeds (start **100000**, disjoint from the n=500 student eval which used 0–499), via `analyze_pi0_weakregions.py`. Per-episode success joined with object category.

## 3. The targeted categories (= the core arm) and arm compositions
**Core arm — 200 demos across exactly the 10 targeted (highest-failure) categories:**

| category | core demos | (plan) |
|---|---|---|
| juice | 29 | 15 |
| spray | 28 | 14 |
| pitcher | 24 | 12 |
| canned_food | 21 | 11 |
| soap_dispenser | 19 | 10 |
| tupperware | 17 | 9 |
| cheese_grater | 16 | 8 |
| ice_cube | 16 | 8 |
| cream_cheese_stick | 15 | 12 |
| jar | 15 | 8 |
| **total** | **200** | |

**Random arm — 200 demos spread across 67 categories** (natural pool distribution; top: mango 7, cucumber 7, eggplant 6, whisk 6, peeler 6, broccoli 6, rolling_pin 6, strainer 5, lemon 5, reamer 5, pear 5, cup 5, bell_pepper 5, tomato 5, …). Only 3 of the 10 targeted categories appear in the random arm at all (juice, cheese_grater, cream_cheese_stick are core-only).

**Base (shared) — 400 demos across 71 categories** (identical in both arms).

(Note: a stale `subsample_plan.json` lists a 25-category spread; the arm actually built/used is the **top-10 concentrated** version above, matching the runbook's "10 targeted categories.")

## 4. Headline results (n=300 each)

| | overall | **targeted (10-cat aggregate)** | non-targeted (coverage) |
|---|---|---|---|
| **baseline** (pi0, no FT) | 58.0% (174/300) | 33.3% (13/39) | 61.7% (161/261) |
| **core** (base+targeted) | **59.3% (178/300)** | **43.2% (16/37)** | 61.6% (162/263) |
| **random** (base+random) | 56.3% (169/300) | 28.2% (11/39) | 60.5% (158/261) |

Overall CI95: baseline [52.3, 63.4], core [53.7, 64.7], random [50.7, 61.8].

**All three checks point the hypothesized way:**
1. **core > random on the targeted categories: +15.0 pp** (43.2% vs 28.2%).
2. **core lifts the targeted regions** (+9.9 pp over baseline); **random does not** (−5.1 pp vs baseline — random data didn't help where the policy is weak).
3. **Coverage preserved** — core ≈ baseline on the 67 non-targeted categories (61.6% vs 61.7%); it didn't trade away the rest.
4. Overall: core (59.3%) > baseline (58.0%) > random (56.3%).

Per-category win/loss on the 10 targeted: **core beats random on 5, loses on 2, ties on 3.**

## 5. Win condition
**Met directionally:** core > random, concentrated on the targeted categories, without hurting the others. **Not statistically conclusive** — see §7.

## 6. Full per-category accuracy (all categories, n=300 each; cell = success% successes/n)

| category | baseline | core | random |
|---|---|---|---|
| apple | 33% 1/3 | 25% 1/4 | 75% 3/4 |
| avocado | 100% 2/2 | 100% 3/3 | 67% 2/3 |
| banana | 67% 2/3 | 75% 3/4 | 75% 3/4 |
| bar_soap | 100% 3/3 | 100% 3/3 | 100% 4/4 |
| beer | 67% 2/3 | 67% 2/3 | 50% 1/2 |
| bell_pepper | 100% 4/4 | 100% 3/3 | 75% 3/4 |
| blender_jug | 50% 2/4 | 0% 0/4 | 0% 0/4 |
| bottled_drink | 40% 2/5 | 50% 3/6 | 60% 3/5 |
| bottled_water | 67% 4/6 | 50% 3/6 | 0% 0/5 |
| bowl | 67% 2/3 | 50% 1/2 | 33% 1/3 |
| boxed_drink | 33% 1/3 | 67% 2/3 | 33% 1/3 |
| broccoli | 67% 2/3 | 50% 2/4 | 75% 3/4 |
| can | 67% 2/3 | 100% 4/4 | 75% 3/4 |
| **canned_food** ⊛ | 100% 1/1 | 100% 1/1 | 100% 1/1 |
| carrot | 67% 2/3 | 67% 2/3 | 67% 2/3 |
| cheese | 60% 3/5 | 83% 5/6 | 80% 4/5 |
| **cheese_grater** ⊛ | 0% 0/3 | 0% 0/2 | 0% 0/3 |
| chicken_drumstick | 80% 4/5 | 100% 5/5 | 60% 3/5 |
| colander | 67% 4/6 | 83% 5/6 | 25% 1/4 |
| condiment_bottle | 100% 3/3 | 100% 3/3 | 0% 0/3 |
| corn | 90% 9/10 | 89% 8/9 | 60% 6/10 |
| **cream_cheese_stick** ⊛ | 20% 1/5 | 20% 1/5 | 0% 0/5 |
| cucumber | 60% 3/5 | 100% 4/4 | 100% 6/6 |
| cup | 50% 2/4 | 50% 2/4 | 75% 3/4 |
| dish_brush | 100% 3/3 | 100% 2/2 | 100% 2/2 |
| egg | 100% 2/2 | 100% 2/2 | 100% 2/2 |
| eggplant | 100% 7/7 | 86% 6/7 | 57% 4/7 |
| fish | 50% 1/2 | 100% 2/2 | 100% 2/2 |
| garlic | 100% 3/3 | 100% 4/4 | 75% 3/4 |
| glass_cup | 100% 2/2 | 0% 0/2 | 33% 1/3 |
| **ice_cube** ⊛ | 50% 3/6 | 43% 3/7 | 12% 1/8 |
| jam | 20% 1/5 | 40% 2/5 | 50% 2/4 |
| **jar** ⊛ | 0% 0/3 | 50% 1/2 | 100% 2/2 |
| jug | 50% 2/4 | 0% 0/6 | 25% 1/4 |
| jug_wide_opening | 50% 2/4 | 25% 1/4 | 20% 1/5 |
| **juice** ⊛ | 0% 0/1 | 0% 0/1 | 0% 0/1 |
| ketchup | 0% 0/2 | 0% 0/2 | 50% 1/2 |
| kettle_non_electric | 0% 0/1 | 0% 0/1 | 0% 0/1 |
| kiwi | 50% 1/2 | 100% 2/2 | 100% 1/1 |
| ladle | 80% 4/5 | 83% 5/6 | 60% 3/5 |
| lemon | 75% 3/4 | 100% 3/3 | 100% 3/3 |
| lemon_wedge | 80% 4/5 | 60% 3/5 | 20% 1/5 |
| lime | 0% 0/3 | 33% 1/3 | 67% 2/3 |
| mango | 100% 3/3 | 75% 3/4 | 67% 2/3 |
| measuring_cup | 80% 4/5 | 0% 0/2 | 100% 5/5 |
| milk | 50% 1/2 | 0% 0/2 | 50% 1/2 |
| mug | 100% 1/1 | 50% 1/2 | 100% 1/1 |
| mushroom | 67% 2/3 | 75% 3/4 | 100% 4/4 |
| onion | 50% 2/4 | 80% 4/5 | 67% 4/6 |
| orange | 75% 3/4 | 20% 1/5 | 67% 2/3 |
| peach | 75% 6/8 | 67% 4/6 | 71% 5/7 |
| pear | 57% 4/7 | 86% 6/7 | 75% 6/8 |
| peeler | 67% 2/3 | 100% 2/2 | 67% 2/3 |
| **pitcher** ⊛ | 57% 4/7 | 60% 3/5 | 20% 1/5 |
| pizza_cutter | 0% 0/1 | – | – |
| potato | 80% 4/5 | 80% 4/5 | 100% 5/5 |
| reamer | 0% 0/4 | 25% 1/4 | 0% 0/4 |
| rolling_pin | 50% 2/4 | 100% 3/3 | 100% 2/2 |
| salt_and_pepper_shaker | 67% 2/3 | 50% 1/2 | 100% 1/1 |
| saucepan | 60% 3/5 | 20% 1/5 | 0% 0/5 |
| saucepan_lid | 25% 1/4 | 33% 1/3 | 0% 0/4 |
| shrimp | 33% 1/3 | 25% 1/4 | 25% 1/4 |
| **soap_dispenser** ⊛ | 60% 3/5 | 60% 3/5 | 17% 1/6 |
| sponge | 33% 2/6 | 83% 5/6 | 83% 5/6 |
| **spray** ⊛ | 0% 0/2 | 100% 1/1 | 0% 0/1 |
| squash | 100% 2/2 | 100% 3/3 | 50% 1/2 |
| steak | 0% 0/1 | 0% 0/2 | 100% 1/1 |
| strainer | 75% 3/4 | 75% 3/4 | 75% 3/4 |
| straw | 0% 0/3 | 33% 1/3 | 67% 2/3 |
| sweet_potato | 83% 5/6 | 50% 3/6 | 86% 6/7 |
| tangerine | 100% 3/3 | 75% 3/4 | 50% 2/4 |
| teapot | 50% 1/2 | 50% 1/2 | 0% 0/2 |
| tomato | 67% 2/3 | 67% 2/3 | 100% 2/2 |
| tongs | 0% 0/5 | 80% 4/5 | 60% 3/5 |
| **tupperware** ⊛ | 17% 1/6 | 38% 3/8 | 71% 5/7 |
| water_bottle | 0% 0/1 | 0% 0/2 | 0% 0/1 |
| whisk | 50% 2/4 | 0% 0/1 | 100% 2/2 |
| wine | 0% 0/3 | 25% 1/4 | 100% 3/3 |
| wooden_spoon | 50% 2/4 | 25% 1/4 | 60% 3/5 |
| yogurt | 80% 4/5 | 50% 2/4 | 60% 3/5 |

⊛ = one of the 10 targeted categories.

## 7. Honest caveats (read before claiming a win)
- **Small per-category n.** 300 episodes / ~79 categories → the 10 targeted categories hold only ~37–39 episodes total (denominators of 1–8 per cell). Individual cells flip 20–100 pp on a single rollout.
- **Significance.** The +15 pp targeted effect (core vs random) has SE ≈ 11 pp (≈1.4σ) — **directional, not significant.** Overall CIs overlap heavily.
- **Consistency is the encouraging part:** the same direction shows up in *all four* summaries (targeted aggregate, vs-baseline lift, coverage preserved, overall), which is more than a single noisy number — but it needs more episodes (n≥800, or a targeted-category-stratified eval) to be defensible.
- **Uncertainty term was inert** (per the core-algorithm report): the selection here is effectively **competence-based** (P(fail) only).

## 8. Status / notes
- A third arm, **`coverage`** (diversity guard), was fine-tuned in parallel and is being evaluated separately (`pi0_ppc2sink_coverage`); it can be folded into this table when done.
- Artifacts: eval JSON/reports in `weakregion/eval_{baseline,core,random}/`; fine-tune checkpoints in `openpi/checkpoints/pi0_ppc2sink_{core,random}/{core,random}_v1/19999`; datasets in `/data/xinyua11/ft_arms/`.
- Invariant held: identical recipe across arms; only the +200 demos differ.
