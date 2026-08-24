# Success-Factor Analysis: Shape, Location, Layout/Style vs Object Category

**Date:** 2026-07-20
**Data:** all 28 eval runs pooled (7,112 episodes, `factor_analysis/pooled_episodes.csv`) — 27 pi0 checkpoints + 1 GR00T run, PickPlaceCounterToSink, 81 object categories, 50 layouts, 50 styles.
**Method:** every analysis controls for checkpoint (run fixed effects / within-run demeaning / run-stratified permutation). All 16 headline findings were independently re-derived by a second pass (all CONFIRMED). Scripts in `factor_analysis/scripts/`, plots in `factor_analysis/plots/`, full structured results in `factor_analysis/wf_result.json`.

---

## TL;DR

| Factor | Effect | Verdict |
|---|---|---|
| Object **category** | strongest single factor (CV R² 0.073, AUC 0.66); signal is REAL (2.4× null spread, split-half r=0.73) but per-category estimates carry ±12pp noise → looks patternless by eye | real, noisy per-category |
| Object **shape (height)** | grasp-phase only; inverted-U, peak SR at 4–8 cm, ramp down above ~10 cm, h>0.21 m costs −22pp; explains ~half the category signal, incl. within-category | real, mechanistic |
| Object **width** | ~nothing (AUC 0.541) | null |
| **Start location** | radial: outer rim of placement region (26% of episodes) −18.3pp; NO near/far or left/right gradient; absolute kitchen coords irrelevant | real, radial |
| **End location** | = sink, fully determined by layout | fixed given layout |
| Kitchen **style** | large + stable: true SD ~0.08 SR, best-worst ~0.31 SR, split-half r=0.69 across checkpoints; hurts grasp AND post-grasp (visual difficulty) | real, large |
| Kitchen **layout** | ~null after controlling category+style (split-half r=0.16, CV R² 0.002); hardness uncorrelated with counter→sink distance | ~null |

Full CV model (category+shape+location+style+layout): within-run AUC 0.703. Failure is 89% `fail_no_grasp` everywhere — all factors act mainly through the grasp phase.

---

## 1. Why you can't see a category pattern

- Median n = 58 episodes/category → Wilson 95% CI half-width ±11.7pp; 60/81 categories have half-width > 10pp. Mid-table ordering is unreadable **by construction**.
- But the signal is real: spread of per-category run-demeaned SR = 0.155 vs 0.063 under a within-run label-permutation null (2.4×, p=0.0005). True category-signal SD ≈ 0.14. Per-category SRs replicate across disjoint run halves (r=0.73) and disjoint eval configs (r=0.66).
- Only ~43% of the category signal is explained by measured shape (53% by shape+location+layout+style). The rest is category-idiosyncratic grasp affordance:
  - Hardest: kettle_non_electric (−0.49), blender_jug (−0.37), saucepan_lid (−0.33), water_bottle (−0.30), cheese_grater (−0.25), teapot (−0.24)
  - Easiest: canned_food (+0.28), cucumber, corn, potato (+0.25), tangerine (+0.24)
  - Pattern: tall / handled / lidded containers vs small compact graspables.

## 2. Object shape

- **Height is the only shape feature that matters** (within-run AUC 0.616; width 0.541; aspect is a height proxy). Entirely a **grasp-phase** effect: grasped-vs-not AUC 0.633; success-given-grasped AUC 0.509 (null).
- **Inverted-U**: peak SR at height 0.04–0.08 m (+0.11 demeaned), flat objects <4 cm slightly below peak, graded ramp down above ~0.10 m, −0.20 at 0.23–0.30 m. Quadratic h+h² AUC 0.656.
- Best single threshold **h > 0.212 m → −23.6pp SR** (grasp rate 0.59 → 0.20); survives out-of-sample threshold selection (−22.3pp).
- **Within-category height variation predicts failure with the same per-meter slope as between-category** (−5.6 vs −7.1 logit/m, p<0.003) → shape is causal-ish, not just a category label proxy.
- Head-to-head (held-out runs): category-only AUC 0.661 > shape-only 0.645; combined 0.669. Shape explains 53% of per-category SR variance (r = −0.587 with mean height).

## 3. Start/end location

- **The effect is radial, not directional.** Rim of the counter placement region (max(|x_rel|,|y_rel|)>0.65; 26% of episodes): run-adjusted SR 0.330 vs 0.513 interior (**−18.3pp**, p=1e-4, negative in 27/28 runs). 2D heatmap: corner cells 0.25–0.35 vs central 0.55–0.66 (`plots/loc_heatmap_xy.png`).
- No near/far gradient (−1.4pp, p=0.36) and no left/right gradient (p=0.15); the "column" effect is edge-vs-center (−5.5pp, p=1e-4).
- **Absolute kitchen coordinates add nothing** (ΔR² ≤ 0.0006) — it's position *within the placement region*, not kitchen-frame reachability.
- Same failure mechanism everywhere (no-grasp share of failures 87–90% in every band/column).
- Location ⊥ category/shape (randomization checked, p=0.55–0.67) → location and shape are **additive** (~4–5% of within-run variance each, together 9.5%).
- End location: the sink is a fixture at fixed coordinates in each layout yaml — pinning layout pins the goal exactly (verified: identical sink pos across seeds).

## 4. Kitchen layout / style

- **Style is the environmental factor that matters**: true SD ≈ 0.084 SR, EB-shrunk best-worst gap ≈ 0.31 SR, split-half reliability r=0.69 across disjoint checkpoint halves (survives controlling run+category+layout: r=0.67). Hardest styles: 49, 43, 42, 21, 22; easiest: 27, 36, 31, 30, 17. Style hurts grasp AND transport/place → broad visual/perceptual difficulty.
- **Layout is at best weak**: marginal spread evaporates after controlling category+style (p≈0.10, split-half r≈0.16, CV R² 0.002, added-last ≈0); no correlation with counter→sink distance (r=0.08, p=0.59). Caution: naive per-layout SR tables suffer winner's curse — the null best-worst gap at n≈142/id is already ~0.19 SR.

## 5. Variance decomposition (run-CV ridge, held-out runs)

| family | R² alone | AUC alone | added-last ΔR² |
|---|---|---|---|
| category | 0.073 | 0.660 | +0.028 |
| shape | 0.045 | 0.613 | +0.002 |
| location | 0.040 | 0.609 | +0.025 |
| style | 0.021 | 0.581 | +0.006 |
| layout | 0.002 | 0.528 | −0.005 |
| full | 0.116 | 0.703 | — |

Category and location carry unique signal; shape is absorbed by category (it's *how* category acts); layout is null.

---

## 6. Fixing layout/style/placement: YES — verified recipe

Verified by code-reading the fork + a 5-seed smoke test (layout/style/sink/robot-base identical across seeds, object XY spread 1.9×1.7 cm, object instance varying). Module: [policy_analysis/fixed_context.py](../policy_analysis/fixed_context.py).

**Knobs (all confirmed working):**
1. **Layout+style**: `layout_and_style_ids=[[11,11]]` (any pair, both in 11–60 = train split; 1–10 is test). Consumed at `kitchen.py:392-394,424-445,591-597`.
2. **CRITICAL**: `split='pretrain'` **silently overrides** layout/style ids (`env_utils.py:86-104`) → pass `split=None` on the `gym.make`/`create_env` path. For direct `robosuite.make`, **omit `split` entirely** (Kitchen.__init__ has no such kwarg → TypeError).
3. **Object start XY**: no kwarg exists; monkeypatch `PickPlaceCounterToSink._get_obj_cfgs` (`kitchen_pick_place.py:294-318`): `loc='left'` (kills the hidden left/right-of-sink coin flip), `size=(0.02,0.02)`, `ensure_object_boundary_in_range=False`, `rotation=0.0`. Residual XY jitter ~2 cm (uniform draw + 10-step physics settle). Alternative: subclass + robosuite auto-registration; or `env.set_ep_meta(saved_meta)` for exact replay.
4. **Robot base**: `robot_spawn_deviation_pos_x/pos_y/rot = 0.0` (defaults 0.15/0.05/0.0).
5. **Object sweep**: `obj_groups='<category>'` or an absolute mjcf xml path for one exact instance (`kitchen_object_utils.py:361-382`). Categories must be graspable+washable.
6. **Distractors** (`distr_counter`, `distr_sink`) are re-sampled from all categories every episode — pin their `obj_groups` in the same monkeypatch for strict object-only variation.
7. **Seeding**: env rng is seeded once at construction; fresh-env-per-seed = pure function of seed; or gym `reset(seed=k)` re-seeds per episode (`gym_wrapper.py:342-344`).

**Pipeline insertion points:** `record_pi0_demos.py:119` (gym.make + import fixed_context), same for `capture_rollouts.py:24` / `rollout_eval.py:63-70` / `build_eval_set.py:26-30`; add the kwargs into `build_env_info` so they land in hdf5 `env_args` (then mimicgen + convert_hdf5_lerobot inherit fixed context automatically — but the monkeypatch must be imported in every process that creates new episodes; use the subclass route to avoid that).

**Gotchas:** gym wrapper default `split='test'` raises in create_env — always pass split explicitly; tiny placement region + `ensure_object_boundary_in_range=True` raises PlacementError; per-layout left/right region feasibility varies (50 retries then RuntimeError); `hard_reset` is forced → ~10–20 s per reset; keep gen seeds (≥100000) disjoint from eval seeds (<1000) — note `record_pi0_demos.py` defaults to `--seed 100`, so pass the high seed explicitly.

---

## 7. What the data says you should actually fix

Recommendation given the goals (isolate object learnability; prescription arms):

1. **Fix style, not just layout.** Style is the large, stable env nuisance (±0.31 SR between styles); layout is statistically ~null but pinning it is free and pins the sink/goal. Pinning one (layout, style) pair removes the biggest env confound.
2. **Don't pin location to a point — control the rim.** The −18pp factor is rim-vs-interior of the placement region. For a *clean object probe*, pin XY. For *training data*, consider keeping interior variation (e.g. shrink `size` from (0.30,0.40) to ~(0.15,0.20)) rather than a single point, to avoid a policy that only works at one spot.
3. **Expect a comparability break.** Fixed-context arms are NOT comparable to prior arms (core/random/coverage trained+evaled under full variation). A single-(layout,style) policy will look better on matched fixed-context eval and likely worse on the standard `split='pretrain'` eval (distribution shift). Report both.
4. **Object-only variation is exactly the right instrument for the prescription/headroom-probe direction**: with style/layout/location/robot-base pinned, per-object success differences become interpretable learnability signal at ~90 eps/object noise levels instead of being swamped by ±12pp env noise — and the decomposition says category carries unique signal beyond shape, so sweep instances within category, not just height bins.

---

# Part 2: Missed-factors pass (same date, second verified workflow)

Full structured results: `factor_analysis/wf2_result.json`. 13/14 findings CONFIRMED by independent recomputation, 1 PARTIAL (corrected below).

## 8. REINTERPRETATION: the "rim" penalty is distance-from-sink along the counter

The two placement regions are narrow bands flanking the sink at lateral ±0.57 m; the depth coordinate never exceeds 0.58, so `rim>0.65` ≡ `|along-counter distance from sink|>0.65` with agreement exactly 1.0000. The gradient is monotone (+0.09 demeaned SR at the sink-near end → −0.30 at the far end; FE slope −0.35 along-counter vs −0.02 depth) and grasp/approach-mediated: the robot parks laterally centered on the sink, so far placements require a long lateral reach *before* grasping. **Prescribe demos at large along-counter distance from the sink**, not an undirected "rim band". Fig: `plots/side_fig.png`.

## 9. Hidden left/right-of-sink coin flip — recovered from data

The sampler's left/right region choice was never logged but is recoverable: 41/50 layouts cleanly bimodal in obj_xy_abs; side = sign of the separating rel coordinate. Signed effect: the **robot's-right side of the sink is ~4pp worse** (config-level p=0.004–0.018), consistent with policy/embodiment chirality (base parks centered; established for pi0 family only; GR00T n=83 uninformative). Layout-idiosyncratic side effects beyond this are weak (p=0.03 marginal). Side is orthogonal to the rim/location effect.

**Methodological warning (applies to ANY analysis of the pooled CSV):** episode-level permutation tests are anticonservative because ~1,000 configs repeat across runs with correlated outcomes — the unsigned side test gives p=0.0026 at episode level but p≈0.32 at the correct config level. Cluster by config signature.

## 10. No compounding weak pocket — factors are additive

tall×rim is mildly SUB-additive on the probability scale (+5pp better than additive prediction; log-odds-additive describes the joint cell within 1.1pp); tall×hard-style and rim×hard-style are null with power to detect ±5–6pp; interaction terms add exactly zero held-out R²/AUC (0.1187→0.1182). Treat height, along-counter distance, style as independent targeting axes.

## 11. Variance ceiling and the enumerable hard-config set

Run-demeaned success variance splits: **measured factors 11.9% / config-idiosyncratic 10.3% (perm p=0.002) / stochastic ~78%** (1,178 configs repeated across ≥2 runs; the 10.3% is a lower bound since repeats span different checkpoints). Any config-based predictor tops out at R²≈0.22 — unmeasured config properties (instance mesh, yaw, distractors) carry about as much signal as all modeled factors combined. Among repeated configs: **28% never succeed under any checkpoint** (enumerable hard-config target list), 16% always succeed, 55% mixed; failure MODE repeats (0.926 modal phase agreement vs 0.811 chance).

## 12. Failures are early; timeout censoring is small; two dissociable mechanisms

Horizon = 600 steps; one more 60-step window would rescue only ~2.6% of failures (+1.4pp); only 1.5% of failures ever got within 10 cm of the sink. Taxonomy of 3,803 failures: 50% never moved the object >5 mm, 29% nudged (0.5–2 cm), 9% partial lift, 5% grasped-stalled-far, 4% carried-near, 2% reached-sink-no-place. Mechanism dissociation: **far-from-sink placements fail by never touching** (+7.8pp never-touched, approach failure) while **tall objects fail by touch-then-fumble** (−9.7pp never-touched but no grasp; driven by graspable-body tall categories — handle-topped ones like kettle are never-touched). Failure depth is a stable config property (cross-run r=0.30) and height predicts it (causal reinforcement). Among successes, rim (+27 steps), hard styles, hard categories are also *slower* (difficulty continuum); height is not — a discrete grasp gate. Prescription: approach/reach demos for far placements, grasp-securing demos for tall objects; place-stage data addresses ~2% of failures. Plots: `plots/nearmiss_taxonomy_hists.png`, `plots/nearmiss_steps_hist.png`.

## 13. Arm × factor: targeted arms lifted the weak REGION, not the weak CELLS

The most decision-relevant result (`plots/arms_weakcell_summary.png`):
- In the well-powered STRAT eval (~280 eps/arm on targeted-10): core +11.0pp over baseline (p=0.008) but **random +8.9pp (p=0.024); core−random only +2.0pp (p=0.89)** — most of the weak-region lift is generic fine-tuning. (BALCAT targeted subset: core−random +3.8pp, also small.) The family-E "+15pp core-over-random on targeted" was small-n (n≈38) noise.
- **No arm flattened the tall cliff**: tall SR baseline 0.09 → core 0.21 (best); saturate with ALL 610 targeted demos only 0.14. The tall-vs-mid penalty *grew* under core (−30 → −38pp) because mid-height objects gained most. Tall failures remain absolute never-lift failures in every arm.
- **Rim penalty untouched** in every arm (pooled DD −0.4pp, p=1.0; core's rim gain +11.2pp = its center gain +11.1pp — uniform lift). The "influence arm worsens rim" signal is a non-robust uncorrected trend (PARTIAL verdict), not a finding.
- **Hard-style gap unaddressed** by any arm (all interactions ns, trends ≤0).
- **No dose-response** within core's allocation: demos-per-category vs per-category gain r=+0.06 (p=0.87).
- Core's real edge = region-level concentration + **coverage preservation** (non-targeted slice: core −0.1pp vs coverage −8.9, influence −4.4, value −9.4).

Paper implication: validates region-level prescription with coverage preservation as the honest headline; deflates cell-level "fix the weakness" claims; strongest quantitative support yet for the generation-ceiling argument (residual weak cells are data-resistant to reallocating the fixed sim pool → human/G1 new-data sources).

## 14. Nothing is permanently lost — metadata is recoverable by seed replay

`env.get_ep_meta()` already exposes object mjcf path, both distractor cfgs (identity+region), robot base pose, cam configs, textures, lang — the harness kept only 6 fields (`analyze_pi0_weakregions.py:197-207`). No saved eval artifact has instance/yaw/distractor/base-pose. BUT scene generation is fully deterministic in the env seed (seed = base_seed + episode; base seeds: 0 default, 100000 paired arms, 300000 failstates, or stored in `phase1_data/eval_set.json`), so a seed-replay script can re-instantiate every past episode, verify the recorded fingerprint (category/layout/style/obj_xy), and extract full metadata — no policy re-run needed. Training hdf5s already carry everything (ep_meta + full model XML incl. mass/friction; start yaw recoverable from states[0]). Forward patch spec (3 harness files: target_meta + records.append in analyze_pi0_weakregions.py, rec dict in rollout_eval.py) is in `factor_analysis/wf2_result.json` (metagaps). Priority for hunting the residual variance: object instance > yaw > distractors > robot base > mass.

## Remaining gaps (not answerable from existing data)

- Cross-policy universality of the side/style/rim refinements: only one 100-episode GR00T run — underpowered for everything except the main height effect.
- No evals on the test split (layouts 1–10 / unseen styles): generalization of factor structure unknown.
- Instance/yaw/distractor effects: recoverable via seed replay (section 14) but not yet extracted — the single highest-value next analysis given the R²≈0.22 ceiling vs 0.12 achieved.
