# Stationary Data-Prescription Bandit — v1 Design (frozen 2026-07-23)

**Status: approved design.** This binds the owner's v1 spec to concrete repo assets and records the
decisions resolved on 2026-07-23. It supersedes, as active plans: `HEADROOM_PROBE_DESIGN.md` (killed,
never ran — no pull is inherited from it), the arm definitions and Gate-Race-Confirm scheduler of
`MAB_ARM_DESIGN.md`, and `ANCHOR_PIPELINE.md`. `prescription_compare/` (the calibrated sandbox)
remains reference material only; no code path from it is on the critical path.

**Purpose** (unchanged from the spec): identify which *type* of additional training data most improves
a base policy, via repeated independent experiments. A base policy is trained once; each bandit pull
finetunes from the same pretrained base on `D0` + that pull's fresh in-region demos, evaluates on a
frozen set, and scores the improvement. Every pull starts identically, so arm values are fixed
quantities and the bandit is genuinely stationary: best-arm identification, then exploitation.

Task: **PickPlaceCounterToSink**, policy **pi0 (LoRA)**, on the local 2×H100 box.

---

## 0. Decisions resolved 2026-07-23 (owner)

1. **Zero inherited pulls.** The 4-arm Hard/Middle/Easy/Random headroom probe was design-only and was
   killed before implementation; nothing on disk corresponds to it. v1 starts at pull 0 and reuses
   only components (env, recipe, eval harness, retrieval tooling).
2. **B is raised from 20; chosen by rule at arms-freeze.** With the retrieval channel, demos are free —
   pull cost is the finetune, independent of B. Rule: **B = max{b ∈ {200, 100, 60, 20} : min over
   retained arms of |W ∩ region_k| ≥ 3b}** (target 200). Equal for every arm, frozen before pull 1,
   never changed. Rationale: +200-demo arms produced ~+11pp vs base and +2pp (n.s.) core-vs-random;
   +20-demo effects would sit inside the measured noise floor.
3. **E is saved-state, 150 starts, ×3 repeats per checkpoint** (450 rollouts per eval). Measured fact:
   same-seed scene replay across processes reproduces the exact scene only ~2/3 of the time (recurring
   ~1.17 m left/right-of-sink offsets, occasional category flips), so a seed list is NOT a frozen eval
   set. E is materialized as saved states restored via `env.reset_to` (pattern:
   `policy_analysis/check_train_eval_disjoint.py`).
4. **Arms come from data-driven clustering** (spec §2) over a fresh diagnosis batch of `pi_0` — not
   from the hand-named mechanism arms of `MAB_ARM_DESIGN.md`. **style_id is excluded from the
   descriptor**: it is unrecoverable for pool demos (the MimicGen LeRobot conversion carries no
   ep_meta), so a style arm could not be retrieved. Style is still logged eval-side and reported as a
   limitation (it is a large factor, best–worst ≈ 0.31 SR).
5. **Channel: retrieval only, W = pool − D0 = 9,485 demos.** The MimicGen backup stays in the spec as
   the escalation path in principle, but it is not wired on this box and historically scored 0/20 on
   this task; if the escalation trigger ever fires, the run halts for an explicit decision — channels
   are never mixed within a run. W does not exclude episodes consumed by pre-v1 arms (those trained
   other checkpoints, not `pi_0`; arm distributions must not be distorted by history).
6. **`pi_0` must be trained.** `pi0_ppc2sink_basewarmup/warmup_v1` only ran to 6k steps (it was the
   influence experiment's neutral reference, not a base policy). `pi_0` = full recipe on `D0` alone.

---

## 1. Frozen assets (create once, never modify)

1. **Pool**: `/data/xinyua11/robocasa_pkg/datasets/v1.0/pretrain/atomic/PickPlaceCounterToSink/20250819/mg/demo/2025-08-20-22-32-27/lerobot`
   — 9,885 MimicGen episodes, 79 object categories.
   **D0** = the 400 episode indices in `weakregion/arms.json:base_episodes`, already materialized as
   `/data/xinyua11/ft_arms/ppc2sink_base_only`. **W = pool − D0** (9,485 episodes).
2. **Recipe R** (identical to all 11 prior arms; openpi conda env, `scripts/train.py`):
   `Pi0Config(max_token_len=96, paligemma_variant='gemma_2b_lora', action_expert_variant='gemma_300m_lora')`,
   matching LoRA freeze filter, `weight_loader = /data/xinyua11/checkpoints/pi0/pi0_robocasa_pretrain_human300/multitask_learning/75000/params`,
   20k steps, batch 32, AdamW + cosine (1k warmup, peak 2.5e-5 → 2.5e-6), EMA off.
   **Seed is passed per-run via `--seed`** (tyro CLI; default 42 is never used by v1).
   Paired seeds: **s_j = 1000 + j**; pull j of every arm uses s_j.
3. **`pi_0`** = finetune(pretrain-75k, D0, R, seed=1000). One 8h run; its checkpoint id is recorded in
   `config.yaml` and never retrained.
4. **Eval set E**: 150 saved-state starts (per start: ep_meta JSON + model XML + MuJoCo state vector),
   sampled from fresh env randomization at seed base **500000** (disjoint from every prior eval base:
   0, 1–137, 100000, 300000, and from the diagnosis base 600000), stratified ≈50/50/50 by
   `p_hat_0` terciles (easy/mid/hard). Restored via `env.reset_to`; scored by the task's scripted
   success check at the standard horizon. Frozen (hashed) before any pull.
   **Day-one gate**: validate end-to-end that `reset_to` on a saved start reproduces the identical
   scene and a deterministic scripted-success readout across processes, before E is built. If
   restoration proves unreliable, fallback = fingerprint-verified seed lists (re-scan and discard
   mismatches at eval time), and the design note must record the switch.
   ε-exclusion: no demo used in any pull may be a near-duplicate of an eval start —
   **same object category ∧ |Δ(x,y)| < 0.10 m in sink-relative coordinates** (generous because pool
   x/y are anchor-recovered with error).
5. **Baseline b**: `pi_0` on E, 3 independent repeats (the same ×3 protocol as every pull — b's error
   is shared by all deltas and cancels in the ranking; its 3 repeats double as the eval-only noise
   measurement).
6. **Noise floor σ_e** (measured before any ranking is interpreted):
   - eval-only: spread of paired per-start outcomes across the 3 baseline repeats of `pi_0`;
   - finetune+eval: **2 null pulls** = recipe R on `D0` alone at seeds s_1=1001 and s_2=1002,
     evaluated exactly like pulls. Their deltas vs b are the honest zero point. (No two prior
     finetunes ever differed only in seed — this measurement is new.)

---

## 2. Round 0 — diagnosis, map, clusters, arms

1. **Condition space X** (logged per episode, everywhere): object category (81-way) and exact instance
   mjcf path, object height/width, sink-relative x/y and side, yaw, layout_id, style_id, robot base
   spawn pose, seed. Pool-demo features (already recovered in
   `weakregion/factor_analysis/fx_pool.json`, 9,885 rows): category, h, w, layout, sink-relative x/y,
   side. The clustering descriptor uses the intersection of eval-side and pool-side features.
2. **Diagnosis batch**: N=300 conditions × m=8 rollouts of `pi_0` (2,400 rollouts, ~27–37h, pipelined
   with other GPU work). There is no Sobol here — the influential knobs are categorical and the env
   samples scenes from its seed — so space-filling = **stratified seed-scan**: scan fresh seeds (base
   600000), keep conditions to fill a balanced grid over (category-difficulty tercile from the prior
   per-category table) × (height bin) × (|along-counter offset from sink| bin); save each kept
   condition as a saved state; run m=8 via `reset_to`. Per episode log: full x, binary success,
   failure signature, `max_lift`, `min_sink_dist`, and **new: minimum end-effector-to-object
   distance**, which splits never-reached from reached-no-grasp. Failure signature = existing 4-stage
   classifier (`analyze_pi0_weakregions.py`: success / fail_no_grasp / fail_grasped_no_transport /
   fail_reached_sink_no_place) refined by the EE-distance split → 5 stages matching the spec's list
   (placed-wrong stays folded into reached-sink-no-place; it is 2% of failures).
3. **Difficulty map `p_hat_0(x)`**: L2 logistic regression with interactions; category enters
   target-encoded (per-category diagnosis rate with shrinkage toward the global mean), continuous
   knobs standardized with a quadratic height term. Validate: held-out log-loss + calibration curve;
   sanity-check against known effects (height inverted-U with cliff at h>0.212 m, −18pp beyond 0.65 m
   along-counter offset, category ordering). Known ceiling: config-based predictability tops out at
   R² ≈ 0.22 / AUC ≈ 0.70 (~78% of outcome variance is rollout-stochastic) — do not over-tune past it;
   never threshold raw per-condition fractions. Alongside `p_hat_0`, fit a multinomial **stage model
   `p_stage(x)`** on the same features; both models, not empirical per-condition fractions, supply the
   descriptor blocks below, so condition-space and pool-demo-space embeddings are commensurable.
4. **Cluster into arms**: per-condition descriptor
   **z = [standardized (h, w, x_rel, y_rel, side), p_hat_0(x), p_stage(x) (5-dim)]**
   (blocks standardized; style/layout excluded — style unrecoverable pool-side, layout measured ~null).
   K-means, k ∈ {3..8} by silhouette, capped so total arms ≤ 7 including Random; merge clusters
   holding <5% of conditions. **Naming test** enforced: every cluster earns a human name from dominant
   failure mode + region, or k is too big.
5. **Per-arm samplers**: truncated Gaussian (mean/cov of member conditions, clipped to knob ranges)
   recorded in `arms.yaml` for reproducibility and the dormant backup channel. The retrieval channel
   needs only **membership**: demo → z(demo) (p_hat_0 and stage distribution evaluated from the fitted
   models on its recovered knobs) → nearest centroid. **Random arm**: uniform draw over all of W —
   always included; the control the winner must beat.
6. **Freeze** clusters, samplers, names → `arms.yaml` (hashed). Then compute the **per-arm well-count
   table** (a reportable deliverable) and apply the B rule (§0.2).
7. **`pool_demos.parquet`**: built from `fx_pool.json` + the pool's `meta/episodes.jsonl`,
   provenance-hashed. Day-one deliverable alongside the well-count table.

---

## 3. One pull (arm k, seed s_j)

```
demos = draw B distinct demos uniform from W ∩ region_k, farthest-point spread in the
        standardized recovered-knob space (h, w, x_rel, y_rel)
        (with-replacement across pulls of the same arm; all-distinct within a pull)
gates: quality (light — pool demos are pre-filtered successes: length bounds, parse sanity)
       novelty (not a near-duplicate of any D0 demo or same-pull keeper: same rule family as ε)
       ε-exclusion vs every eval start (§1.4)
dataset = build_lerobot_subset(D0_episodes + demo_ids)          # existing tooling
config  = clone of the shared TrainConfig, data_dirs only        # pattern of the 11 prior arms
ckpt    = train(config, seed=s_j)                                # 20k steps, ~8h
delta   = mean over 3 repeats of [eval(ckpt, E) − b]             # paired per-start; log per-stratum
ledger.log(pull)                                                 # §6, including failed pulls
```

Redraw demos on every pull of the same arm — the arm's value is the expected effect of its demo
distribution, and redrawing makes repeated pulls independent samples. Cost per pull ≈ 8h train + ~5h
eval (450 rollouts × ~40 s), two-wide on the 2 GPUs → ~2 pulls/day.

## 4. Scheduler — successive elimination (unchanged from spec)

Survivors start at all K arms; each round pulls every survivor once at the round's paired seed;
mean ± t-based CI per arm with **σ̂ floored by the measured σ_e**; eliminate arms whose UCB falls
below the leader's LCB; δ = 0.1; stop at one survivor, LCB-clears-all, or budget T.
**T = 12–16 pulls** (plus `pi_0`, 2 null pulls, and the exploit finetune → 16–20 total finetunes ≈
8–10 days wall two-wide). Ties inside the noise floor are ties; break toward the cheaper arm.

## 5. Exploit phase

Collect **2×B** demos from the winning arm with the same pull machinery (statistical tie → split
proportionally), one final finetune from base on `D0` + batch, eval on E ×3 for the headline number.
Deliverables: final checkpoint, its delta, and the arm-value table with CIs.

## 6. Ledger (all analyses read only from here)

`bandit_v1/ledger/`: `episodes.parquet` (episode_id, phase ∈ {diag, eval, null}, arm, pull_id, full x,
policy_id, outcome, failure_stage, cost), `pulls.parquet` (pull_id, arm, seed, demo_ids,
checkpoint_id, delta, per-stratum deltas, per-repeat deltas), `pool_demos.parquet`, `arms.yaml`,
`config.yaml` (recipe, knob ranges, E hash, channel, ε, B, all constants). Hash frozen files; seed
everything. All v1 code lives in `bandit_v1/`.

## 7. Analyses (from the ledger)

1. Arm ranking with CIs; **winner vs Random** is the causal headline.
2. Value curve: realized delta vs each pull's mean `p_hat_0` (equivalently p(1−p)) — the learnability read.
3. Per-stratum transfer: which eval stratum each arm moves (mechanism figure; easy stratum doubles as
   the within-task retention check — the saturation arm showed retention is the binding constraint).
4. Logged selector ablation: novelty/diversity/gradient-alignment scores of chosen demos vs realized
   delta (never deployed as the selector in v1).
5. Noise accounting: within-arm pull-to-pull spread vs σ_e; null-pull deltas plotted alongside arms.

## 8. Defaults

N=300 diag conditions, m=8, arms ≤ 7 incl. Random, B by rule (target 200), E=150 saved starts ×3
(baseline ×3, final ×3), T=12–16 pulls, δ=0.1, 2 null pulls, exploit 2×B, ε=0.10 m same-category,
diag seed base 600000, E seed base 500000, training seeds 1000+j.

## 9. Invariants (violating any corrupts the experiment)

Never re-cluster mid-run. Never touch E for demo selection; enforce ε for every demo. One channel for
all arms; if the well cannot fill an arm at the frozen B, the run halts (never per-arm B, never mixed
channels; exception: the exploit batch may be topped up post-identification, noted in the report).
Redraw demos every pull; never reuse a pull's demos elsewhere. Identical recipe every finetune; paired
seeds across arms. Keep Random to the end. Measure σ_e before interpreting any ranking; differences
inside it are ties. E is saved-state, restoration-validated; log failed pulls too.

## 10. Known risks and open items

- **reset_to end-to-end determinism is unvalidated** (the plumbing exists and is used for byte-exact
  demo restoration, but no one has round-tripped env-sampled states across processes). Day-one gate;
  fallback documented in §1.4.
- **Style is invisible to arms** (unrecoverable pool-side) despite being a large factor — scoped out,
  reported as a limitation.
- **Recovered pool x/y carry anchor-pipeline error** — ε and the novelty gate are deliberately
  conservative; the well draw uses recovered features only for membership and spread, not for exact
  matching.
- **p_hat_0 is intentionally coarse** (ceiling R²≈0.22); arms bundle conditions by failure mode, not
  by point predictions.
- **26.2% "base(400)" stratified number**: provenance unclear (basewarmup stopped at 6k steps);
  irrelevant to v1 — `pi_0` and b are measured fresh.
- **GPU contention**: eval ~40 s/rollout was measured alongside concurrent training; dedicated-GPU
  evals may run faster. Cost table is conservative.
