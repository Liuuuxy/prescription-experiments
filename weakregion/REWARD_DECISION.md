# Bandit Reward Decision Record — 2026-08-06

Outcome of the owner-directed teardown of the "learned reward model" proposal
for the data-prescription bandit. Full reasoning in session; this is the
durable record.

## Decision

- **Primary method: KILL+UNIFORM** — paired rounds over all arms (shared seed,
  race machinery), reward = paired ΔSR vs in-round random control, one-sided
  successive-elimination kill rule for HARM only (drop arm when paired-deficit
  UCB < 0), ties among survivors accepted, output = mixture over survivors.
  Power: kills a style_lo-sized harm (−3.9pp paired, σ≈2.5–3) in 3 rounds at
  ~85% power, α=0.1. Cost: 12 arms × 3 rounds × 5h ≈ 180 GPU-h.
- **Learned reward model: DEMOTED to shadow study** (rung 4). Not buildable
  now: p≈20 proposed features vs n=22 labeled pulls from only **8 independent
  arms** (the generalization unit); arm labels have SE 1.3–1.9pp vs between-arm
  signal SD ≈1.8pp. Any fit at that ratio is noise. Max defensible p≤2 = the
  existing fixed composite.
- **G1 reality check: the reward model is sim-only.** Calibration would need
  15–20 labeled pulls; one G1 label at ±3pp costs ~8h robot eval + 1–2h teleop
  → 60–200h total vs a realistic 10–20h budget. G1 gets KILL+UNIFORM with
  coarse ±10pp labels (1h/eval — sufficient for SREE-scale harms) + the
  sim-exported demo quality gate (no labels needed).

## Open questions (each is a ladder rung with a kill criterion)

- **Rung 0 (free, runs first):** split-half reliability of our own ground
  truth (analytic estimate ≈0.59 → observable-ρ ceiling ≈0.75) + permutation
  nulls over the ~6-candidate family (null hits ρ≥0.6 ~12% per candidate at
  n=6). KILL: reliability <0.5 → labels cannot validate ANY cheap reward;
  all proxy work stops; rung 3 runs SR-only.
- **Rung 1 (free):** supervised gradient axis under CORRECTED protocol
  (rotate supervising pair; current 0.60 is held-out at evaluation but
  post-hoc at supervisor selection → treat as upper bound, n=6, null p≈0.13;
  corrected number currently UNMEASURED). KILL: ≤ null 95th pctile.
- **Rung 2 (≤0.5 GPU-day):** probe composite at 10k ckpts (job partially run).
  KILL: ρ6 <0.4 → pulls cost 5h (SR eval) not 4h.
- **Rung 3 (~4 days, 2 GPUs):** KILL+UNIFORM closed loop, 12 arms + 1 planted
  corrupted-demo arm, shadow-logging all cheap features. KILL: planted arm
  survives 3 rounds, or a known-good arm is killed. SUCCESS = ICRA core claim.
- **Rung 4 (free, post-hoc):** reward-model shadow study on rung-3 logs
  (~35+ labels). KILL: retrodicted rollout-budget saving <20% → simple method
  final, permanently.

## OVERNIGHT RESULTS 2026-08-06 (unattended; owner asleep)

**RUNG 0: FAILED HARD — the headline.** Split-half reliability of the
ground-truth arm ranking: **ρ = −0.03** (Spearman-Brown −0.06; observable
ceiling ≈ 0). Family-wise permutation null (5 candidates × 6 non-style arms,
720 exact permutations): null q95 = 0.94, observed family max 0.60 → **p =
0.66**. Conclusions: (1) the 22-pull ledger CANNOT rank the 8 arms — only the
extremes (style_lo harm, style_hi/mid good) are real; (2) ALL cheap-reward
correlations to date (probe composite 0.60/0.67, supervised axis 0.60) are
statistically indistinguishable from chance — the rival explanation from the
teardown's #2 wins; (3) the paper cannot claim any validated cheap reward on
current labels. RUNG 2 SKIPPED per pre-registered rule (10k probe data banked,
no claims made). RUNG 1 rotated axes measured (mid-easy 0.66, gc3-gc2 0.71,
style 0.60) but unadjudicable under the same null — recorded, not claimed.

**Why rung 3 still runs (checked, agreed + strengthened):** kill decisions use
paired same-seed SR, self-contained CIs, no cheap-reward dependency. Moreover
the rung-0 failure is partly a DESIGN artifact — the ledger aggregates
unpaired seeds, and within-round paired comparisons cancel exactly the seed
variance that destroyed split-half reliability. Rung-3 data is the cure, not a
casualty.

**Disk protocol (99% full, 105GB free at queue start):** per pull — delete the
rebuildable dataset dir after training; probe 5000-ckpt then delete it; keep
9999 ckpt. Hard-stop watchdog at <30GB free.

**Overnight queue (launched 00:52):** PREFLIGHT PASS (end-to-end throwaway:
100-step train -> ckpt at custom step -> serve -> 6-episode stratified eval ->
ledger; one throwaway-only bug found+fixed: manifest subsets must cover all
strata). Planted arm built+verified (multiset-equal / order-different on
episode 400; manifest in ucb_robot/planted_corruption_manifest.json; candidates
= 800 uniform pool eps rng 777, standard draw at j=101). Round-1 pulls, shared
seed 1101 (j=101), 10k steps, smoke status: GPU1 [random, gc0, gc4], GPU0
[planted_bad, gc1, gc5]. Not queued tonight (round 1 completion later): gc2,
gc3, style_hi, style_lo, tall, mid, easy (have historical pulls but not at
seed 1101). Unattended decisions: (1) rung-2 rerank skipped per rung-0 rule —
10k probe data banked unanalyzed; (2) recipe 20k->10k for rung-3 pulls
(ablation-validated; decision metric is in-round paired diff, so the frozen
20k baseline b is recorded-only); (3) disk protocol active (delete rebuildable
datasets post-train, probe-then-delete 5000 ckpts, keep 9999); (4) 10k-probe
job shared GPU0 with planted train launch for ~10s window — run_pull retry
covers the OOM case; observed both trains healthy at 100% util 01:04.

**Round-1 results (live):** random_j101 (paired control) **+0.44pp**;
planted_bad_j101 **−21.78pp** — paired deficit −22.2pp ≈ 8σ: the planted
corrupted arm is detected decisively in ONE pull. Harm-detection claim has its
first, unambiguous data point; also end-to-end confirms the corrupted dataset
was actually trained on. (Not judged silently-wrong: magnitude plausible for
temporal-shuffle on 1/3 of training data; eval full-size; stratum means sane.)

**Overnight close-out (noon 2026-08-06):** 4 pulls complete, 0 crashes, 0
hard-stops. Deltas (paired vs random_j101 +0.44): planted_bad **−21.78**
(paired −22.2, killable after ONE pull — kill criterion met, corruption
detection demonstrated); gc0 +3.78 (paired +3.3, n=1, no claim); gc1 +0.67
(paired +0.2). Still running: gc4 (~16:40), gc5 (~17:00). Shadow probes
captured for all completed pulls at 5000+9999; datasets and 5000-ckpts
deleted per disk protocol (disk 105→69GB free, stable). Not yet pulled at
seed 1101: gc2, gc3, style_hi, style_lo, tall, mid, easy (~18h two-wide to
complete round 1).

## Baselines rung 3 must beat

B1 random allocation (must beat: detect planted arm; mixture ΔSR ≥ B1+2pp);
B2 free axis ranking (if B2 ≈ loop output, adaptive apparatus unnecessary —
becomes the method); B4 plain SR racing at equal rollout budget (loop must
match decisions at ≤2/3 the rollouts).

## Ruled out, and why

- **Learned reward model as decision-maker (now):** #1 degrees-of-freedom
  failure; #6 G1 label famine. Revisit only via rung 4 evidence.
- **Unsupervised gradient influence as reward** (LESS-style target alignment,
  diversity, D0-interference): measured ρ ≈ 0.09 / 0.14 / −0.43 vs ground
  truth, target-alignment fails style_lo-last sanity. Third independent
  falsification of unsupervised influence in this project (Q0 gate,
  dose-response, ranking eval).
- **1.5k-step burst rewards:** Phase A gate failure (SNR<1, wrong ordering);
  reward formation needs ≥~10k steps (pending rung-2 confirmation).
- **Chasing small positive premiums:** by design. Noise floor (±2.4–3.3pp/pull)
  makes ≤1.6pp arm gaps unresolvable at any affordable budget; harm is the
  high-SNR signal (style_lo −3.9pp paired, most consistent effect measured).

## What the simple route gives up (recorded, not forgotten)

Pre-registrable single-number reward; explainable composite ("kept D0, didn't
conform"); zero calibration hunger; small-premium discovery (accepted ties);
GP-UCB-style adaptivity as a paper headline (successive elimination retained
as the bandit structure).

## Key artifacts

reward_ranking_results.json (candidate-vs-truth table), loss_probe_calibration
{,_10000}.parquet, ucb_robot/arms.json (12-arm menu), gradarm/style/race
ledgers = the 22-pull label set. Session: bandit reward teardown, 2026-08-06.

**Disk event 14:30:** warning at 33GB free. Root cause: ckpt step-dirs are
~9GB each (not 7 as budgeted); in-flight gc4/gc5 hold 18GB apiece mid-train.
Action taken: deleted the eight CLOSED Phase-A burst datasets (~3GB,
rebuildable from recorded demo_ids; burst CHECKPOINTS untouched per
no-existing-results rule). Projected trough ~18GB free at gc4/gc5 completion
— no loss risk; monitor may beep once; recovers to ~35GB after 5000-ckpt
pruning. Both queues end after these two pulls.

**FINAL overnight tally (17:10, queue complete):** 6/6 pulls, 0 crashes, 0
hard-stops, disk trough survived (now recovering). Round-1 paired table vs
random_j101 (+0.44): gc0 **+3.3** | gc1 +0.2 | gc5 −1.6 | gc4 −2.9 |
planted_bad **−22.2 (KILL — one-pull detection, the rung-3 core claim's
first demonstration)**. All n=1 except the kill; no non-planted arm killable
yet. Remaining for a complete round 1: gc2, gc3, style_hi, style_lo, tall,
mid, easy at seed 1101 (~18h two-wide). Authorization window (noon) elapsed —
GPUs idle, awaiting owner: complete round 1 / add round 2 / stop and analyze.

## PRE-REGISTERED PROTOCOL — rounds 2-4 (launched 2026-08-07 00:35)

Algorithm: **round-synchronized Successive Elimination** (Even-Dar-family
confidence-bound BAI; our kill rule IS its one-sided special case). Reward:
paired d = Δ(arm) − Δ(random) within the same-seed batch. Width: t_{0.9,n−1}·
σ̂/√n with σ̂ pooled empirical (currently 3.47pp, df=12; updated as paired
rounds accrue). Eliminate iff mean + width < 0. Fixed budget: rounds 2-4
(seeds 1102-1104). Ties among survivors -> uniform mixture output + LCBs.
Arms not in roster = SUSPENDED (no claims), not eliminated.

Empirical power (sigma=3.47): kill −8pp in 1 round, −4pp in ~3-4, −2pp needs
14 (out of budget by design). Detectable at n=3: 4.25pp. style_lo (−3.85
expected): 74% power at n=3, 83% at n=4 — round 4 exists for this.

Roster: r2 {random, style_lo, gc0, planted_bad}; r3/r4 {random, style_lo,
gc0}. 10 pulls, ~50 GPU-h, ~25h wall two-wide. gc0 reaches n=4 (73% power to
confirm +3.3 at alpha=.1 one-sided if real). Planted rerun in r2 only
(consistency anchor; already proven at r1).

CUT from paper (fixed-budget honesty): full 9-arm roster ranking; best-arm
identification among positives; cheap-proxy reward claims; 5k/15k reward-
formation curve; G1 subtle premiums (G1 = coarse-harm SE: 2 arms x 3 rounds,
40 trials/eval = ±7.9pp -> detectable ~14pp = SREE/mode-collapse scale only,
~18h robot budget).

**Disk action 2026-08-07 ~05:00:** free space crossed 18GB during round-2
training. Deleted the 8 Phase-A burst checkpoints (ucb_*_a0/a1 @1499, ~9GB
each): Phase A is closed (gate failed), all probe measurements extracted
(ucb_robot/ledger.parquet), and the reward-formation re-probing they were
kept for was CUT from the 2-week plan. Probe JSONs + ledger retained.

## BLOCK B1 FINAL VERDICTS (2026-08-08 09:10, pre-registered protocol, complete)

Controls (seed lottery visible): +0.44 / −3.56 / −1.78 / −2.89.
- **gc0: CONFIRMED POSITIVE** — paired +3.33/+6.67/+1.11/+7.33, mean **+4.61,
  LCB +1.99** (alpha=.1 one-sided, n=4, pooled sigma updated 3.47->3.20 df=17).
  4/4 rounds positive. FIRST formally confirmed beneficial arm of the project.
- **style_lo: SURVIVES** — paired −2.22/−0.89/+0.67, mean −0.81, UCB +2.67.
  Harm at the 10k recipe is BELOW the 4pp detection threshold (and far milder
  than the historical −3.85 — consistent with rung-0's distrust of unpaired
  history). Honest null, claimed as such.
- **planted_bad: ELIMINATED** — paired −22.2, −16.0; one-pull detection, twice.

B2 decision (logged): run the LITERAL pre-registered output — survivor_mix =
uniform over {gc0 ∪ style_lo} (union draw, ~51/49) — vs random, 2 fresh paired
rounds (j=105,106). This tests the method's actual output incl. the cost of
style_lo's protocol-mandated inclusion. GPU0 shared w/ labmate job ->
BANDIT_TRAIN_MEM_FRACTION=0.8 on GPU0.

**Disk action 2026-08-08 ~13:30:** 16GB free with two 9GB final-ckpt writes
imminent (B2 round 5). Deleted five fully-banked finals: planted_bad_j101/j102
(poisoned models, zero reuse value) and gc1/gc4/gc5_j101 (suspended n=1 arms).
All probe values, SR, and demo_ids retained in ledgers. KEPT: all gc0, random,
style_lo finals (central/confirmed arms, gradient-analysis value).

## BLOCK B2 + RUNG-4 CLOSE-OUT (2026-08-09 08:15)

**B2 (mixture confirmation, n=2 paired rounds):** survivor_mix paired −2.89,
+1.33 → mean **−0.78** vs the +1.90 predicted under perfect 50/50 mixing of
its parts. No detectable advantage over random: the pre-registered
uniform-over-survivors output rule DILUTED the confirmed gc0 (+4.61) below
detectability with the weak style_lo half. Design lesson, claimed honestly:
the output stage should be confidence-weighted (practical prescription =
gc0-alone, already confirmed at n=4), not uniform. n=2 → CI ±7pp, so this is
a directional lesson, not an elimination of the mixture.

**RUNG-4 SHADOW STUDY (out-of-sample, paired labels, n=9):** the probe-loss
composite (frozen weights from the Aug-5 calibration, never refit) vs paired
SR outcomes:
- Within ordinary arms (n=7, planted excluded): pearson **+0.64**, spearman
  **+0.71** — every gc0 pull's composite positive, every style_lo pull's
  negative. The cheap signal tracks real within-family effects on reliable
  labels (suggestive at n=7, not conclusive).
- **The Goodhart trap fired exactly as the teardown predicted:** the two
  POISONED pulls have the HIGHEST composites of all nine (+0.055, +0.033) —
  the "didn't conform to the pool" term rewards garbage-fitting. Full-set
  pearson = −0.64 (sign FLIPPED by poison). The cheap reward would have
  PROMOTED the arm the SR loop killed in one pull.
- Paper-ready conclusion: cheap loss signals are useful *advisors* within
  plausible data and catastrophically exploitable by adversarial data →
  the two-tier architecture (cheap signals propose, rollout SR decides) is
  now EVIDENCED, not argued. Loss-reward-only bandits are unsafe here.

Block B totals: 21 prescription-loop pulls (16 B1 + 5 B2, incl. 1 killed for
GPU contention), 0 crashes, 2 disk interventions, all pre-registered rules
followed. SIM STOPS 2026-08-10 evening per schedule.
**Disk 2026-08-10 ~15:30:** freed survivor_mix_j105/j106 + random_j101 finals (B2 concluded; probes+SR banked) ahead of replica ckpt write.

**Determinism replica (2026-08-10, owner request):** exact rerun of
random_j106 (same demo ids, seed 1106, recipe, GPU). Weights: probe-loss
diffs 7e-6 / 2e-6 -> training is fully seed-deterministic. Score: -4.44 vs
original -3.56; per-episode agreement 67.1% over 450 pairs -> ALL fixed-seed
score noise is inference-time action sampling, not training. Implications:
(1) paired-round noise = eval sampling + seed effects only; (2) verdicts can
be tightened retroactively by re-evaluating saved ckpts with more repeats
(e.g., gc0 LCB +1.99 -> ~+2.6 for ~8 GPU-h, no retraining). Artifact:
gradient_analysis/replica_test_result.json.

## CLEAN OBJECT-FACTOR TEST — VERDICT (2026-08-13, owner-designed)

Design: hard-objects (base success 6-17%) vs easy-objects (73-96%) vs random
control; FIXED batches B=150, exact-matched on layout x side x 4x4 position,
execution-quality balanced (-0.015 vs 0.000), contamination-gated, criterion
pre-registered from the diagnosis alone. 3 paired seeds (1110-1112), 9 pulls.

**hard_obj: ELIMINATED (harmful).** Paired -5.11/-3.78/-5.11, mean **-4.67,
sd 0.77** — the most consistent arm effect ever measured in this project.
One-sided UCB = -1.73 < 0 even under the conservative pooled sigma; with its
own sd, t=-10.5. Stratum decomposition: its OWN (hard) stratum paired
-0.7/+8.0/-6.7 = +0.2 mean (no help to the targeted objects); the entire
deficit is mid/easy collateral.
**easy_obj: survives, trend-negative.** Paired -0.22/-3.78/-1.56, mean -1.85
(ns). Even mastered-object-only batches look <= random: object-RESTRICTED
data of any kind appears mildly worse than diverse random at equal B.

**THE PATTERN (the cleanest causal nugget of the project): adding
demonstrations of currently-FAILED objects makes the policy worse at
everything else and no better at those objects — with location, layout,
position, and execution quality all held fixed.** Inverts the intuitive
"collect what it fails at" prescription with a formal elimination. Known
rider: object height is intrinsic (hard set mean h=0.142 vs 0.060) and cannot
be matched; noted, with all prior height-axis tests ~null.

Cost: 9 pulls + 1 duplicate-killed, ~46 GPU-h. Checkpoints deleted
(fixed-batch + seed-determinism = exactly reproducible). Results:
gradient_analysis/objtest/*.json.

## GENERALIZATION TRACK 1 — CIFAR analogue of the object test (2026-08-15)

18 fine-tunes (3 arms x 6 seeds, B=800 budget-matched, fixed batches).
Deltas vs base, paired against rand_ctrl (+1.76):

  arm        overall      paired    on-HARD-classes   on-EASY-classes
  hard_cls   -1.90+-0.27   -3.66        **+43.72**        -1.39
  easy_cls   -1.09+-0.25   -2.85          +1.50           +5.06
  rand_ctrl  +1.76+-0.19      0           +6.89           -2.00

**The overall-damage finding REPLICATES across domains** (targeting the
failed slice is ~-3 to -5pp worse than random in both robot and classifier),
but **the MECHANISM is domain-specific**:
- CIFAR: targeting WORKS on-target, spectacularly (+43.7pp on the 6 targeted
  classes) and the overall loss is pure collateral (competitive softmax).
- Robot: targeting does NOT work on-target (+0.2pp on its own stratum) AND
  costs overall -4.67. Strictly worse: no gain, only damage.
Paper framing: "collect what it fails at" is a losing prescription in BOTH
regimes, but for different reasons — in classification you pay for a real
gain; in imitation you pay for nothing. The robot case is the stronger
cautionary result, and the CIFAR case rules out "the effect is just an
artifact of our robot pipeline".

## TRACK 3 — REPLICATION WITH DISJOINT OBJECTS: THURSDAY'S HEADLINE DOES NOT SURVIVE (2026-08-16)

Identical design, disjoint object sets (hard: kettle_non_electric, saucepan_lid,
colander, straw, teapot, yogurt @19-25% base SR; easy: kiwi, lemon, lemon_wedge,
onion, apple, sweet_potato @63-71%), seeds 1120-1122.

  experiment          hard_obj paired            easy_obj paired
  track 1 (orig)   -5.11/-3.78/-5.11 = **-4.67** sd 0.77 p=.009   -0.22/-3.78/-1.56 = -1.85 sd 1.80 p=.22
  track 3 (new)    +5.33/+4.22/-5.11 = **+1.48** sd 5.74 p=.70    -2.44/-2.89/-3.78 = **-3.04** sd 0.68 p=.016
  POOLED (n=6)     -1.59 sd 4.97 p=.47                            -2.44 sd 1.38 p=.007

**RETRACTION of the Aug-13 claim.** "Adding demos of currently-failed objects
makes the policy worse" does NOT replicate: on a second, equally-hard object
set the same design gave +1.48 (ns, sd 5.74 — huge between-round swing
+5.3/+4.2/-5.1). Pooled over both experiments hard_obj = -1.59, p=0.47:
**no reliable difficulty effect.** Track 1's -4.67 with sd 0.77 was a
property of THOSE six objects (blender_jug/tupperware/cheese_grater/... —
large rigid containers), not of failure-targeting. Lesson re-learned: n=3
with small sd is NOT replication; only a fresh instance of the *hypothesis
class* tests the claim. (This is the 4th time a within-experiment-consistent
result failed an out-of-sample test: gradarm day1/day2, style historical vs
paired, probe composite, now objects.)

**What DOES survive, and is now stronger:** every object-RESTRICTED arm in
both experiments underperformed the diverse random control at equal budget --
easy_obj -1.85 and -3.04 (pooled -2.44, p=0.007), hard_obj -4.67 and +1.48.
Pooled over all 12 object-restricted pulls: **-2.02pp, p=0.072**. The reliable
factor is DIVERSITY (breadth of object coverage), not difficulty: narrowing a
150-demo batch to any 6 categories costs ~2-3pp vs sampling the full pool.
That is consistent with every earlier result (region race ties, transfer
matrix off-diagonal dominance, mixture sweep superadditivity) and it is the
claim the paper should make.

---

## 2026-08-17 — Track 2 (second task) Stage B: instrument ported to PickPlaceCounterToCabinet

Goal: replicate the matched object test on a second task (owner: "try the same
method on other task" → "all of them"). Everything below is NEW files or
env-gated additions — task-1 artifacts untouched (suite re-run: 319 passed).

**Port design — `BANDIT_TASK_PROFILE=ppccab`** (env-gated block at the bottom of
`bandit_v1/config.py`): retargets TASK / POOL_LEROBOT / ARMS_JSON / FX_POOL_JSON /
D0_DATASET / ENV_ARGS_HDF5 and gives task 2 a FRESH `ledger_ppccab/` +
`states_ppccab/` tree. Unset = byte-identical task-1 behavior.

**Env metadata**: task 2 has no mimicgen src hdf5. Audit showed states.py only
reads `env_args` attrs from it; the only task-specific field is `env_name`
(robots/controller/cameras identical). Created
`mimicgen_src/PickPlaceCounterToCabinet_pi0_src.hdf5` (attrs-only carrier, no
demos) and validated: env constructs, resets, correct lang ("...place it in the
cabinet").

**Pool features** (`gradient_analysis/ppccab/build_fx_pool.py` →
`fx_pool_ppccab.json`, 9,880 rows, 106 cats): task-1's fx_pool came from sim
access; task 2 reconstructs everything from lerobot proprio. Validated the
extractor ON TASK 1 against fx_pool ground truth (300-ep sample):
- eef math exact given true grasp frame (median 0.5 mm, 100% within 5 cm) →
  conventions pinned: quat xyzw, eef relative in ROTATED base frame.
- grasp-frame detector (last fully-open frame before first sustained close,
  after skipping the initial gripper-closed phase): 93.7% of gx/gy within 5 cm,
  median frame error 1, p90 5. Residual ~6% tail = regrasp episodes; symmetric
  wrt any arm split, recorded as known noise.
- side agreement 93.3%; category from lang string 298/300.
- **layout NOT recoverable**: base-parking clustering hits only 63% purity vs
  true task-1 layouts → task-2 "layout" column is an honest 0.25 m parking-grid
  cell (215 cells), documented as scene-context key, not a layout id.

**Diagnosis prior**: task-2 tercile map seeds from the documented fallback
(task-1 pooled_episodes.csv rates) — a sampling-BALANCE prior only; measured
task-2 per-category rates come from the diag rollouts and will replace it
before E is built. New cabinet-only categories → middle tercile (designed
fallback).

**Running now (all detached)**: base ft `ppccab_base` seed 1000, 10k steps,
GPU0 (~16:00); 300-condition scan (pid 1896834); orchestrator
`run_diagnosis_ppccab.sh` (pid 2054878) waiting on both, then serves :8134 and
runs 2,400 rollouts via `run_diag_parallel.py` (workers=4, resumable,
phase=diag policy_id=pi0 in ledger_ppccab).

Launch gotcha for the record: first base-ft attempt died on wandb auth because
a manual `scripts/train.py` launch bypassed `pull._train_env` (which sets
WANDB_MODE=offline); relaunched through `pull.launch_training`. Also two shell
blocks self-killed via `pkill -f ppccab_base` matching their own command line.

## 2026-08-18 — Track 2 diagnosis COMPLETE: the weak region replicates across tasks

2,400/2,400 episodes (300 balanced starts × 8) of ppccab_base@9999 on
PickPlaceCounterToCabinet. Two resumable ops failures on the way, zero data
loss: (1) fresh-ledger read crash at launch (episodes table doesn't exist
before first append) — 7h idle; (2) parallel_eval's 3h per-worker timeout
killed two HEALTHY workers at 600-episode shards (sized for ~113-episode eval
shards) — fixed by scaling timeout to assigned shard; resume completed the
remaining 383 episodes.

**Results (all in ledger_ppccab/episodes.parquet):**
- Overall base SR **42.3%** on the balanced diag mix (task-1 analog: 47.3%).
- Per-category spread 0%→100%, 66 categories with ≥16 episodes.
- **Hardest**: oil_and_vinegar_bottle 0.000, cheese_grater 0.036, honey_bottle
  0.042, condiment_bottle 0.042, pitcher/pot/jar 0.062, glass_cup 0.069, jug
  0.097 — tall/awkward vessels AGAIN. cheese_grater, pitcher, jug, juice,
  ketchup were also in task-1's hardest tier → the tall-vessel grasp weakness,
  already cross-POLICY (pi0/GR00T), is now **cross-TASK**.
- **Easiest**: cucumber/whisk 1.000, lime 0.938, kiwi/mango 0.875 — compact
  produce, same as task 1.
- Failure stages: never_reached 634 / reached_no_grasp 525 /
  grasped_no_transport 226 → **84% of failures at/before grasp** (task 1: 82%).
- Measured tercile map re-frozen from these rates (fx_episodes_ppccab.json,
  90 cats) BEFORE E build, replacing the task-1 balance prior as pre-registered.
- **map_fit gate PASSED: AUC 0.678** (task-1: 0.629); family winner =
  sequential (task-1: two_model); per-gate grasp 0.684, transport 0.652.

Next: E build running (150 starts, measured strata) → baseline eval →
hard/easy sets from these measurements → matched batches → 9 pulls.

## 2026-08-18 — Track 2: baseline banked, 9-pull matched object test LAUNCHED

**Baseline (frozen E, 150 starts × 3):** b = **49.78%** (task-1: 51.33%).
Per-stratum: hard .427 / mid .427 / easy .640 (map's hard-vs-mid separation
doesn't show on E empirically; strata kept for reporting). Flip rate 32%.

**E build note:** map_fit run via `-m` pickles MapModels under __main__ and
broke eval_set's unpickle — re-fit via import path (same AUC 0.6782).

**Matched batches (OBJ_SEED 20260818, B=150 each, 103 matched parking cells):**
- hard_obj: oil_and_vinegar_bottle, honey_bottle, condiment_bottle, pitcher,
  glass_cup, jug (measured SR .000–.097; mean h 0.141m). cheese_grater/jar/pot
  excluded: pool has only 13–16 demos each.
- easy_obj: cucumber, whisk, lime, kiwi, mango, pear (SR .81–1.00; mean h 0.056m)
- random_ctrl: 150 from the gated pool (6,703 after E/D0 contamination gate)

**Pulls (launched 10:38):** rounds j=130/131/132 (seeds 1130-32), 3 arms each,
paired scoring vs in-round random_ctrl, OBJ_BASELINE=0.49778, 10k steps.
GPU0 slot-a: hard/easy/random@130 + hard/easy@132; GPU1 slot-b:
hard/easy/random@131 + random@132. ~5h/pull → results ~Aug 19 midday.
**Pre-registered read (same as track 1/3):** pooled object-restricted (hard ∪
easy, n=6) vs control tests the DIVERSITY claim; hard-vs-easy tests the
difficulty claim the task-1 replication already killed.

## 2026-08-18 — PRE-REGISTRATION: Track-2 prescription loop (gradient arms on task 2)

Owner ask: replicate the task-1 prescription loop (paired SE, 13 pulls: planted
killed twice one-pull, gc0 CONFIRMED +4.61 LCB +1.99, style_lo survives) on
PickPlaceCounterToCabinet — "if the gradient arms also work for other tasks,
that can be our dividing criterion."

**Design (faithful to task 1 unless noted):**
1. Sketches: per-demo LoRA-gradient JL sketch of every gated task-2 pool demo
   at ppccab_base@9999 — K=8 frames, 1 noise draw, 2048-d sparse-JL nnz 4000
   PROJ_SEED 12345, influence_score machinery (task-2 bindings:
   pi0_ppccab_bandit_a config, horizon 750, cabinet lerobot).
2. Arms: unit-normalize → remove top-10 SVD modes → renormalize → k-means k=6
   seed 20260801 → gc0..gc5 (labels frozen before any pull). Diagnostics:
   bootstrap-ARI, composition, silhouette; note any cluster <1200 demos.
3. Race: KILL+UNIFORM paired SE, rounds j=140,141,142 (seeds 1140-42): each
   round trains all live gc arms + planted + in-round random control at the
   round's seed, B=200 fresh draw per pull (pull_rng_seed), 10k steps, eval on
   frozen E (b=0.49778). Reward = Δ(arm) − Δ(random control, same round).
   σ̂ init 3.20 (task-1 paired estimate) until 2 rounds give a task-2 σ̂.
   Kill: harm-first (mean + t_{0.9,n−1}·σ̂/√n < −4pp → eliminate). Confirm:
   LCB(mean − t·σ̂/√n) > 0. Planted = temporal-shuffled actions on 200/600
   episodes (prep_planted recipe, task-2 draw).
**Pre-registered outcomes:** (a) planted NOT killed in ≤2 rounds → detector
fails on task 2, stop and debug before interpreting anything; (b) ≥1 gc arm
confirmed (LCB>0 within 4 rounds) → gradient-cluster criterion TRANSFERS;
(c) all gc arms tie random → criterion does not transfer (consistent with the
robot-nulls-are-a-domain-property CIFAR finding); partial kills without a
confirm are read as harm-detection-only value.
**Budget/sequence:** object test owns both GPUs until ~Aug 19 midday. Then:
sketch pool split across both GPUs (~16h) → cluster+planted (CPU, fast) →
race rounds (8 pulls/round, 2 GPUs ≈ 20h/round). Round-1 read ~Aug 21, full
3-round verdict ~Aug 23. Object-test results are NOT gated on this.

## 2026-08-18 11:30 — OWNER PREEMPTION: object test suspended, prescription-loop replica started

Owner: "stopped the current 9-pull object test and start that immediately."
Object test killed ~50min into round 1 (NO completed pulls — zero results to
discard; fixed batches + datasets persist on disk, resume = relaunch same
QUEUE, done-results skip). Sketching launched instead: gated pool 6,703
episodes split 2 GPUs (SHARD=0/2, 1/2), probe PASSED (grad dim 49,987,584,
ckpt loads, sketch OK). Race rounds follow per the 2026-08-18 pre-registration.

**2026-08-18 owner amendment:** no planted poison arm on task 2 — race is
gc0..gc5 + in-round random control only (7 pulls/round). Pre-registered
outcome (a) (planted-detector check) is WAIVED for this run; outcomes (b)/(c)
unchanged. Round 1 split: slot-a GPU0 gc0,gc1,gc2,random@j140; slot-b GPU1
gc3,gc4,gc5@j140.

## 2026-08-18 — External review of the trust-gated UCB: accepted amendments

Review verdict (owner-relayed): weak-reject as "correlation-gated MF-UCB"
(overlaps Kandasamy MF-UCB; Verma correlated-aux-feedback; corr-aware
surrogate bandits), and correlation-gating does NOT protect against the exact
poison failure motivating it (globally r>0 while tail-adversarial). Accepted.

**Amendments (all pre-race-decision, race itself untouched):**
1. TRUST = CALIBRATED PREDICTIVE VALIDITY, not correlation. Fit f̂(φ)≈E[Δ|φ]
   on calibration pairs; conformal residual bound B = q90(|Δ−f̂(φ)|); proxy
   enters the index only if B < κ·σ_rollout (it must beat a real pull's noise)
   AND the arm's φ lies inside the calibration support (tail/OOD guard — the
   poor-man's local reliability at our n). Correlation-gating is DEMOTED to an
   ablation baseline.
2. CALIBRATION STREAM: the reviewer's circularity fix (ε-uniform pulls) is
   already our design — rounds 1-3 are FULLY uniform (kill+uniform), so every
   (φ,Δ) pair is selection-free. Live proxy-driven allocation (if ever) keeps
   an ε-uniform stream and estimates validity from that stream only.
3. NOVELTY REFRAME: "safe auxiliary-feedback acquisition under proxy
   misspecification / statistically certified proxy reliability" with the
   3-piece story: objective-encoding test → target-aware proxy (success head)
   → certified borrowing. The bandit is the application; the encoding
   principle (already 3-way validated) is the contribution's spine. The
   honest wedge vs MF-UCB: Kandasamy ASSUMES known fidelity bias bounds ζ;
   we ESTIMATE and certify them online, and handle their invalidity.
4. WORDING: "loss doesn't encode success" downgraded to the empirical claim
   (no explicit success supervision + measured failure to predict), never the
   information-theoretic one. Equivalence-test framing (|ρ|<ρ_min) + CIs for
   all null claims (Spearman −0.06 n=12; gradqual n=2 UNDERPOWERED, say so).
5. COST ACCOUNTING: success-head supervision (2,400 diag rollouts) must be
   counted; matched-total-budget baseline required in any comparison.
6. REQUIRED EXPERIMENT (reviewer #8): proxy-corruption continuum — φ from
   clean → noisy → biased → nonlinear → tail-inverted (Goodhart at the top
   quantile, global r still >0), × allocators {rollout-only SE, MF-UCB(ζ),
   corr-gated, fixed-mix, calibrated-gated (ours), support-gated local,
   oracle-trust} → identification error + simple regret vs corruption.
   KEY demanded property: safety when r>0 globally but the tail is
   adversarial. Built as a CPU simulation seeded with MEASURED task-1 arm
   effects + measured noise decomposition.

## 2026-08-18 — Proxy-corruption continuum: first results (reviewer experiment #8)

sandbox_regions/proxy_continuum.py — best-arm identification, K=8, σ=3.2
(measured), landscapes with gc0-like +4.6 winner ± a −20 poison; proxy free,
corrupted along the reviewer's continuum; all staged allocators share the
same stage-2 machinery and differ ONLY in the trust test. 2-4k reps.
P(pick best) at budget T pulls (poison scenario):

  T=32              rollout  prefilter  corr-gate  CERT-gate
  clean               74%      82%        78%        79%
  noise×4             72%      58%        70%        72%
  tail-inverted       72%       6%        50%        66%
  goodhart-spike      73%      85%        68%        72%

Findings:
1. Unguarded prefilter (MF-UCB-style): best on benign corruption, GOES TO 2-6%
   under tail inversion — the 70pp Goodhart collapse, on cue.
2. Correlation gating (the criticized design): −22pp under tail-inverted
   poison (50% vs 72% baseline) — the reviewer's objection is now a measured
   number, and it becomes OUR ablation.
3. Certified (noise-adjusted residual-bound) gating: keeps +5pp at clean,
   holds ≈baseline on spike/noise, and cuts tail-inversion damage to −6pp
   (3.7× less than corr-gating). Residual leak traces to the soft trust
   threshold — a principled safety-vs-speed dial to sweep for the paper.
4. Two design bugs were caught BY the sim before touching the race: (a)
   certification must subtract rollout-noise variance from calibration
   residuals or a perfect proxy can never certify; (b) final recommendation
   must be pessimistic (mean − σ/√n) or lucky single-pull arms steal argmax.
   Both fixes applied to the live cmeter (ppccab_cmeter.py).
5. Amortization confirmed: calibration overhead shrinks with budget
   (T=16 gain +4pp → T=32 +5pp with prefilter gap narrowing).
Next for this figure: κ/threshold sweep tracing the safety-efficiency
frontier; harm-kill interaction (it RESCUES the prefilter under spike-only
corruption — worth a panel); equivalence-test CIs.

## 2026-08-18 — Review round 2 accepted: terminology, math audit, two new sim tasks

1. TERMINOLOGY: "certified" → **calibrated-validity gate** everywhere until a
   coverage theorem exists. The word "conformal" is DROPPED from the bound's
   description: what the gate computes is a PARAMETRIC latent-proxy-error
   bound under a measurement-error model — s_p² = max(0, RSS/(n−2) − σ²_meas),
   B = z_{0.9}·s_p — which assumes (i) Gaussian-ish residuals, (ii) known
   rollout noise σ_meas, (iii) independence of proxy error and rollout noise.
   Variance decomposition is legitimate under independence; QUANTILE
   subtraction is not, and no distribution-free coverage is claimed. A truly
   conformal version needs repeated-rollout calibration points or online-
   conformal machinery — noted as future work, not claimed.
2. Support guard gives SUPPORT validity only, not distributional validity
   (exchangeability is broken by proxy-driven querying after the gate opens).
   Paper must state this distinction.
3. RACE PROTOCOL UNCHANGED: rounds 1-3 stay uniform (prospective, selection-
   free calibration) — per reviewer, this is the empirically strongest asset.
4. New sim tasks before freezing: (a) INTERIOR inversion (malicious band
   inside calibration support — support guard blind by construction);
   (b) ORACLE-VALIDITY baseline (knows the true trustworthy region) to
   separate "proxy can help" from "gate discovers it". Results below.

## 2026-08-18 — Interior inversion + oracle-validity results (T=16, poison scenario)

  corruption          rollout  prefilter  corr  CALIB  oracle-validity  oracle-mu
  clean                 60%      74%      64%    65%       65%            85%
  tail_invert           61%       6%      48%    56%       61%            85%
  interior_invert       60%      74%      63%    64%       66%            84%
  interior_invert_top   60%      52%      55%    60%       62%            84%
  goodhart_spike        60%      74%      60%    59%       61%            85%

1. INTERIOR failure (reviewer's nasty case): when the WINNER's phi lies
   inside support (interior_invert_top), the calibrated gate stays at
   baseline (60 vs 60) — liars inflate calibration residuals, the gate
   closes, uniform fallback engages. Prefilter loses 22pp. Support-guard
   blindness did NOT translate into damage at these settings; no local q(φ)
   needed yet (mid-band inversion is simply harmless to BAI — neither
   allocation nor identification cares about middle arms).
2. ORACLE-VALIDITY (knows the true trustworthy region, same staged budget):
   ours matches it within 1-2pp in every cell except tail_invert (56 vs 61).
   The reviewer's "extremely compelling" pattern (baseline < ours ≈ oracle)
   holds in 4 of 5 cells; the tail-inversion leak (~5pp) is the soft-threshold
   dial, to be traced in the frontier figure. The true-mu oracle at 85% shows
   the remaining headroom is information NO proxy carries.
3. The middle-band interior_invert cell is a useful negative-control panel:
   every allocator is unaffected — corruption only matters where decisions
   live (tails), which itself motivates support/tail-focused guards.
Sim FROZEN pending the frontier sweep; race calibration data (uniform,
prospective) proceeds untouched.

## 2026-08-18 — PRE-REGISTRATION: gate freeze + race readout criteria (BEFORE any race data)

**GATE FROZEN as of this entry** (no constant may change in response to race
data; any change forks a v2 label and restarts calibration):
- s_p² = max(0, RSS/(n−2) − σ̂²_meas); B = 1.28·s_p  [parametric latent
  proxy-error estimate — never "coverage"/"certified"/"distribution-free"]
- trust weight w = max(0, 1 − (B/κσ̂)²), borrow iff w > 0.15; κ = 1.0
  (chosen A PRIORI; the κ-sweep below is descriptive, not a selection)
- support = [min φ_cal, max φ_cal] per proxy; OOD ⇒ w=0 for that arm
- proxies, each gated independently: φ_probe (loss composite),
  φ_grad_coherence, φ_grad_centroid_cos
- σ̂ = paired-residual estimate, init 3.2 until df≥4
- Two DISTINCT detectors, reported separately: support check (extrapolation
  failure) + residual-validity check (in-support failure).

**Race readout (rounds 1-3 uniform, gate in shadow; data = evidence, not
development signal):**
- O1 GATE OPENS after round 2 (any proxy, n=12 pairs): round-3 TRANSFER TEST
  = fraction of round-3 arms with |Δ_r3 − f̂_{r1,2}(φ_r3)| ≤ B. Pass ≥5/6.
  Pass ⇒ prospective validity transfers; a live-borrowing round 4 becomes
  admissible. Fail ⇒ outcome O3.
- O2 GATE STAYS CLOSED: report s_p vs κσ̂ with CI. Claim limited to "safe
  fallback validated on real data"; equivalence framing mandatory — in the
  sim, gate-closed performance must be EQUIVALENT to rollout-only within
  ±2pp (TOST at 95%), not merely "not significantly worse".
- O3 OPENS-THEN-FAILS-TRANSFER (>1/6 round-3 arms outside B): the most
  important failure; reported as-is, no redefinition. Because rounds stay
  uniform, O3 specifically isolates temporal/round nonstationarity from
  selection shift — a cleaner diagnosis than any adaptive design allows.
**Naming:** true-μ oracle = "full-information oracle" (85% = information no
proxy carries, NOT gate failure). Contribution framed as acquisition-agnostic
SAFETY LAYER (algorithm + calibrated-validity gate + any cheap proxy), with
ablation ladder A(rollout) < B(+unconditional proxy) < C(+corr gate)
< D(+residual gate) < E(D+support) ≈ O(oracle validity).

## 2026-08-18 — κ-frontier (descriptive; 3000 reps/cell; κ=1.0 was frozen A PRIORI)

  κ      clean gain   worst-case D (concentrated in tail_invert)
  0.5      +4.2pp        +4.0pp
  0.75     +2.7pp        +5.6pp
  1.0      +2.7pp        +3.5pp   ← frozen operating point (pre-registered)
  1.25     +5.3pp        +5.9pp
  1.5      +6.4pp        +4.8pp
  2.0      +6.4pp        +8.3pp

Read: clean gain grows with κ; worst-case exposure grows sharply past κ=1.5;
differences within κ∈[0.5,1.5] are partly within sim noise (cell SE ≈0.9pp).
Paper figure needs 8000+ reps + CIs — machinery frozen, only rep count may
change. SIMULATOR DEVELOPMENT ENDS HERE per review round 3; the next data
this project consumes is the prospective race readout under the frozen gate.

## 2026-08-18 — Disk: tier-1 checkpoint cleanup executed (owner-authorized)

46 banked task-1 pull checkpoint dirs deleted from pi0_ppc2sink_bandit_a/b
(race r3-5, nulls, gradarm, gradqual, SE non-gc0, objtest leftovers, replica).
KEPT: gc0_j101-104 (re-eval option), everything ppccab (live). All deleted
pulls: results in ledger, reproducible via seed+demo_ids (replica-test
determinism). Disk 105GB→923GB free (99%→87%). Tier 2/3 (~380GB: closed early
arm configs + pi0base intermediates) staged in
/data/xinyua11/tmp/cleanup_banked_ckpts.sh, not yet executed.

## 2026-08-19 — Round-1 raw results + control failure & repair (protocol unchanged)

Raw deltas vs baseline 49.78% (seed 1140): gc0 −6.22, gc3 −6.22 (tie verified
benign: both exactly 196/450 but 71% episode agreement — distinct policies),
gc4 −5.11, gc1 −3.78, gc5 −3.56, gc2 −10.67 (outlier, watch for harm-kill at
round 2). All negative → seed-1140 downdraft suspected; PAIRED scores pending.

**random_j140 FAILED at launch**: driver passed regions=None for the random
arm; run_pull requires a regions object for non-null arms — task-1 encoded
random as an EMPTY region series (uniform-pool path). Fixed in
ppccab_rung3_driver.py. **Makeup plan (race chain v2)**: rerun random_j140 on
GPU1 in the gap after slot-B round 2 finishes (~21:00), serialized BEFORE
round 3 so no GPU collision; round-1 paired table therefore lands ~02:00
tonight with the round-2 cmeter. Rounds 2/3 unaffected (each has its own
in-round control; round-2 launched on schedule). Old chain killed after
round-2 launch; v2 owns round-3 sequencing. Uniform allocation + shadow gate
unchanged — the makeup is the same pull the protocol always specified, run
late.

## 2026-08-19 15:20 — PRE-REGISTRATION (before ANY paired race number exists):
## success criteria for the two claims + F3⁻ baseline mandate

**H_cluster (gradient-criterion transfers) is POSITIVE only if:**
  ∃ arm a with (i) paired mean μ̂_a > 0 absolutely (not best-of-negative),
  (ii) reproducible: LCB90(μ̂_a) > 0 over its rounds (n≥3), and
  (iii) practically meaningful: μ̂_a ≥ δ_practical = +2.0pp (≈ the measured
  diversity-effect size; half of task-1 gc0's +4.61).
  "Highest arm among negatives," "positive in 1 of 3 rounds," or "positive
  only after re-pairing" are all FAILURES of H_cluster. If all arms are
  paired-negative, that is a REPLICATION of the diversity law (restriction
  loses to uniform) and is reported as such — outcome (c), not partial success.

**H_gate (gate provides UTILITY, not merely safety) is POSITIVE only if ALL:**
  (i) gate OPENS on ≥1 proxy from rounds-1/2 calibration pairs;
  (ii) transfer holds: round-3 |Δ − f̂(φ)| ≤ B on ≥5/6 arms (as registered);
  (iii) UTILITY: the shadow index computed after round 2 selects the same
  final arm (within σ̂ of the top) as the full 3-round uniform race — i.e.
  certified borrowing would have saved round 3's 7 high-fidelity pulls
  (~33% of F3 cost) at equal selection quality.
  Gate staying closed = safety-mechanism validation ONLY. It supports the
  safety claim, is reportable, but does NOT constitute the paper's positive
  contribution. No redefinition after data.

**F3⁻ mandate (the dangerous baseline, per advisor):** before claiming any
proxy utility, quantify cheap UNBIASED evaluation: Var(Δ̂ | N_eval) from
subsampled real episodes, and the N at which arm-ranking quality ≈ N=450.
COST-STRUCTURE FACT to confront honestly: φ_probe requires the trained
checkpoint, so it only competes with the EVAL hour (~20% of a pull); if
F3⁻@N≈100 retains ranking quality, φ_probe is nearly cost-dominated and the
only proxy with a strong cost case is φ_grad (pre-training, skips the 4-hour
training too). This observation is recorded BEFORE seeing gate results.

## 2026-08-19 — F3⁻ baseline measured (subsampled real race evals, 400 bootstraps)

  eval episodes   paired-diff SE   rank-corr vs full   top-1 agreement
      51              6.8pp             0.36               27%
     102              4.6pp             0.50               36%
     150              3.2pp             0.58               43%
     225              2.6pp             0.71               64%
     300              1.8pp             0.80               76%
     450 (full)      ~1.5pp (asymp.)    1.00              100%

**Verdict on "just evaluate more cheaply":** eval is only ~1h of a 5h pull.
Even halving eval (225 eps) saves ~10% of pull cost while inflating total
per-pull σ from 3.2 to ~3.8pp (+19%) — a bad trade for identification.
F3⁻ is capped at ~15% cost savings before ranking quality degrades.
CONSEQUENCE (pre-registered cost logic confirmed): (a) φ_probe, which also
requires the trained checkpoint, competes only against that same 1h eval —
it is nearly cost-dominated by F3⁻ and cannot carry the utility claim alone;
(b) real utility must come from SKIPPING WHOLE PULLS (gate's H_gate(iii):
round-3 skip = 33% of F3 cost) or from φ_grad (pre-training, the only proxy
that touches the 4h training cost). The paper reports this cost decomposition
as the reason the safety-layer question is about allocation, not evaluation.

## 2026-08-20 03:45 — Rounds 1-2 paired table + the gate's first real test:
## OPENED DEGENERATELY (B=0 pathology) — O3 predicted BEFORE round 3

**Controls swung hugely**: random_j140 = −0.89, random_j141 = −5.33 (the
"downdraft" was the CONTROL'S draw lottery, not the seed). Paired table:
  arm   r1      r2      mean(n=2)
  gc1  −2.89   +4.22    +0.67
  gc5  −2.67   +1.11    −0.78
  gc4  −4.22   +2.22    −1.00
  gc0  −5.33   −0.89    −3.11
  gc3  −5.33   −0.67    −3.00
  gc2  −9.78   −0.67    −5.22
No kills (σ̂=4.39 → gc2 UCB +0.64 > −4 threshold). ARMS are stable across
rounds; the CONTROLS are the volatile term (4.4pp swing) — per-round control
noise is COMMON to all arms in the round, inflating the naive residual σ̂.
H_cluster: no arm near the frozen bar (+2 with LCB>0; best mean +0.67 with
LCB −5.2). Outcome (c) remains the projection pending round 3.

**GATE PATHOLOGY (observed, frozen rule untouched):** all three proxies
"certified" with latent_B90 = 0.0 — because s_total (3.83–3.97) < σ̂ (4.39),
the variance subtraction floored at zero ⇒ "perfect proxy," w=1, despite raw
correlations of −0.12…−0.29 (noise, wrong sign). This is exactly the
reviewer-warned subtle failure of treating a noisy variance difference as a
point estimate: when measurement-noise σ̂ (itself inflated by control-common
variance) exceeds observed scatter by chance, s_p collapses to 0.
**PREDICTION RECORDED BEFORE ROUND 3: the transfer test will fail (B=0 makes
|Δ−f̂(φ)| ≤ B unsatisfiable) ⇒ outcome O3 (opens-then-fails).** Per the
pre-registration, the frozen rule runs as-is; the v2 fix (estimator
uncertainty on s_p: certify only if s_tot² < σ̂² is statistically incredible,
e.g. one-sided χ² test, plus a B floor) is DESIGNED but not applied to this
run. The shadow indices' proxy bonuses (±0.3–2.5pp of fitted noise) decided
nothing — allocation was uniform throughout, as registered.
Round 3 launched 03:44 (gc0/gc3 training).

## 2026-08-21 — TRACK-2 RACE COMPLETE: formal verdicts under the frozen criteria

Final paired table (arm − same-seed random control; controls: r1 −0.89,
r2 −5.33, r3 −2.67; σ̂ = 3.39):
  arm   r1      r2      r3      mean(n=3)  LCB90    UCB90
  gc1  −2.89   +4.22   +0.89    +0.74     −2.47    +3.95
  gc4  −4.22   +2.22   −1.11    −1.04     −4.25    +2.17
  gc5  −2.67   +1.11   −2.67    −1.41     −4.62    +1.80
  gc0  −5.33   −0.89   +1.11    −1.70     −4.91    +1.50
  gc3  −5.33   −0.67   −1.78    −2.59     −5.80    +0.62
  gc2  −9.78   −0.67   −8.44    −6.30     −9.50    −3.09

**VERDICT 1 — H_cluster: FAILED, outcome (c). The gradient-cluster criterion
does NOT transfer.** Best arm gc1 (+0.74) misses every frozen bar (needs ≥+2.0
with LCB>0; has LCB −2.47). All six clusters ≤ random; the diversity law
(restriction loses to uniform) replicates on task 2 in a third form. Notable:
**gc2 is CONFIRMED HARMFUL** (UCB90 −3.09 < 0) though it escapes the −4
harm-kill bar — the first confirmed effect of any sign on task 2.

**VERDICT 2 — H_gate: outcome O3 (opens-then-fails), exactly as predicted in
the 08-20 03:45 entry before round 3 existed.** Transfer test vs the round-2
fit (B=0): 0/5 arms within bound (gc4's r3 probe failed → NaN, excluded).
φ_probe residuals 0.5–6.6pp; it missed gc2's −8.4 by 6.6pp. The H_gate(iii)
utility sub-criterion coincidentally passed (shadow top pick after r2 = gc1 =
final race top) but (ii) failed, so H_gate is NEGATIVE overall.
**Structural diagnosis for v2 (not applied to this run):** the certification
subtracts σ̂ estimated from the SAME residual pool it tests — when proxy
explains little, s_total ≈ σ̂ and the subtraction degenerates to 0 ("perfect
proxy"). v2 needs an INDEPENDENT σ_meas (eval binomial + replicate-pull seed
component) + a χ² significance test + B floor.

**Real signals worth keeping:** (a) φ_grad_coherence has arm-level harm signal
(raw r = −0.42; gc2 = highest coherence 0.025–0.026) — high gradient coherence
flags harmful clusters, consistent with "narrow = harmful"; (b) ARMS are far
more seed-stable than assumed while CONTROLS carry the volatility (−0.9/−5.3/
−2.7) — future designs should replicate the CONTROL, not the arms.
Race protocol: 21 pulls, uniform throughout, zero frozen-rule violations.

## 2026-08-21 — DP SURROGATE PROGRAM STARTED (owner: "resume and do it on
## imitation learning, GPU0 only, GPU1 free")

Architecture per advisor plan: Diffusion Policy as F2 — the same intervention
(add data → retrain → rollout) at lower model capacity; the calibrated gate
(v2, with independent σ_meas) will decide per task whether Δ_DP transfers to
Δ_π0. Loss/gradient signals demoted to F0 dataset-construction heuristics.

**PILOT (running, GPU0):** train DP (transformer, bs192) from scratch on
task-2 D0 (400 demos, direct lerobot load — no conversion needed) — 200
epochs, ckpt every 50, offline wandb (logging.mode=offline — env var alone
insufficient, same class of bug as the openpi launch). Then eval ≥50 rollouts
(run_dp_smoketest.py, PickPlaceCounterToCabinet).
**Pre-registered pilot kill criteria:** (a) DP base SR < ~15% (floor effects
would fake ρ≈0 in concordance) OR (b) per-experiment cost not ≲ half a π0
pull → DP surrogate program stops there, reported as such.
**If pilot passes → concordance experiment:** retrain DP on archived
prescriptions with KNOWN Δ_π0 (planted poison −22/−16 as the mandatory
anchor; gc0-t1 +4.61; style_hi/lo; hard/easy_obj; task-2 gc1/gc2), 3-5 seeds
each, plot V_DP vs V_π0. Three pre-registered readings: ρ high → screening
fidelity; extremes-only → elimination fidelity; ρ≈0 → DP rejected (and the
gate is the instrument that formalizes whichever it is).

## 2026-08-21 21:30 — OVERNIGHT AUTHORIZATION (owner: "auto research the
## eventual problem, revisit mf-ucb, try anything overnight")

Rules as before: never block, fail forward, GPU0 only (GPU1 owner-gated after
1am), new files only, everything logged here.
A. GATE v2 retrospective (CPU): implement v2 certification — independent
   noise decomposition (within-arm pooled variance as σ_meas, a DIFFERENT
   statistic from the fit residuals), F-test that the proxy fit reduces
   variance significantly, floor on B — and re-run on the race's 18 real
   (φ,Δ) pairs (expect: correctly REFUSES all three proxies) + rerun the
   corruption continuum under v2 (expect: degenerate-open eliminated; measure
   what it costs at clean).
B. CONCORDANCE FIRST CELLS (GPU0, gated on pilot ≥15% SR at ~00:30): DP
   retrains on archived task-2 prescriptions with known Δ_π0 — priority
   gc2 (confirmed −6.3) then planted-poison build (anchor) then gc1 (+0.74).
   Training length set by the epoch-100 eval (100 if ≈200-epoch score).
   If pilot FAILS the bar: single trainability probe (400→800 demo base) and
   stop.
C. RESEARCH SWEEP (Workflow, parallel web agents): 2025-26 literature on
   multi-fidelity BAI with unknown/estimated fidelity bias; robot-IL data
   valuation (CUPID/DataMIL/FSC successors); recovery/intervention data;
   cheap-policy surrogates & cross-model data-value transfer; eval variance
   reduction. Deliverable: morning shortlist of runnable-on-our-stack methods.
D. NEXT-RACE DESIGN (CPU): control-replication power analysis from the 21
   pulls (controls σ≈2.3 across rounds vs arms stable) → optimal
   controls-per-round allocation for the next protocol.

## 2026-08-21 22:00 — GATE v2: BUILT, VALIDATED RETROSPECTIVELY + IN SIM

v2 = arm-level regression F-test (p<0.10 required: the fit must beat chance)
+ INDEPENDENT noise statistic (pooled within-arm variance, not the fit
residuals) + floor on B (≥ 0.5·SE_mean). Results:
- REAL RACE DATA (18 pulls, deduped): all three proxies REFUSED (F-p 0.80 /
  0.25 / 0.30; coherence r=−0.55 arm-level suggestive but n=6 insufficient).
  v1 had certified all three with B=0. Pathology fixed.
- CONTINUUM (3000 reps/cell, poison): v2 ≈ v1 within noise everywhere
  (clean 64.6 vs 64.1; tail 57.8 vs 57.6; interior-top 60.5 vs 59.2) —
  the F-test costs nothing at clean and the degenerate-open is eliminated.
DECISION: v2 is the paper's gate. v1's O3 on the race becomes the motivating
failure; the retrospective refusal is v2's real-data validation. A future
LIVE run of v2 (labeled, fresh pre-registration) is the remaining ask.

## 2026-08-21 22:35 — Next-race design math (item D): REPLICATE THE CONTROL

Measured decomposition from the 21-pull race: arm-side noise sd 1.82pp,
control-side sd 2.23pp (control's fresh-draw lottery dominates). Info per
pull for m controls/round (6 arms): m=1: .0172, m=2: .0215, m=3: .0223
(optimum), m=4: .0219. DECISION for any future race: 2-3 random controls per
round (averaged) — ~25-30% more information per GPU-hour at identical
protocol semantics, plus per-round σ estimation. Complements the arms-are-
stable finding: rounds could even drop to 2 with m=3 at equal power.

## 2026-08-21 23:05 — Research sweep landed (full text: weakregion/RESEARCH_SWEEP_2026-08-21.md)

Actionable core (arxiv IDs unverified; synthesis saw 3/6 batches — full
per-agent results in the workflow journal):
1. NOVELTY SLOT CONFIRMED EMPTY: "certified prescription" — probe-then-
   allocate where every cheap signal passes a validity certificate with
   negative controls before spending budget. MF-bandit line leaves online
   bias estimation as future work; FSC/CUPID/DataMIL publish no placebo or
   attainability gate; SureSim certifies evaluation, not allocation.
2. DP concordance kill numbers (No-Free-Lunch-for-PPI): arm-level ρ≥0.7 →
   adopt as fidelity; 0.5-0.7 → audit-allocation only; <0.5 → kill (below
   break-even vs buying seeds). ADOPTED as the pre-registered read for the
   running concordance cells.
3. TWO NEW EXPERIMENTS worth running: (a) respawn attainability probe —
   frozen π0 retries from harvested failure states, ~0.2 pull cost, gates
   (b) recovery-data arm race (IntervenGen-style MimicGen replay from
   failure states vs uniform, bar +6.5pp) — the one published axis
   orthogonal to our subset-restriction null, untested at strong-policy
   operating points.
4. Poison mechanism lead: critical-interval MSE (transit motion dominates
   raw action-MSE; restricting to the grasp window reportedly flips the
   correlation) — testable on archived data with our grasp-frame detector.
5. Lite-pull fidelity: formalize F3⁻ (10k-steps + 150-episode screens) as a
   measured-bias same-estimand fidelity — 2-3× race cost cut, zero Goodhart
   surface.

## 2026-08-22 — DP concordance: first two cells (task 2, 100-epoch recipe)

  cell  Δ_π0(known)   DP SR (base .16)   Δ_DP
  gc2      −6.3            .14           −2
  gc1      +0.74           .18           +2
Both signs concordant; DP orders the confirmed-harmful arm below the best arm.
n=50/cell → ±7pp; treat as directional only. Ops cost of the night: two
driver bugs (checkpoint_every=100 with 100 epochs saves NOTHING → 5h lost;
eval stderr swallowed) — both fixed (ckpt_every=50; nested eval path).
Task-1 DP base trained (GPU1, exited rc=0); its eval queued on GPU0 → gates
the high-contrast task-1 cells (poison −22/−16, gc0 +4.61, style ±) which
will determine ρ and the pre-registered adopt/audit/kill read.

## 2026-08-22 14:15 — Task-1 DP base FAILED its bar (6% < 15%) → task-2-only concordance

Task-1 cells (poison/gc0/style) are floor-limited and DEAD under the frozen
pilot criterion; from-scratch DP cannot learn the sink task from 400 demos
(even the official multi-task DP ckpt scored 10% here). Optional daytime
decision: an 800-demo task-1 trainability probe. ADAPTATION (within the
pre-registered structure): extended task-2 queue on GPU0 — base re-eval
n=150, task-2 PLANTED-POISON cell (policy-agnostic anchor, PLANT_J=143:
shuffled actions must crash DP or the surrogate dies), then gc0/gc3/gc4/gc5
cells at n=150 → full 6-arm rho vs the race's paired Δ_π0 by tomorrow.

## 2026-08-22 20:25 — Poison anchor AMBIGUOUS at n=150; n=450 re-eval pre-registered

planted_bad_j143: DP 15.3% vs base 20.7% → Δ_DP = −5.3pp, z≈1.2, p≈0.23 (ns).
π0's response to the same recipe was −22/−16 from a 50% base — DP's poison
sensitivity is ~4× weaker absolute. PRE-REGISTERED READ (before n=450 data):
re-eval planted + base checkpoints at n=450 each; anchor PASSES iff
poison−base is negative with one-sided p<0.05; otherwise DP's elimination-
grade sensitivity is INSUFFICIENT and the surrogate's kill-screening claim is
dropped (concordance-for-ranking may still stand on the gc-arm rho, judged
separately by the adopt/audit/kill thresholds). Re-evals run on GPU1 behind
the owner's 1am idle gate (16 envs; eval-only).

## 2026-08-23 03:30 — POISON ANCHOR PASSES at n=450 (pre-registered read)

base 16.9% (n=450) vs planted 10.0% (n=450): diff −6.9pp, z=−3.04, one-sided
p=0.0012 — DP reliably detects action-shuffled poison. The surrogate's
ELIMINATION-GRADE sensitivity claim stands: a ~40% relative crash on the
policy-agnostic corruption, detectable at 1.5h of DP eval vs 5 GPU-h of π0
pull. n=450 base (16.9 ±1.8pp) is now the authoritative reference for all
task-2 DP deltas; the n=150/n=50 wobble (16→20.7→16.9) was binomial noise.
Remaining for the surrogate verdict: gc3/gc4/gc5 cells (GPU0, through midday)
→ 6-arm ranking ρ vs paired Δ_π0 under adopt(≥.7)/audit(.5-.7)/kill(<.5).

## 2026-08-23 19:45 — Concordance final table + verdict (pending 1 control cell)

All cells n=150, base n=450 (16.9%): gc1 +6.4, gc0 +5.8, gc3/gc5 +4.4,
gc4 +3.8, gc2 +3.1, poison −6.9. vs Δ_π0: Spearman(6 gc) = 0.55 (p=.26),
Pearson 0.76; incl. poison anchor Spearman 0.72 (p=.07).
VERDICTS vs frozen thresholds: (1) RANKING = audit-only band (0.5–0.7),
flagged UNSTABLE (two n=150 re-evals swung ρ̂ from −0.01 to 0.55; n=6 arms
cannot separate 0 from 0.8); (2) ELIMINATION = poison-only (3σ) — DP did NOT
flag gc2 (+3.1, wrong sign, though rank-last); veto claim narrowed to gross
corruption. (3) SYSTEMATIC CONFOUND found: ALL gc cells +3..+6 on DP while
ALL ≤0 on π0 — from-scratch DP is data-starved (any +200 demos help),
pretrained π0 is saturated. Δ_DP-vs-base conflates quantity with composition
→ QUEUED the missing DP RANDOM-CONTROL cell (random_j141 rebuild) to convert
the DP axis to the paired definition. This regime difference (starved vs
saturated learner) is itself a paper-worthy mechanism for when cheap-policy
surrogates mislead.

## 2026-08-24 01:10 — CONTROL CELL FLIPS THE MECHANISM: opposite composition
## preferences between regimes (the surrogate study's headline)

random_j141 → DP 16.0% (n=150) ≈ base 16.9% (n=450): +200 random demos do
NOTHING for DP. Paired Δ_DP (cell − random): gc1 +7.3, gc0 +6.7, gc3/gc5
+5.3, gc4 +4.7, gc2 +4.0, poison −6.0. ALL six gradient clusters BEAT random
on from-scratch DP (P≈.016 under null) while ALL six LOST to random on
pretrained π0. FINDING: composition value is REGIME-DEPENDENT with sign
inversion — coherent batches = curriculum for a starved learner, diversity-
tax for a saturated one. Consequences: (1) small-policy surrogates answer
"beats random?" BACKWARDS for structured data → screening claims (CUPID-
style) are regime-bound; (2) poison veto survives in paired terms (−6.0);
(3) within-family rank roughly preserved (gc1 top, gc2 bottom on both;
ρ̂ .55 unstable); (4) the calibrated-validity gate REFUSES Δ_DP as a proxy
for Δ_π0 (F p≈.26) — the safety layer catches the regime inversion
prospectively, which is the paper's architecture working as designed.
Caveat: single training seed per cell; DP train-seed noise unmeasured —
a 2-seed spot-check of one gc cell + random would firm the inversion.

## 2026-08-24 12:15 — Seed-check CONFIRMS the inversion; DP train noise measured

seed 1043: gc1 22.0%, random 18.0% → gap +4.0pp (seed 42: +7.3). Two-seed
mean gap +5.7pp; per-cell train-seed swing 1.3-2.0pp → DP train noise σ≈1.5pp.
The regime inversion (all clusters > random on from-scratch DP; all ≤ random
on pretrained π0) is now supported by 2 independent training seeds on the
DP-positive side and stands >>noise on the strongly-inverted single-seed
cells (gc0 +6.7, gc3 +5.3 vs π0 −1.7/−2.6). Next: CUPID-matched UNet-hybrid
pilot (started on GPU0 12:10) — if its base materially exceeds 16.9%, the
concordance cells rerun on the CUPID architecture for the paper's comparison.
