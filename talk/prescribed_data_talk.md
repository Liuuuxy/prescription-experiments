# The Value of Prescribed Data
### Insights from my experiments on data-efficient robot imitation learning

**Companion document to the 20-minute talk.** Contains the full narrative, every
experiment and result, the complete speaker script, the statistics, and reproduction
pointers. The slide deck is `prescribed_data_talk.pptx` (26 slides; per-slide speaker
notes are embedded in the deck and reproduced here).

---

## 0. One-paragraph abstract

Training robots and humanoids is bottlenecked by *demonstration data*, and the frontier
question is not *how much* data but *which* data. I study **prescribed data**: a closed
loop that evaluates a trained policy, localizes its failure regions, and prescribes the
data to collect next. On a RoboCasa manipulation task, with pi0 and GR00T as students and
an identical-recipe LoRA fine-tuning protocol, I show two things. **(1) Applied result:**
prescribed data works — densifying a policy's top failure categories beats random
collection and beats broad coverage — *but only when concentrated*; spreading the same
budget thin causes collateral forgetting. **(2) Deeper result:** a *trivial* failure-rate
heuristic beat two sophisticated gradient-influence selectors (LESS/TracIn), and the reason
generalizes — **gradient-influence data selection works if and only if the loss gradient
encodes the target distinction**. On this robot task, object identity is a *nuisance* to an
action-prediction loss, exactly as image brightness is a nuisance to a classification loss,
so influence tops out near chance-plus. The contribution is a validated diagnostic and
principle (predict success in advance from the best single-SVD-mode AUC of the gradient
cloud), one genuine method win (whitening: 0.605 → 0.677 ranking AUC), and an honest
ceiling: on this pool the limit is data coverage, not the selection algorithm. The same
"which, not how much" lesson recurs on a real **Unitree G1 humanoid** pouring task via
demonstration-consistency clustering (qualitative + training-loss so far; a rollout-success
eval is the pending rigorous step).

---

## 1. Talk structure & timing (≈ 20 min)

| Part | Content | Time | Slides |
|---|---|---|---|
| Title | The Value of Prescribed Data | 0:30 | 1 |
| Part 1 | Why prescribed data matters | 2:00 | 2–4 |
| Part 2 | What people have done so far | 4:00 | 5–8 |
| Part 3 | My theory — and why it's unique | 2:30 | 9–11 |
| Part 4 | Evidence — H1/H2, H3 (+3 robustness slides), H4, **4b visual retrieval**, H5, G1 | 9:30 | 12–24 |
| Part 5 | The path forward | 2:30 | 25–27 |
| Close | Takeaways | 1:00 | 28 |
| — | Appendix / backup | Q&A | 29–31 |

*(Deck is now 31 slides. Part 4 figure slides: overall-success bar, targeted-region fragility,
per-category heatmap, the height cut, and the visual-retrieval selection chart — see `talk/figs/`.
If pressed for time, the per-category heatmap + height slides can be shown quickly or moved to
backup; the visual-retrieval slide (4b) and the fragility slide are the two most novel additions.)*

The two punchlines are delivered **in sequence**: first the applied win (concentrated
prescribed data beats random), then the deeper methodological principle (selection works
only when the gradient encodes the target).

---

## Part 1 — Why prescribed data matters (2 min)

### The bottleneck is data — and it's expensive
- Robot & humanoid learning has shifted: model architectures and compute are largely
  commoditized (pi0, GR00T, diffusion policies are public). **The scarce resource is
  demonstration data.**
- Every demonstration has a real cost: teleoperation, human operator time, hardware wear,
  scene resets.
- "Just collect more data" scales cost linearly but returns **diminish** — most new demos
  repeat what the policy already does well. A policy at 55% success does not need 10,000
  more *random* demos; it needs the *right* few hundred.
- So the real lever is **data efficiency**: maximize improvement *per added demonstration*.

### Not how much data — which data
- **Prescribed data** = a closed loop that turns a trained policy into a data-collection
  spec: **evaluate** the policy → **find** where it fails → **prescribe** the data type to
  add → collect it → retrain → repeat.
- Contrast with the status quo: collect a big i.i.d. pool once, train, hope coverage is
  enough.
- Grounded in a concrete downstream problem: a **Unitree G1 humanoid** learning to **pour**
  (a chemistry-pouring task) where each teleoperated demo takes minutes — you cannot afford
  to collect everything, so you must choose *which* (see the real-robot corroboration in
  Part 4 and the closing loop in Part 5).
- The two questions this talk answers: (1) does prescribing from failure regions actually
  pay off? and (2) **what makes it work or fail?**

---

## Part 2 — What people have done so far (4 min)

Three families of prior work, and the gap each leaves.

### Approach 1 — scale the data (brute force)
- **Large cross-embodiment datasets:** Open X-Embodiment / RT-X, DROID, BridgeData — pool
  demos across labs, tasks, robots.
- **Bet:** enough diverse data + a big model ⇒ generalization emerges (the LLM scaling-law
  playbook).
- **Works**, and is the backbone of today's VLAs (pi0, GR00T, OpenVLA are trained this way).
- **But:** cost grows linearly, long-tail coverage stays thin, and it says nothing about
  *which* demo to collect *next* for a given policy. Scaling is a strategy for the *field*,
  not for one lab with a fixed budget and a specific weak policy.

### Approach 2 — select the data (be smart about which)
- **Active learning / uncertainty sampling** — collect where the model is least certain
  (classic, decades deep).
- **Influence functions & gradient attribution** — score each training point by its effect
  on a target loss: **LESS, TracIn, DataInf, Datamodels**. Strong results in *LLM
  fine-tuning* and *image classification*.
- **Coreset / coverage / diversity selection** — pick a representative or maximally-spread
  subset.
- **Embedding / scene retrieval** — pick demos *visually* similar to the states the policy
  fails in (BehaviorRetrieval / FlowRetrieval / STRAP-style; CLIP or DINOv2 kNN & coverage).
- **The catch:** almost all of this is validated on **held-out loss** in
  **classification / language**, not on **closed-loop robot success**, and rarely
  head-to-head.

### Approach 3 — correct the data (interactive / failure-driven)
- **DAgger & interactive IL:** an expert corrects the policy at the states it actually
  visits. Right instinct (target failures) but needs an expert in the loop at every step.
- **Hard-example mining / failure replay:** over-sample where the model errs.

### The gap this work targets
1. Validate selection on **closed-loop rollout success**, not held-out loss.
2. Test **cross-policy** — is a weak region universal or one policy's quirk?
3. Run signals **head-to-head** under an identical recipe.
4. Be **honest** — power analysis + adversarial checks, and report the negatives.

---

## Part 3 — My theory, and why it's unique (2.5 min)

### The thesis
> **A trained policy's own failure regions are a prescription for what data to collect next.**

Concretely: evaluate the policy, localize *where* and *how* it fails, then add
demonstrations that **densify exactly those regions** — and expect more
improvement-per-demo than random collection.

Testable sub-claims, built up in sequence:
1. Failures are **structured** (localizable), not random noise.
2. The weak region is a **shared data gap**, not a hardware limit — so data *can* fix it.
3. **Concentrated** targeted data beats random; and — the surprise — does it beat a
   **principled** influence signal?

The honest test: hold the training recipe fixed, change **only** the 200 selected demos,
and measure real rollout success.

### Why this setup is unique
**Methodology**
- Closed-loop: real simulator + real rollout eval (not held-out loss).
- Identical-recipe invariant: every arm is dataclasses-identical except its `data_dirs`.
- Power analysis up front — knows what it can and can't detect.
- Adversarial verification of every positive claim.

**Scope**
- Cross-policy: pi0 (flow-matching) *and* GR00T (diffusion).
- Head-to-head signals: P(fail) · coverage · failure-influence · value-influence.
- A clean sandbox (CIFAR) to isolate *why* methods work.
- Reports the negative results as first-class findings.

> The payoff of this rigor: the single most valuable finding — *why* the sophisticated
> methods failed — only became visible because of the identical-recipe invariant and the
> sandbox.

---

## Part 4 — Evidence: what matters, and what doesn't (8 min)

### The testbed
- **Environment / task:** RoboCasa, `PickPlaceCounterToSink` (rich object/pose variation).
- **Students:** pi0 (~55–58%) and GR00T (~56–66%) — public VLA checkpoints. pi0 baseline
  checkpoint = `pi0_robocasa_pretrain_human300 @ 75000`.
- **Candidate pool:** 9,885 MimicGen demos over 79 object categories.
- **Fine-tune:** LoRA, 20k steps, batch size 32; every arm = 400 shared **base** demos +
  a 200-demo **selected** arm.
- **Eval:** rollout success, n=300, fixed seed 100000, scene-paired across arms; reported
  as overall / targeted-10 / non-targeted.
- **Invariant:** each arm's openpi config is dataclasses-verified identical to `core`
  except its `data_dirs` — only the 200 demos differ.
- **Data generation:** no human demos needed — a validated expert-teacher loop turns
  pi0/GR00T rollouts into trainable LeRobot data.

---

### Hypothesis 1 — failures are structured, not random

**Experiment.** pi0 on `PickPlaceCounterToSink`, n=150 rollouts, per-episode object
geometry logged directly (not category-averaged), a numpy logistic failure predictor, and
an embodiment-limit test. Scripts: `policy_analysis/analyze_pi0_weakregions.py`,
`predict_failure.py`, `progress_analysis.py`, `embodiment_test.py`.

**What matters ✓**
- One failure **mode** dominates: the **grasp**. 76% of failures never touch the object
  (moves < 1 cm); only ~14% ever grasp. The bottleneck is grasp *initiation/approach*, not
  transport or placement.
- Object **height** is a real, *moderate* predictor: 5-fold CV AUC = **0.628**;
  standardized coeff height = −0.57 (all others ≈ 0). Univariate bins: short (<6 cm) 67% >
  medium 55% > tall (>11 cm) **36%** — a clean ~31-point monotonic gap.
- **Not an embodiment limit.** Panda gripper max aperture = 0.08 m; 137/150 objects exceed
  8 cm bounding diameter yet still succeed 52% (the gripper grasps a narrow side / graspable
  part). So the objects *are* graspable → the height-driven failures are a **skill/data gap
  (epistemic, data-addressable)**, not a physical limit. This validates the premise of the
  whole targeted-data experiment.

**What doesn't matter ✗**
- Simple geometry is a **weak** targeting axis. All geometric features together explain
  only **R² ≈ 0.08–0.09** of the variance. Width, lateral position, depth: coefficients ≈ 0.
- The naive eyeball at n=50 ("tall objects fail 0%, short 100%") was **small-sample noise**
  — it evaporated at n=150. A median-split at n=50 with category-averaged geometry then
  swung to a *false null* ("no effect"). Only n=150 + per-episode geometry + a continuous
  predictor gave the truth.
- ~90% of the failure variance is **object-instance-specific** (which Objaverse mesh /
  affordances) or stochastic — *not* captured by hand-picked geometry.

**Takeaway:** localize the *mode* (grasp) robustly; do not over-trust hand-picked geometry
as a targeting signal.

---

### Hypothesis 2 — the weak region is universal (a data gap, not a quirk)

**Experiment.** Replicate the exact weak-region analysis on **GR00T** — a completely
different architecture (diffusion + Eagle VLM) vs pi0 (flow-matching VLA) — n=100. Script:
`Isaac-GR00T/scripts/analyze_groot_weakregions.py` (instrumented inference client) +
`predict_failure.py`.

| Metric | pi0 (flow-matching, n=150) | GR00T (diffusion, n=100) |
|---|---|---|
| Overall success | 52.7% | 56.0% |
| Dominant failure mode | 86% no-grasp | 80% no-grasp |
| Geometry→success CV AUC | 0.628 | 0.629 |
| Dominant predictor (coeff) | height (−0.57) | height (−0.78) |
| Tall-object success | ~36% | ~32% |
| Short-object success | ~67% | ~66% |

**What matters ✓** Two architecturally very different policies fail at the **same rate,
same mode, same objects, same height effect**. Combined with the embodiment test (objects
*are* graspable), this rules out both "it's the gripper" and "it's a pi0 bug." It is a
**shared training-DATA gap** — both were trained on RoboCasa data that under-serves
tall-object grasps → **universal and data-addressable**. Exactly what prescribed data
should fix.

**What doesn't matter ✗** Policy **disagreement** as an epistemic acquisition signal: the
two policies agree *too much* (aggregate concordance is high) to mine disagreement here.

---

### Hypothesis 3 — read the OVERALL number (the targeted region is fragile)

> **Framing note (revised).** The headline metric is **overall success**, not a "targeted-region"
> slice. We cannot confidently claim the 10 chosen categories *are* the hard region, so the
> targeted numbers are kept for context but are **not** the claim. Figures: `figs/fig_overall.png`,
> `fig_target_instability.png`, `fig_percat_heatmap.png`, `fig_height.png`.

**Experiment.** Identical recipe; the 200-demo arm is the only thing that changes: **core** (top-10
failure categories by P(fail)), **random** (uniform control), **coverage** (same budget spread over
the top-25). Eval n=300, seed 100000, scene-paired.

**The trustworthy result — overall success:**

| Arm (200-demo selection) | **Overall** |
|---|---|
| **core** — top-10 failure categories | **0.593** |
| baseline — no fine-tune | 0.580 |
| random | 0.563 |
| failure-influence | 0.553 |
| coverage — spread over top-25 | 0.517 |
| value-influence | 0.503 |

- **core is the only arm above baseline** (0.593 vs 0.580) — a small, honest win.
- **Concentration matters:** coverage (spread over top-25) is the **worst** arm (0.517); it
  significantly regressed the non-targeted majority (0.528 vs core 0.616, **z=−2.04, p<0.05**) —
  thin spread → collateral forgetting.

**Why the targeted-region metric is demoted (four robustness checks — `fig_target_instability`):**

1. **The "hard region" is definition-dependent.** The top-10 hard categories from an **unbalanced**
   vs a **balanced** (Wilson-LB, equal per-cat) eval overlap only **5/10**. core's lift over baseline
   on the targeted region is **+0.099 (old) → −0.007 (balanced)** — the win flips with the definition.
   (`rebalance_targeted.py` → `targeted_rebalanced.json`.)
2. **On a clean stratified eval** (per_cat ≈ 16–39 on the old-10), core **0.371 ≈ random 0.351**
   (+0.02) — not the +0.15 the noisy n=300 slice suggested. "Targeted beats random" barely holds.
3. **Do it ALL right — balanced-select + balanced-measure — and the verdict vanishes.** A dedicated
   stratified eval on the **balanced-10** categories (`eval_balcat_*`, per_cat≈22) gives **baseline
   0.329 · core 0.352 · random 0.331** — core − baseline = **+0.023**, core − random = **+0.021**,
   both inside noise (n = 139–304). When the hard categories are picked from a balanced eval *and*
   measured with balanced sampling, targeting the failure region moves it essentially nothing. This is
   the fourth (rightmost) group in `fig_target_instability` and the tightest evidence in the study.
4. **Per-category, targeting is uneven and misses the hardest** (`fig_percat_heatmap`): core boosts
   *medium* categories (canned_food +0.32, tupperware +0.23, spray +0.21) but barely moves the
   genuinely-hardest (juice −0.01; cheese_grater still 0.19). random helps too (tupperware +0.42,
   beats core).

**The deeper cut — the real weak axis is height, and every arm regressed the tall objects.** Slicing
the n=300 evals by object-height tertile (`fig_height`): baseline tall = **0.52**, and **every**
intervention arm falls below it (core 0.43, random 0.35, coverage 0.38, influence 0.41, value 0.35).
The category-targeted data helped short/medium objects and **hurt** the height-defined weak region —
consistent across all 5 arms. This echoes H1 (geometry R²≈0.08): category is a poor proxy for the
real weak region.

**Takeaway (revised):** trust the **overall** number; concentration beats spread; and treat any
narrow "targeted region" metric with suspicion — the region itself is unreliable.

---

### Hypothesis 4 — a principled influence signal should beat the heuristic (it didn't)

**Experiment.** Score every pool demo by **gradient influence** (LESS / TracIn adapted to
pi0): `score(z) = cos(∇_LoRA flow-loss(z), g_val)`, take top-200, fine-tune, eval. Two
variants:
- **failure-influence** — target the failure region; `g_val = contrast (mean_hard −
  mean_ref)`. Selected 23% targeted (3.4× pool prevalence).
- **value-influence** — target *overall* balanced held-out success; `g_val = plain
  mean(D_val)`, class-balanced D_val, neutral base-only warmup reference. Selected only 6%
  targeted (below random).

Script: `policy_analysis/influence_score.py`. Complete overall eval, n=300, seed 100000,
scene-paired.

| Arm | **OVERALL** | Targeted (old) | Targeted (bal.) | Selection (targeted frac) |
|---|---|---|---|---|
| **core** — trivial P(fail) heuristic | **0.593** | 0.432 | 0.432 | 100% |
| baseline | 0.580 | 0.333 | 0.439 | — |
| random | 0.563 | 0.282 | 0.235 | ~7% |
| failure-influence — contrast g_val | 0.553 | 0.421 | 0.302 | 23% |
| coverage | 0.517 | 0.429 | 0.289 | spread top-25 |
| value-influence — plain g_val | 0.503 | 0.361 | 0.359 | 6% |

Read the **OVERALL** column — it is the robust one. The two targeted columns (old vs balanced
category definition) **disagree**, which is exactly why the targeted slice cannot carry the claim.

**Complete 6-way overall ranking:**
`core 0.593 > baseline 0.580 > random 0.563 > failure-influence 0.553 > coverage 0.517 >
value-influence 0.503.`

**What this shows ✗**
- **Both gradient-influence methods LOSE** — to the trivial heuristic *and* to baseline.
  The sophisticated LESS/TracIn machinery lost to "just add demos from the 10 categories you
  fail on."
- failure-influence worked **where aimed** (targeted-10 0.421 ≈ core 0.432, both ≫ baseline
  0.333) — gradient-targeting genuinely helped the weak spots — but it **regressed the
  non-targeted majority** (0.573 vs ~0.616), and that ate the gains.
- value-influence picked data **nearly disjoint** from the failure region (6% targeted;
  value ∩ failure = 5/200, value ∩ core = 4/200). It loaded common "value-dense" objects
  (tongs ×14, dish_brush ×12, mug ×11, rolling_pin ×8…). This *looked* like evidence against
  "failure ⇒ missing data," but see the postmortem: the proxy mis-identified value, so it is
  not clean evidence — it is a **selector failure**.

**The unifying observation:** the **non-targeted column predicts the entire ranking.** Every
arm that perturbs the shared-grasp behavior forgets and loses overall; **only core leaves it
intact** (0.616 ≈ 0.617). The heuristic won by *doing less damage*, not by being smarter.
This is catastrophic forgetting of the shared grasp behavior.

**Statistics (n=300, paired McNemar):** only `value < core` is significant (p ≈ 0.02).
`failure-influence vs core` Δ = −0.040 (p ≈ 0.30, NS). Defensible claims: *core is (weakly)
best and the only arm beating baseline; coverage & value significantly regress the majority;
the influence methods did not beat the heuristic.*

---

### Experiment 4b — visual-driven retrieval (2.3 CLIP · 2.4 DINOv2)

Beyond gradients, the natural thing to try is **retrieval**: pick pool demos whose *scene looks
like* the scenes pi0 actually fails in. Two were built (`fig_selection_targeting.png`):

- **2.3 — CLIP retrieval** (`policy_analysis/score_retrieval.py`, CLIP-ViT-L/14): render pi0's
  failure scenes → CLIP-embed; embed every pool demo's first agentview frame; **score(demo) = mean
  of its top-k cosine similarities to the failure set**; take top-200. (`retrieval_core.json` →
  `arms_retrieval.json`.)
- **2.4 — DINOv2 greedy coverage** (`policy_analysis/score_pool.py --encoder dinov2 --method
  submod_coverage`): embed failures + pool with DINOv2, then **greedy facility-location** — add, one
  at a time, the demo with the largest **marginal coverage gain** of the failure distribution
  (diverse + relevant by construction). (`submod_core.json` → `arms_submod.json`.)
- (A third variant, **DINOv2 fail-minus-success contrast** = the `failretr` arm, `failretr.py`, was
  fine-tuned + rollout-evaluated → **0.297** on the targeted-10 (stratified), **below random 0.351 and
  core 0.371**, barely above baseline 0.262. So the one visual arm carried to rollout also loses.)

**Selection-level result (the honest finding).** All three visual selectors pick demos that are
**≈ indistinguishable from random** on the failure region: **CLIP 10%, DINOv2-greedy 9%,
DINOv2-contrast 8%** of their 200 picks fall in the failure categories — vs random/pool **~6–7%**
(core is 100% by construction; failure-influence reaches 22%). Visual *scene* similarity does **not**
recover the hard objects, because a whole-scene embedding (counter, layout, robot arm) is dominated by
the **shared kitchen common-mode** — "looks like a failure scene" ≈ "looks like a kitchen," not "is the
specific tall object that fails." This is the **same common-mode / encoding failure as the gradient
methods (H5), now in visual feature space.**

**Status (important):** 2.3 and 2.4 were **selection-tested, not rollout-fine-tuned/evaluated** (no
`ppc2sink_base_{retrieval,submod}` datasets, no openpi configs) — the selection signal already predicts
no failure-region concentration. The **DINOv2-contrast `failretr` arm WAS carried to rollout: 0.297 on
the targeted-10 (stratified), below random (0.351) and core (0.371)** — confirming that visual retrieval
loses not only at selection but at rollout.

### The full landscape of selection signals tried

| family | method | selection targeted-% (old-10) | rollout overall | status |
|---|---|---|---|---|
| control | random | ~6–7% | 0.563 | evaled |
| heuristic | **core** (P(fail) top-10) | 100% | **0.593** | evaled |
| heuristic | coverage (top-25) | spread | 0.517 | evaled |
| gradient | failure-influence (LESS contrast) | 22% | 0.553 | evaled |
| gradient | value-influence (LESS plain) | 6% | 0.503 | evaled |
| gradient | whitening | — | 0.268 strat (≈ base 0.262) | evaled (strat) |
| gradient | RC-LESS v1 / v2 | — | 0.477 / **0.590 ≈ core** | evaled |
| visual | **2.3 CLIP retrieval** | 10% | — | selection only |
| visual | **2.4 DINOv2 greedy-coverage** | 9% | — | selection only |
| visual | DINOv2 contrast (failretr) | 8% | **0.297** targeted-10 (< random 0.351, core 0.371) | evaled |
| hybrid | catfree (model-hard + whiten) | 14% | — | selection only |

Across **three families** (heuristic, gradient-influence, visual-retrieval), no method rollout-beat the
trivial P(fail) heuristic; RC-LESS v2 (the best-engineered gradient selector) only **ties** core (0.590).
The visual methods don't even concentrate on failures at the selection stage. One unifying cause: on this
task the target (the hard object) is a **common-mode nuisance** in *every* feature space tried — action
gradients and scene embeddings alike.

---

### Hypothesis 5 (the synthesis) — influence works iff the gradient encodes the target

**Why** did influence flop on the robot task but shine in the LLM/vision literature? To
isolate the mechanism I built a clean CIFAR sandbox with a ground-truth "failure region" (20
rare classes at 10%), where everything is controllable. Scripts: `xgradtest/` (CIFAR),
`policy_analysis/influence_offline.py` (robocasa diagnostics + whitening).

| Setting | Does the loss gradient encode the target? | Best SVD-mode AUC | Influence AUC |
|---|---|---|---|
| CIFAR — select by **class** (cross-entropy on class) | **YES** — the target *is* the loss | 0.66 | **0.96** ✓ |
| RoboCasa — select by **object** (flow-loss on **actions**) | **NO** — object is a *nuisance* to the action loss | 0.56 | 0.60 ✗ |
| CIFAR — select by **brightness** (a nuisance var) | **NO** — nuisance to classification loss | 0.53 | 0.50 ✗ |

**The principle.** Gradient-influence data selection works **iff the loss gradient encodes
the target distinction.** You can predict the ceiling *cheaply, in advance*: the **best
single-SVD-mode AUC** of the candidate gradient cloud for your target. RoboCasa
object-category is a *nuisance w.r.t. the action-prediction loss*, exactly as brightness is
a nuisance w.r.t. a classification loss — which is why *every* selector (influence,
contrast, RC-LESS, per-instance) tops out ~0.60. Effective rank is **not** the
discriminator (robocasa rank 1183 > CIFAR 200–586); the *encoding of the target* is.

**And it isn't only gradients (Exp 4b).** The same target — object identity — is *also* a
common-mode nuisance in **visual feature space**: CLIP/DINOv2 whole-scene embeddings select
≈random on the failure region (8–10% ≈ pool 6–7%). So the encoding principle generalizes: gradient
space *or* embedding space, if the target is swamped by a shared common-mode, retrieval/influence
selection cannot recover it.

**The one method win — whitening.** Project out the top-k *shared* ("common-mode") SVD modes
from candidates + target, then cosine (a generalized contrast): robocasa ranking AUC
**0.605 → 0.677** at k=50 (+0.071); top-200 targeted purity **17% → 36%**; max 0.683 at
k=70. Controls pass (random-direction removal is flat; shuffled-label 0.49). You **must
bound k**: on CIFAR (signal already 0.96) whitening helps only ~+0.02 and *collapses* past
k≈30 — over-whitening eats the signal. Whitening beats contrast (0.586) and raw cosine
(0.605).

**…but rollout ≠ ranking (the honest close).** The whitening selection was fine-tuned and
**rollout-evaluated** (stratified, per_cat ≈ 35). It lands at **0.268 ≈ baseline 0.262**, well
below core's **0.371** — the ranking-AUC win did **not** translate into rollout success, exactly as
the honest hedge predicted. A better *ranking* of demos is not a better *policy*; this is the honest
boundary on the whole influence line of work.

**Per-instance targeting is a dead end here.** Per-demo max / top-K / k-means / kNN all ≈
category-mean (best deterministic 0.613 vs 0.607 — noise). The signal is genuinely weak
(~0.61 ceiling), not an averaging artifact — keep the simple category-mean.

---

### Evidence, distilled: what matters vs. what doesn't

| **What matters / is trustworthy ✓** | **What's misleading / backfires ✗** |
|---|---|
| The **OVERALL** number — one metric, no sub-region cherry-picking | The **targeted-region** metric — 2 category defs disagree (5/10) |
| Localizing the failure **mode** (grasp) — robust & cross-policy | Category as a weak-region proxy (geometry R² ≈ 0.08) |
| A data-addressable gap (not a hardware limit) | Targeting by category **HURT** the real (tall) weak region |
| **Concentration** over spread (coverage was worst) | Sophisticated influence when the gradient can't see the target |
| Preserving the shared skill (do-no-harm on the majority) | Whitening's ranking win (0.677) — did **not** translate to rollout |
| Matching the selection **signal** to what the loss encodes | "Value" selection that drifts off the failure region |

Through-line: trust **aggregate, distribution-preserving** signals (overall success, failure mode,
encoding); distrust any **narrow sub-region** metric — including the "targeted-10" this project
started out believing in. Whether a clever signal helps *at all* is decided by one thing — **encoding**.

---

### A real-robot echo — Unitree G1 chemistry pouring

The sim study was a proxy for a real problem, and the same headline shows up on hardware —
reached through a *different* selection axis (demonstration consistency, not failure targeting).

**The setup.** A **Unitree G1 humanoid** with dual **Dex3** dexterous hands performs a
**chemistry-pouring** task (pick up a tube, pour into a beaker). 28-D state / 28-D action
(both arms + hands), a single wrist/high camera at 30 fps. Data is collected by human XR
teleoperation — each episode is *minutes* long, so the datasets are tiny: 11 episodes /
8,792 frames (`chem_pour_xl`), and a separate 26-episode / 18,004-frame left-arm pour set
(`pour_left_sree`). Trained on the same 2×H100 box; policies available: diffusion / ACT /
pi0 / GR00T / SmolVLA. Pipeline: `/data/xinyua11/unitree_lerobot_pipeline`
(teleop → LeRobot convert → train → eval), one-YAML-per-experiment with per-run provenance.

**Prescribed selection (SREE clustering) — the real-robot prescribed-data moment.**
- Training a diffusion policy on the **full 26-episode pool** produced a policy that
  **mode-collapsed** — it froze mid-episode — because the demos spanned two inconsistent
  motion styles plus outliers (episodes 11, 17).
- Rather than random subsampling, a **PCA + pairwise-trajectory-MSE clustering** analysis
  (`code/scripts/create_filtered_dataset.py`) identified the tightest, most consistent
  sub-cluster (episodes 0–7) and dropped the rest → dataset `pour_left_sree_cluster0`.
- **−65% of frames** (18,004 → 6,272), 8 of 26 episodes kept.
- **Result:** the policy converged to a **clean single mode with no freezing** on the
  consistent cluster. Config `c01_sree_cluster0_diffusion.yaml`, 100k steps, ~1.85 h/H100.

**Companion ablations (same task).**
- `bg01_remove_bg` — a `rembg`/U²-Net **background-removal** transform on the camera frames
  (visual-consistency ablation), full 100k-step run completed; a 200-step smoke test
  validated the transform→train wiring first.
- `b02_diffusion_seed_study` — a **5-seed** (1000–5000) consistency/variance study; all five
  converged consistently (~1.8 h each) — the real-robot echo of the sim power analysis.

**Honest caveat (important, given the talk's rigor theme).** The G1 evidence to date is
**qualitative** (mode collapse fixed) plus **training-loss / convergence** — it is **not**
yet a controlled **rollout-success** comparison on the robot (and training loss is not
comparable across differently-sized datasets, so I do not lean on the loss numbers). That
missing rollout-success eval is *exactly* the rigorous evaluation Part 5 calls for. What the
G1 work establishes is that the thesis is **not a simulator artifact**: on a real humanoid,
with a different selection mechanism, *which demonstrations you train on beats how many.*

---

## Part 5 — The path forward (2.5 min)

### Making prescribed selection actually work
1. **Re-aim the gradient at what the action loss encodes** — not object identity:
   grasp-success / motion-phase / per-scene targets. Quantitative bar to clear before
   spending a GPU-hour: **best-mode AUC ≳ 0.7**.
2. **RC-LESS** — the stacked selector that fixes the coupled defects together:
   **retention-constrained** (do-no-harm on the majority) + **coverage-aware**
   (facility-location, not naive top-k) + **Adam-preconditioned** (magnitude-aware) +
   **whitened** (common-mode removed), with a **per-category floor**. By construction it
   *never does worse than core*; it can only win via within-category structure that core,
   being category-coarse, cannot see. Cost: one rescore pass + one 2k-step mini-fit; no
   extra full fine-tunes to rank.
3. **Gate on success, not loss:** a cheap 2k-step LoRA mini-fine-tune that **rejects
   MSE-good / success-bad selections** (the exact trap value/coverage fell into) before any
   full run — the one place actual rollout success enters the loop.
4. **Honest ceiling:** analysis suggests this is a **data-coverage** limit, not an algorithm
   limit. When the most-valuable data lies *outside* the failure region and the pool cannot
   manufacture targeted success, no reweighting/reselection of the existing 9,885 demos
   clears core — only **collecting genuinely different demonstrations** does. Which loops
   back to the point of prescribed data. Treat any sub-0.04 improvement over core as
   *unproven at n=300*.

### From simulator to real robots
**Power up the evidence**
- Paired-seed design + ~16k rollouts to resolve the 1–4 pt gaps that n=300 cannot.
- Multiple tasks & seeds — beyond single-task, single-run.
- Fix the rollout-eval hang (per-rollout timeout) for clean large-n runs.

**Close the real-robot loop**
- The **G1 pouring pipeline is already built** end-to-end (teleop → LeRobot → train → eval);
  the SREE clustering result is the first prescribed-data win on it.
- **Next:** a proper **rollout-success** eval on the robot — not just training-loss /
  mode-collapse — to hold the real-robot claim to the same bar as the sim study.
- Generate prescribed data on demand via the **expert-teacher loop** (pi0/GR00T → trainable
  data, no human demos), so once a region is prescribed it can actually be *filled*.
- Prescribe → collect exactly that region → retrain → re-evaluate **on the G1**.

---

## Close — Takeaways (both punchlines, in sequence)

1. **Measure prescription by OVERALL success, not a targeted slice.** The "targeted region" is
   fragile — two reasonable category definitions disagree 5/10, and category-targeting even *hurt*
   the real, height-defined weak region. On the trustworthy overall metric, concentrated
   failure-data (core) is the only arm above baseline — a small, honest win.
2. **Selection only works when the gradient encodes the target.** A trivial P(fail) heuristic beat
   sophisticated LESS/TracIn influence, and whitening's ranking win did **not** translate to rollout
   (0.268 ≈ baseline) — because object identity is a nuisance to an action-prediction loss. Predict
   it in advance (best-mode AUC).
3. **The contribution is a validated diagnostic + a discipline of honesty** — not a SOTA selector:
   report aggregate metrics, distrust narrow sub-regions, and check the signal is even encodable.

> **The through-line: don't just add more data — prescribe the right data, measure it honestly, and
> first check your training signal can even see it.**

---

## Appendix A — influence method & why it's hard

- **Score:** `score(z) = cos(∇_LoRA flow-loss(z), g_val)`, take top-200. **Streamed** (no
  projection matrix): build `g_val` first, then dot each candidate's gradient on-device and
  keep only the scalar — the trainable set (LoRA adapters + full SigLIP vision tower +
  action-expert heads under `freeze_filter = All(.*llm.*, Not(.*lora.*))`) is hundreds of
  millions of params, so a fixed Gaussian projection cannot be materialized. Gradient target
  = the ~50M LoRA adapters.
- **K = 8** frames/demo; loss **masked to the 12 real action dims** (`--real_dim 12`) —
  pi0 pads actions 12→32 and the 20 zero-pad columns carry pure-noise flow targets that
  otherwise swamp the gradient.
- **Gradient target matters:** last-layer readout (`action_out_proj`) is motion-dominated &
  object-blind (smoke AUC 0.44) → must use the LoRA adapters (they route through
  object-conditioned attention).
- **Reference checkpoint matters** (the single most consequential choice): must be
  **selection-neutral** *and* **in-regime** with non-degenerate LoRA gradients → a
  **base-only LoRA warmup** (fine-tune pretrained pi0 on the 400 base demos, save
  2k/4k/6k), *not* the trained arms (biased — bakes in a selection & already absorbs the
  failure demos), *not* cold pretrained pi0 (LoRA adapters are a no-op → gradient identically
  zero on half the dims, degenerate).
- **g_val direction:** `contrast (mean_hard − mean_ref)` for the *failure* target (cancels
  the common mode; smoke AUC 0.35 → 0.63); `plain mean` for the *value* target (contrast
  would re-bake the failure target). Plain mean collapses onto the common "generic
  pick-place" mode (97% of candidates have positive cosine; hard-vs-easy AUC 0.35).
- **Four coupled defects** (all one root — the dominant common-mode direction *is* the
  forgetting direction):
  (F1) common-mode collapse (residual mean cancels at O(σ/√n) while the shared term stays
  O(1)); (F2) magnitude-blindness (unit cosine drops ‖∇L‖, and top-cosine demos carry the
  *smallest* gradients — the easy/redundant ones); (F4) loss ≠ success (flow-MSE is a lossy,
  open-loop proxy for sparse, non-differentiable, policy-induced task success); (F3/F5) no
  retention / no diversity term (modular top-k drops the submodular redundancy penalty — it
  picked 14 tongs — and the retention coupling that protects the 88% majority).

---

## Appendix A2 — RC-LESS (the roadmap's "single best next method")

Two stages (`policy_analysis/influence_score.py sketch` → `rc_select.py`):

1. **Gradient sketch.** For each demo, `g(z) = normalize(∇_LoRA flow-matching-loss)` at a neutral
   base-only LoRA warmup, K=8 frames, masked to the 12 real action dims. The ~50M-dim gradient is
   compressed with a **fixed very-sparse Johnson–Lindenstrauss projection** (preserves cosine).
   Produces sketches for `dval` (targeted-10 held-out), `ret` (class-balanced non-targeted holdout),
   and `cand` (pool).
2. **Retention-constrained coverage select.** g_R = mean retention direction (the shared-grasp /
   forgetting direction, ≈0.78 collinear with the task gradient); k-means the D_val sketches into
   **m=14 gradient modes**; **coverage(z)** = mean top-3 cosine to the modes; **score(z) =
   coverage(z) − λ·max(0, ⟨z, g_R⟩ − ρ)** (retention penalty); select via a **per-category floor +
   cap** on the targeted-10 (reimplements core's depth-on-holes ⇒ RC-LESS ≥ core by construction)
   then **greedy facility-location** fill.

**What ran, and the punchline:**

| variant | key knobs | targeted-frac | rollout n=300 |
|---|---|---|---|
| RC-LESS v1 (`rc_core.json`) | λ=1 · contrast-center ON | 0.60 | **0.477** (worst) |
| RC-LESS v2 (`rc_core_v2.json`) | λ=0 · no-center · floor=20 | 0.96 | **0.590** |
| core (trivial heuristic) | top-10 P(fail) | 1.00 | **0.593** |

v1's centering + retention penalty steer selection *away* from the task gradient (because g_R ≈ 0.78 ∥
it) → below baseline. v2 turns them off, and floor=20 × 10 targeted categories fills the entire
200-demo budget → **RC-LESS degenerates to core** (0.590 ≈ 0.593). The best-engineered selector in the
study **never beats the trivial heuristic** — the strongest single piece of evidence that this is a
**data-coverage ceiling, not a selection-algorithm ceiling**. (Caveat: v2's floor/cap are so large the
coverage/retention machinery barely acts — it's a demonstration that *making RC-LESS safe turns it into
core*, not a clean test of coverage+retention; a floor≈8/cap≈15/small-λ magnitude-aware variant was
proposed but not run to rollout.) Files: `policy_analysis/{influence_score.py,rc_select.py}`,
`weakregion/{rc_core,rc_core_v2}.json`, evals `weakregion/eval_{rc,rc2}`.

---

## Appendix B — power, honesty, and statistics

- **Power analysis up front** (`policy_analysis/power_analysis.py`): for 80% power (p≈0.45,
  weak-region fraction 0.5), detecting Δ=15% needs ~5–9 seeds; Δ=10% needs ~12–24; Δ=5%
  needs ~50–125 (infeasible). MDE for S=3, R=100 is ~20–28%. **n=300 resolves only large
  effects.** Between-seed SD of success (σ_seed) is the dominant unknown — measure it first.
- At **n=300 paired McNemar**, only **`value < core` is significant** (p ≈ 0.02); most
  overall gaps (1–4 pts) are within noise.
- **Defensible claims:** core is (weakly) best & the only arm beating baseline; coverage &
  value significantly regress the majority; the influence methods did not beat the heuristic.
- **A worked example of small-sample overconfidence:** the "height fails 0%" claim at n=50
  was refuted at n=150 (and a crude n=50 median-split produced a *false null* in between).
- **The CIFAR sandbox** gives clean ground truth (influence AUC 0.96–1.0) → the influence
  *machinery is sound*; robocasa's weakness is the **target encoding**, not a bug.
- Every positive result was **adversarially re-derived** (e.g. the gradient-encoding
  principle, re-derived 2026-06-30); negative results are reported as first-class findings.

---

## Appendix C — artifacts & reproduction

- **Weak-region analysis:** `policy_analysis/analyze_pi0_weakregions.py`,
  `predict_failure.py`, `progress_analysis.py`, `embodiment_test.py`,
  `coverage_geometry.py`. Outputs under `weakregion/pi0_PickPlaceCounterToSink/`,
  `weakregion/groot_PickPlaceCounterToSink/`.
- **Selection arms:** `core`/`random`/`coverage` via `build_arms.py` +
  `build_lerobot_subset.py`; datasets `/data/xinyua11/ft_arms/ppc2sink_base_{core,random,
  coverage,influence,value}`.
- **Visual retrieval:** `policy_analysis/score_retrieval.py` (2.3 CLIP mean-top-k) →
  `retrieval_core.json`/`arms_retrieval.json`; `policy_analysis/score_pool.py --encoder dinov2
  --method submod_coverage` (2.4 greedy facility-location) → `submod_core.json`/`arms_submod.json`;
  `policy_analysis/failretr.py` (DINOv2 fail−success contrast) → `arms_failretr.json`. Selection-only
  except failretr (rollout-evaled: 0.297 targeted-10). Figure: `talk/figs/fig_selection_targeting.png`.
- **Influence:** `policy_analysis/influence_score.py` (`probe|smoke|vsmoke|full`),
  `influence_offline.py` (offline diagnostics + whitening); selections
  `weakregion/influence_core.json`, `weakregion/value_core.json`; reference config
  `pi0_ppc2sink_basewarmup`.
- **CIFAR sandbox:** `/data/xinyua11/xgradtest/` — `xgrad.py`, `grad_geometry.py`,
  `cifar_control.py`.
- **Fine-tune configs (openpi):** `pi0_ppc2sink_{core,random,coverage,influence,value,
  basewarmup}` — each dataclasses-verified identical except `data_dirs` (the invariant).
- **Eval harness:** `serve_and_eval.sh` / `serve_and_eval_strat.sh`, aggregated by
  `aggregate_eval4.py`. n=300, seed 100000, scene-paired.
- **Reports:** `weakregion/INFLUENCE_REPORT.md`, `INFLUENCE_IMPROVEMENT_ROADMAP.md`
  (RC-LESS design + theoretical ceiling), `VALUE_INFLUENCE_POSTMORTEM.md`.
- **G1 real-robot pipeline:** `/data/xinyua11/unitree_lerobot_pipeline` — `PIPELINE.md`,
  configs `code/configs/experiments/g1_chem_pour/{b01_baseline, b02_diffusion_seed_study,
  bg01_remove_bg, c01_sree_cluster0_diffusion}.yaml`; clustering script
  `code/scripts/create_filtered_dataset.py`; runs under `runs/{g1_chem_pour_xl_diff_bg,
  g1_chem_pour_xl_diff_seed_study, sree_cluster0_diff, pour_left_sree_diff_250k}`; dataset
  `/data/xinyua11/datasets/g1_chem_pour_xl` (HF `liuuu121/chem_pour_xl`).

---

## Appendix D — key numbers reference card

| Fact | Value |
|---|---|
| pi0 baseline success | ~52.7–58% |
| GR00T baseline success | ~56–66% |
| Dominant failure mode | grasp (76% never-touched; 80–86% no-grasp) |
| Height predictor CV AUC (pi0 / GR00T) | 0.628 / 0.629 |
| Tall vs short object success | ~36% vs ~67% |
| Geometry variance explained (R²) | ~0.08 |
| Candidate pool | 9,885 demos / 79 categories |
| Fine-tune budget | 400 base + 200 selected, LoRA 20k steps |
| **core** overall / targeted / non-targeted | **0.593 / 0.432 / 0.616** |
| baseline | 0.580 / 0.333 / 0.617 |
| random | 0.563 / 0.282 / 0.605 |
| failure-influence | 0.553 / 0.421 / 0.573 |
| coverage | 0.517 / 0.429 / 0.528 |
| value-influence | 0.503 / 0.361 / 0.523 |
| **Targeted-region fragility** | old vs balanced top-10 overlap **5/10**; core lift +0.099 → **−0.007** |
| Stratified old-10 (per_cat≈16–39) | baseline 0.262 · **core 0.371 ≈ random 0.351** · whiten 0.268 |
| **Balanced-select + balanced-measure** (the clean one) | baseline **0.329** · core **0.352** · random **0.331** → core−base +0.023, core−rand +0.021 (all tie) |
| **Whiten ROLLOUT** (H5, now done) | **0.268 ≈ baseline 0.262 ≪ core 0.371** → ranking win did NOT translate |
| Height regression (tall tertile) | baseline **0.52**; every arm below (core 0.43, random 0.35, …) |
| Per-category (core Δ vs base) | helps canned_food +0.32 / tupperware +0.23; misses juice −0.01, cheese_grater still 0.19 |
| Influence AUC — CIFAR class / robocasa obj / CIFAR brightness | 0.96 / 0.60 / 0.50 |
| **Visual retrieval** selection targeted-% (2.3 CLIP / 2.4 DINOv2 / failretr) | 10% / 9% / 8% ≈ random 6% (selection-only, no rollout) |
| RC-LESS v2 rollout | 0.590 ≈ core 0.593 (ties, best gradient selector) |
| Whitening ranking AUC (robocasa) | 0.605 → 0.677 (bounded k) |
| Eval significance (n=300 McNemar) | only value < core significant (p≈0.02) |
| **G1 pouring** robot / hands | Unitree G1 humanoid + Dex3 (28-D state/action) |
| G1 dataset (chem_pour_xl) | 11 episodes / 8,792 frames / 30 fps (~4.9 min) |
| G1 SREE clustering selection | 8 of 26 episodes kept; 18,004 → 6,272 frames (−65%) |
| G1 SREE result | full pool mode-collapses/freezes; consistent cluster converges clean (qualitative + train-loss, no rollout eval yet) |

---

## Speaker script (per slide)

The following is the spoken script, sized to ~20 minutes; it is also embedded as speaker
notes in `prescribed_data_talk.pptx`.

**[Slide 1 — Title]**
Good [morning/afternoon]. My talk is called *The Value of Prescribed Data*. The one-line
version: when we train robots and humanoids, the bottleneck is no longer compute or model
size — it's data. And the question I care about is not *how much* data, but *which* data.
"Prescribed data" means letting the policy itself tell us what to collect next. I'll give
you two things by the end: an applied result — prescribed data from a policy's own failure
regions beats random data, when collected in a concentrated way — and a deeper, more
surprising lesson — sophisticated data-selection methods lost to a trivial heuristic, and I
can tell you the exact principle that governs when data selection works at all. This is a
talk with honest negative results; those turned out to be the most useful part.

**[Slide 3 — the bottleneck]**
Start with the pain. In modern robot learning the model isn't the hard part anymore — pi0,
GR00T, diffusion policies are all public. What's expensive is data: every demo costs human
teleoperation time, operator attention, hardware wear, scene resets. And here's the trap
everyone falls into: "the policy is at 55%, let's collect more data." But more *random* data
mostly re-teaches what the policy already knows; returns diminish fast. If your policy
already grasps mugs reliably, the hundredth mug demo is nearly worthless. So the objective is
data efficiency: improvement per added demonstration. That reframes the problem from "how
much" to "which."

**[Slide 4 — which data]**
Here's the idea in one word: prescription. Instead of collecting a giant i.i.d. dataset once
and hoping, we close a loop: evaluate, find where it fails, prescribe what kind of data to
add, collect exactly that, retrain, repeat. A doctor doesn't prescribe every drug in the
pharmacy — they diagnose, then prescribe the specific thing. We want the policy to diagnose
itself. This is motivated by a real-robot pouring and data-selection task where you genuinely
can't collect everything. Two questions for the rest of the talk: does prescribing from
failure regions actually beat random data? And — the more interesting one — what determines
whether it works?

**[Slide 6 — scale]**
Family one: scale it. Open X-Embodiment, RT-X, DROID, Bridge — pool enormous amounts of data
across labs and robots, train a big model. The LLM scaling-law playbook, and it works — it's
how every VLA you've heard of was built. But cost is linear, the long tail stays thin, and —
most important for us — it's a *field-level* strategy. It never answers "which demo next?"
for one lab with a fixed budget and one specific weak policy.

**[Slide 7 — select]**
Family two, where my work lives: select it. The oldest branch is active learning — collect
where the model is uncertain. The branch I lean on is influence functions and gradient
attribution — LESS, TracIn, DataInf, Datamodels — score every candidate by how much its
gradient aligns with a target gradient, take the top ones. Real, strong results — in LLM
fine-tuning and image classification. And coverage/coreset selection: pick a diverse subset.
Here's the catch that motivated my whole project: almost all this evidence is on *held-out
loss*, in *classification or language*. Very little has been tested on *closed-loop robot
success*, where a policy rolls out, compounds its errors, and either finishes or doesn't. And
they're rarely run head-to-head. That's the gap.

**[Slide 8 — correct + the gap]**
Family three: correct it. DAgger puts an expert in the loop to correct the policy at the
states it visits — the right instinct, target failures — but it needs that expert online at
every step. So the gap I target, on the right: validate selection on closed-loop rollout
success; check whether a weak region is universal across very different policies; run signals
head-to-head under a strictly identical recipe; and be ruthlessly honest — power analysis,
adversarial checks, report the negatives. That honesty is what made the project informative.

**[Slide 10 — thesis]**
My thesis in one sentence: a trained policy's own failure regions are a prescription for what
to collect next. Evaluate, localize where and how it fails, add demos that densify exactly
those regions, and you should get more improvement per demo than random. That breaks into a
chain of testable claims: are failures even structured; is the weak region fixable data or a
hardware limit; does concentrated targeted data beat random — and can a fancy influence
method beat a dumb heuristic. The discipline that holds it together: fix the recipe
completely, change only the 200 demos, measure actual rollout success.

**[Slide 11 — why unique]**
What makes this different: everything is closed-loop — a real simulator, real rollout eval,
not a loss proxy. Every experiment obeys an identical-recipe invariant — the only thing that
changes is which 200 demos I add. I did a power analysis up front, so I know what I can
detect. I adversarially verified every positive claim. I test cross-policy — pi0
flow-matching and GR00T diffusion. I run four selection signals head-to-head. And I built a
clean CIFAR sandbox to isolate *why* methods work. The payoff: the single most valuable
finding — why the sophisticated methods lost — only became visible because of the invariant
and the sandbox.

**[Slide 12 — testbed]**
Quick setup. One RoboCasa task, PickPlaceCounterToSink — deliberately one task so I can debug
it, but with real object, pose, grasp, and placement variation. Students are pi0 and GR00T,
off-the-shelf, around 55%. The candidate pool is ~9,900 MimicGen demos across 79 categories.
Every experiment fine-tunes with LoRA for 20k steps on 400 base demos plus a 200-demo
selected arm. Eval is rollout success — n=300, fixed seed, scene-paired. And I always split
results three ways: overall, the targeted-10 categories, and the non-targeted majority —
because that split is where the whole story lives.

**[Slide 13 — H1]**
Hypothesis 1: failures are structured — you can localize them. Experiment: 150 pi0 rollouts,
log per-episode geometry, fit a logistic predictor, run an embodiment test. What matters:
one dominant mode — the grasp; 76% of failures never touch the object. And height is a real
predictor — AUC 0.63, tall 36% versus short 67%. Crucially, the embodiment test shows the
objects fit the gripper — they *are* graspable — so this is a data-addressable skill gap, not
hardware. But — the first "what doesn't matter" — geometry is weak: all features together
explain 8% of variance; the n=50 eyeball said "tall fails 100%" and that was pure noise. 90%
of the variance is object-instance-specific. Localize the mode robustly; don't over-trust
geometry.

**[Slide 14 — H2]**
Hypothesis 2: is the weak region universal or a pi0 quirk? If universal across very different
models, it's a shared *data* gap. Experiment: run the same analysis on GR00T — diffusion with
an Eagle VLM, versus pi0's flow-matching. The table is almost eerie: same overall success,
same grasp-dominated mode, same AUC 0.628 versus 0.629, same height predictor, same
tall-object 32–36%, same short-object 66%. Two architectures that share nothing internally
fail identically. With the embodiment test, that rules out the gripper and rules out a pi0
bug — it's a shared training-data gap. Exactly what prescribed data should fix. The one
"doesn't matter": I hoped to mine policy disagreement, but they agree too much.

**[Slide 15 — H3]**
The central applied experiment. Same recipe, only the 200-demo arm changes: core concentrates
on the top-10 failure categories; random is the control; coverage spreads the same budget
over the top-25. Punchline first: core wins — 0.593, the only arm beating the 0.580 baseline.
It lifts the failure region from 0.333 to 0.432 without hurting the majority, which stays at
0.616 versus baseline 0.617. Surgical. Now the part I didn't expect: coverage, spreading the
*same* budget wider, is the *worst* arm — 0.517. It matched core on shared categories but
significantly regressed the majority, z of −2.04. Thin spread causes drift and forgetting.
The lesson: targeting works, but *concentration* is doing the work. Depth beats breadth.

**[Slide 16 — H4]**
Here's where I expected to show off. Core is a dumb heuristic — top-10 categories by failure
rate. Surely a principled method — gradient influence, LESS/TracIn — does better? I built two
variants: failure-influence aimed at the weak region, value-influence aimed at overall
success. The result humbled me. Both *lose* — not just to core, to baseline. Read the order:
core 0.593, baseline 0.580, random 0.563, failure-influence 0.553, coverage 0.517,
value-influence 0.503. The trivial heuristic beat both sophisticated selectors. Two
diagnostics: failure-influence worked where aimed — 0.421, matching core — but regressed the
majority and that ate the gains. And value-influence, asked "what improves overall success,"
picked demos almost entirely *outside* the failure region, only 6% targeted. The unifying
fact: the non-targeted column predicts the whole ranking. Every arm that perturbs the shared
grasp forgets and loses. Only core leaves it intact. The heuristic won by doing less damage.
That demanded an explanation.

**[Slide 17 — H5]**
The synthesis, and the real contribution. Why did influence work in the LLM/vision literature
but flop here? I built a clean CIFAR sandbox with a ground-truth failure region. The table is
the whole story. Select CIFAR images by class, with cross-entropy on that class — influence
gets 0.96, brilliant. Select CIFAR images by brightness — a nuisance the loss ignores —
influence collapses to 0.50, chance. Robocasa sits in the middle at 0.60, and now we know
why: I select by object identity, but the loss is flow-matching on *actions*, and object
identity is a nuisance to an action loss — structurally the same as brightness for a
classifier. So the principle: influence selection works iff the loss gradient encodes the
target — and you can predict the ceiling in advance from the best single SVD-mode AUC of the
gradient cloud. There was one real method win: whitening — project out the shared common-mode
directions and robocasa ranking goes 0.605 to 0.677, purity 17 to 36% — but you must bound
how many modes you remove or you delete the signal too.

**[Slide 18 — distilled]**
All five experiments on one slide — the "what matters / what doesn't" the title promised.
Matters: localizing the mode; confirming it's a data gap; concentration; preserving the
shared skill; matching the signal to what the loss encodes; whitening with bounded k. Doesn't,
or backfires: hand-picked geometry and small-sample eyeballing; spreading thin for coverage;
sophisticated influence when the gradient can't see the target; value selection that drifts
off the failure region; last-layer TracIn; policy disagreement. Through-line: the winning
moves concentrate and preserve; the losing moves spread and perturb.

**[Slide 20 — G1 real-robot corroboration]**
Before the path forward, let me close the loop back to that real robot — because the same lesson shows up
on hardware. This is a Unitree G1 humanoid with dexterous hands, learning the chemistry-pouring task I
mentioned at the start: pick up a tube, pour it into a beaker. Data is human teleoperation in a headset —
each episode is minutes long, so the whole dataset is tiny, 11 to 26 episodes. Here's the prescribed-data
moment: trained on the *full* 26-episode pool, the diffusion policy mode-collapsed and froze mid-episode,
because the demos came from two inconsistent motion styles plus outliers. So instead of dumping all the
data in, we clustered the demonstrations — PCA plus pairwise trajectory MSE — and kept only the tight,
consistent 8-episode cluster: a 65% cut in frames. The policy then converged cleanly to a single mode, no
freezing. The honest caveat, and it matters given everything I said about rigor: this is qualitative plus
training-loss, not yet a controlled rollout-success comparison — that's exactly the eval my path-forward
calls for. But the headline is the *same* as the simulator study, reached through a different selection
axis — demonstration consistency instead of failure targeting: which demonstrations you train on beats how
many. Prescribed data isn't a sim artifact; it's already earning its keep on a real humanoid.

**[Slide 22 — make it work]**
Where this goes, by leverage. One, the direct implication of H5: re-aim the gradient at what
the action loss encodes — grasp-success, motion-phase, per-scene — not object category; and I
have a bar, best-mode AUC above 0.7 before you bother. Two, the concrete next method,
RC-LESS: stack the fixes that address orthogonal defects — retention-constrained,
coverage-aware, Adam-preconditioned, whitened — plus a per-category floor that guarantees it
never does worse than core while giving it a shot at within-category structure core can't see.
Three, the deepest lesson from the failures: gate on *success, not loss* — a cheap 2k-step
mini-fine-tune that rejects the MSE-good, success-bad selections. And an honest ceiling: this
may be a data-coverage limit, not an algorithm limit — when the valuable data is outside the
failure region, only collecting *different* demos beats core. Which loops back to prescribed
data.

**[Slide 23 — real robots]**
Two threads. Evidence: n=300 can't resolve 1–4 point gaps — the power analysis says so — so a
paired-seed design, ~16k rollouts, multiple tasks and seeds, and fix the eval hang with a
per-rollout timeout. Impact: take the whole prescription loop to the real-robot pouring task
that motivated it, and use the validated expert-teacher loop — pi0/GR00T generate trainable
data on demand, no human teleoperation — so once I can prescribe a region I can actually fill
it. Prescribe, collect, retrain, re-evaluate — the loop closes.

**[Slide 24 — takeaways]**
Three takeaways, in sequence. One, the applied win: prescribed data works — when concentrated.
Densifying top failure categories beats random and beats coverage; the mechanism is surgical,
depth without disturbing the shared skill. Two, the transferable lesson: selection only works
when the training gradient encodes the target — a trivial heuristic beat sophisticated
influence because object identity is a nuisance to the action loss, and the best single-mode
AUC predicts this before you train. Three, honestly: this is a validated diagnostic and
principle, not a new state-of-the-art selector — plus one genuine win in whitening and an
honest data-coverage ceiling. The through-line for anyone doing data-efficient learning:
don't just add more data — prescribe the right data, and first check your training signal can
even see it. Thank you — I'm happy to take questions.
