# Postmortem: Why the VALUE-influence pi0 selector lost (0.503, worst arm)

All numbers recomputed from `weakregion/eval_*/weakregion.json`, `weakregion/value_core.json`,
`weakregion/arms*.json`, and the MG `episodes.jsonl` pool. n=300 per arm, scene-paired seed100000.
(Multi-agent investigation: 7 hypotheses investigated, 6 adversarially verified, then synthesized.)

## 1. Why it failed

Overall success: baseline **0.580**, core **0.593** (win), random **0.563**, coverage **0.517**,
**value 0.503** (worst). Split on core's 10 failure categories vs the rest:

| arm | targeted (n≈36) | non-targeted (n≈264) | ΔNT vs baseline |
|---|---|---|---|
| baseline | 0.333 | 0.617 | — |
| core | 0.432 | 0.616 | **−0.001** |
| coverage | 0.429 | 0.528 | −0.089 |
| **value** | 0.361 | **0.523** | **−0.094** |
| random | 0.282 | 0.605 | −0.011 |

- **The entire loss lives in the non-targeted ~88% of the eval.** Value's targeted-10 is actually
  *fine* (0.361 > baseline 0.333). 100% of its deficit is broad forgetting on the common majority.
- The damage is at **grasp**: value has +25 extra `fail_no_grasp` vs baseline (128 vs 103), the most
  total failures (149). Grasp is the phase **shared across every object category**, which is why one
  bad nudge regresses *globally*, not in a few classes.
- Training was healthy (loss 0.230→0.071, normal grad norm, identical to other arms) → this is a
  **data-selection failure, not an optimization failure**.
- **Significance (honesty):** core>value p=**0.013**, baseline>value p=**0.048**, random>value
  p=0.083 (**NS**). Value is *significantly* worse than core, marginally worse than baseline, **not
  significantly worse than random**. "Worst arm" is the point estimate; the solid claim is value<core.

### Ranked root causes

**PRIMARY 1 — Plain-mean `g_val` is common-mode-dominated → the score ranks demos by gradient
*genericness*, not value to any target.** **97.25%** of all 9172 candidates have *positive* cosine to
`g_val` (a discriminative direction would split ~50/50); the top-200 sit at **+2.54σ** of the score
distribution. The team's own smoke already showed plain `g_val` gives hard-vs-easy AUC **0.35** (below
chance) vs **0.63** for contrast — the value arm shipped the plain form anyway.

**PRIMARY 2 — the actual value-vs-random discriminator: skew toward a handle/elongated-grasp cluster.**
Value and random are nearly identical in category *density* (62 vs 67 cats, both median 3) and
base-redundancy (avg base count 6.02 vs 6.20), yet only value forgets. The *one* axis that separates
them: **handle/grasp-cluster fraction 0.360 (value) vs 0.205 (random)** — top picks tongs(14),
dish_brush(12), mug(11), rolling_pin(8), straw, at 3.5–5× pool frequency. The over-emphasized
elongated/handled-grasp mode **displaces the base's well-fit majority grasp behavior** → global grasp
regression.

**SECONDARY — magnitude-blindness selects low-information demos.** Score is cosine over *unit-normalized*
gradients (magnitude discarded). The vsmoke showed top-cosine demos carry the *smallest* raw gradients
(raw 1-step update: top −0.000018 vs random −0.000042). Suggestive (n=5 groups, no CI), not load-bearing
on its own, but explains *why* the selected set is weak/redundant.

### Refuted (do NOT use these explanations)
- **Literal category-skew / distribution-preservation:** core is the *biggest* category perturbation
  (10 cats, 88/200 into base-absent holes) yet *wins*; value perturbs the histogram *least* (KL 0.237 <
  core 0.481) yet forgets *most*. Coverage of the histogram does not predict ranking.
- **Per-target density/concentration:** refuted by value≈random density.
- **"Redundant easy categories":** value's picks are *harder* on average than random's (top pick tongs
  has 0.00 baseline success); random picked easier cats yet didn't forget.
- **Overfitting / unfair 19999 checkpoint:** refuted — train curves identical across arms.

## 2. Is the user's intuition right?

**"If start/target places differ, averaging all gradients into one `g_val` is bad."** → **Yes, largely
correct — it's the primary mechanical cause — with one refinement.**

Averaging 313 heterogeneous, pose-varying unit tugs: the **target-specific** components (roughly
orthogonal across diverse objects/poses) partially **cancel**, while the **shared "generic arm motion"**
component is in-phase and **survives**. So the mean collapses onto the common mode (97% positive cosine,
AUC 0.35), and ranking by cosine to it rewards *genericness*.

**Refinement:** genericness *alone* can't separate value from random (random is *more* spread/generic and
was harmless). The operative lever is **actively maximizing** the common-mode cosine to the **extreme
tail (+2.54σ)**, which loads the budget onto the over-represented handle/grasp cluster. Precise
statement: *averaging heterogeneous-pose gradients yields a common-mode direction; ranking by alignment
to it concentrates the budget on one dominant grasp mode that displaces the base's well-fit majority
behavior.*

## 3. How to improve it (prioritized; tied to a confirmed cause)

1. **[BEST NEXT — cheap to prototype] Don't collapse to one mean.** Cluster the D_val unit tugs into m
   directions (by object/pose, or k-means on the unit vectors); score each candidate by its **max** (or
   top-k) cosine over the m target directions, and select greedily for **coverage** (facility-location /
   submodular). Directly implements the user's intuition: reward demos that strongly help *some specific*
   target, not weakly hug the average. *(Needs the per-D_val-demo tugs retained — currently only the mean
   is stored.)*
2. **[cheap] Contrast `g_val` even for the value objective** (`g_val = normalize(mean(D_val) −
   mean(random_pool))`) — the form that smoke-passed 0.63 vs 0.35. One-flag change; keeps a balanced D_val.
3. **[cheap] Magnitude-aware scoring** — drop the per-demo unit normalization (`<g_raw(z), g_val_unit>`)
   so informative (higher-loss) demos rank up; re-run the raw-update vsmoke and require top-200 to *beat*
   random before any fine-tune.
4. **[medium] Adam-preconditioned gradient features (canonical LESS).** Score on the Adam update
   `−η·m/(√v+ε)`, not raw grads — down-weights the low-curvature common mode the optimizer already handles.
5. **[expensive] Align the target with task SUCCESS, not flow-matching loss** (weight D_val by per-category
   failure rate, or TracIn against rollout outcomes). The winning *failure* arm worked precisely because
   its D_val was the failure categories.
6. **[cheap guardrail] Diversity / coverage caps** (cap per-category picks; coverage over *failing*
   targets). But note the winning recipe is **depth-on-holes** (core: ~10 failing cats × ~20 demos), so
   the right objective is *uniform-over-failing-cats with a per-target density floor*, not just spread.
7. **[cheap-ish de-risk] Validate with mini-fine-tunes** — short 2k-step LoRA fits on a few candidate
   200-subsets, gated on keeping non-targeted ≥~0.60, before committing a 20k-step arm.

## 4. Bottom line

The value arm lost entirely through **broad forgetting on the ~88% non-targeted majority** (non-targeted
0.523 vs baseline 0.617; +25 no-grasp failures; its own targeted-10 was fine). The proximate cause is the
**selection objective, not the optimizer or the histogram**: averaging 313 heterogeneous unit gradients
into one *plain* `g_val` collapses onto the common-mode "generic pick-place" direction (97% positive
cosine, AUC 0.35), so ranking by cosine to it pushed the budget to the +2.54σ tail onto a single
over-represented handle/grasp cluster (0.36 vs random 0.21) — the only axis distinguishing value from the
harmless random arm. The user's intuition is essentially right; the fixes — per-target/clustered (or
contrast) influence, magnitude-aware / Adam-preconditioned features, success-aligned targets, and
depth-on-failing-targets coverage — are exactly what separated the winning core arm, and should be
validated with cheap mini-fine-tunes before any further 20k-step run.
