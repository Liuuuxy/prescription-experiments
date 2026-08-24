# Q0: Gradient-Encoding Gate for the Tall-Vessel Region (2026-07-31)

**Question.** Does the π₀ LoRA flow-matching gradient, at the exact checkpoint every bandit
pull fine-tunes from (`pi0_ppc2sink_pi0base/pi0_v1/19999`), encode membership in the
`tall_vessel_grasp_fail` region (the behavior-descriptor cluster the bandit's worst-teaching
arm draws from)? Per the gradient-encoding principle (memory
`influence-gated-by-gradient-encoding`), any gradient-based teachability/influence signal
for this region can only work if it does. Bar from prior validations: best-single-SVD-mode
AUC ≳ 0.7 workable; ~0.55 weak/nuisance-like.

**Setup.** `grad_sketches.py`: per-demo LoRA gradients (~50M dims, `mode_grad=lora`,
`real_dim=12`, K=8 frames), sparse-JL sketched to 2048 dims (fidelity self-check: true-cos vs
sketch-cos corr **0.942** on full 50M-dim grads). Sets: 120 in-region vs 120 out-of-region
(stratified over the other arms; assignment via the race's own `wells.assign_regions`, frozen
arms.yaml + map), all eight 200-demo round-3/4 treatment draws, 120 D0 episodes.
Analysis: `analyze_gate.py` (selftest: planted-signal 0.99, shuffle 0.63). Report JSON:
`q0_gate_report.json`.

## Result: the gate FAILS

| statistic | value | control / bar |
|---|---|---|
| best-single-SVD-mode AUC | **0.577** | shuffle-null floor of the same statistic: mean **0.598**, max 0.622 (8 shuffles) |
| split-half contrast AUC (supervised) | **0.572** | shuffled labels 0.501, random direction 0.536 |
| whitened (top-10 modes removed) | 0.646 / 0.660 | still ≪ 0.7; k=25/50 degrade (0.53 / 0.35) — over-whitening collapse, same pattern as CIFAR |

The unsupervised statistic is **below its own multiple-comparisons null floor**; the
supervised direction barely separates held-out individuals. Region membership is at most
marginally present in the training gradient. This is measured on the *behavior descriptor*
(model-predicted difficulty/stage), so it closes the remaining hope that kinematic region
labels encode better than object category did (0.56 → ~0.60 in the prior work).

## The treatment sets are gradient-exchangeable

Scoring each 200-demo pull draw against the region contrast direction (fit on the full gate
sets — optimistic for every set equally):

| pull set | cos(region dir) | cos(common mode) | self-sim | grad-norm |
|---|---|---|---|---|
| tall_j3 / tall_j4 (100% in-region) | +0.024 / +0.021 | 0.200 / 0.199 | 0.045 / 0.044 | 3.29 / 3.33 |
| random_j3 / random_j4 (~23% in-region) | +0.019 / +0.010 | 0.206 / 0.205 | 0.046 / 0.045 | 3.20 / 3.45 |
| mid_j3 / mid_j4 | +0.010 / +0.006 | 0.203 / 0.202 | 0.044 / 0.045 | 2.83 / 3.01 |
| easy_j3 / easy_j4 | +0.014 / +0.016 | 0.201 / 0.201 | 0.044 / 0.044 | 3.06 / 3.02 |

The shared "generic pick-place" component (~0.20) is **~10× larger** than any region-specific
residual (≤0.024), and the tall-vs-random gap on the region direction is ~0.01. In gradient
space, a fully in-region draw is nearly indistinguishable from a random draw.

**Internal-validity check:** trained-on D0 demos have grad-norm **0.554** vs **3.136** for
unseen pool demos (6× absorption gap) — the pipeline detects real training structure, so the
flat region signal is an absence, not insensitivity. (Corollary: a D0-mean "retention
direction" at π₀ is a post-absorption residual and too weak to use; per-set cos(g_R) spread
0.000–0.014 — do not over-read.)

## Interpretation

This is the mechanism behind both behavioral anomalies of rounds 3–4:

1. **Tall arm never moved its own hard stratum** (−2.7pp, then 0.0 vs baseline) **while
   helping easy/mid** (+13.4/+4.4 in j4): its demos' gradients are ~pure common mode, so
   fine-tuning on them teaches "generic pick-place", which is precisely what they delivered.
2. **Random tied or beat selection** on the target stratum (+2.7/+7.3): if gradients are
   region-blind, all B=200 draws are approximately exchangeable at the loss level — data
   *composition* cannot matter at the region level, only data *quantity* / dose.

For the recipe framing: this is the **teachability-gate exhibit** — a few GPU-minutes of
gradient probing at π₀ predicts (and now explains) the null that cost multiple 13-GPU-hour
pulls to observe behaviorally. It also sharpens the ClusterUCB/LESS contrast: the LLM
playbook's cheap influence-proxy reward is exactly what fails here — the VLA action loss does
not encode the deployment-relevant distinction.

## Caveats

Single checkpoint (19999 — the branching point, the right one for "what fine-tuning sees at
start"); K=8 frames with single noise/time draws adds per-demo noise that attenuates all
cosines (but the 6× absorption gap and 0.20 common-mode cosines show sensitivity);
2048-d JL sketches (pairwise cos mae 0.016; set-mean contrasts average this down by ~√200);
region labels are themselves model-derived (frozen map, held-out AUC 0.629). Consistent with
three independent prior measurements (category best-mode 0.56; per-instance ceiling ~0.61;
whitening max 0.68).

## Next (Q2, needs round-5 artifacts + pull checkpoints)

- **Absorption dynamics:** recompute tall-draw grad norms along the tall_j3/j4 fine-tune
  trajectory (5000/10000/15000/19999) — did the model absorb the demos (norms → small) while
  the hard stratum stayed flat? That separates "didn't learn the demos" from "the demos
  don't contain the skill".
- **Predictive stat across 12 pulls:** with round 5, test whether any π₀-time gradient
  statistic of a draw predicts realized Δ (the design's logged-only selector ablation).
- Robustness: second checkpoint (basewarmup), K=16, category-direction cross-check.
