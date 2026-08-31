# ADPIL Amendment A1 (DRAFT — v0.2 candidate, awaiting owner review)

Date: 2026-08-30. Trigger: Stage 0 (stage0/STAGE0_REPORT.md) fired the v0.1 §10.5
affordability clause, and the design-ceiling computation shows the natural-Can
oracle gate cannot pass for a true reason (ceiling +0.2 pp ΔAULC vs floor 3.9–4.9 pp
at 12 histories). Per §19, this amendment states what changes and why. No results
of any confirmatory contrast were visible when it was drafted; the quantities used
are noise parameters and prior-experiment measurements.

## A1.1 New §14.2a — design ceiling before detection floor (binding)

Before any evidence-producing stage, compute BOTH the detection floor and the design
ceiling from the frozen response model, prevalence, dose caps, and cost model. If
ceiling < 2× floor, the stage may not run in that configuration. The Stage-0
computation (ceiling_floor.py, hashed) is the reference implementation.
Consequence, already evaluated: natural-Can Stage 2 is not runnable.

## A1.2 Manipulated heterogeneity: the study becomes a three-point dial

The scientific question shifts from "does natural Can contain allocation value"
(measured answer: no, bounded at +0.4 pp) to "can a deployable selector capture
allocation value where it exists, and refuse where it does not." Three frozen
configurations per task, identical machinery:

- **D0 (placebo / natural):** flat measured costs (1.0–1.2×), natural exam.
  Predicted advantage ≈ 0. The pre-registered CORRECT selector behavior is refusal
  (§11.1); a selector that activates and wins here is winning on noise and the
  configuration doubles as the placebo arm.
- **D1 (moderate cost tilt):** designed per-zone acquisition-cost model with spread
  calibrated so the predicted active advantage in cost-normalized AULC ≈ 1.5× floor.
- **D2 (strong tilt):** cost spread + exam-prevalence tilt calibrated to ≈ 3× floor.

Cost models are frozen in the task manifest before any run; predictions come from
the A1.3 response surface and are recorded pre-hoc. The deliverable is the dial
curve (advantage vs designed heterogeneity) with the refusal margin tested at D0 —
the attainability-gate contribution no competitor (FSC included) has.

## A1.3 Stage 2 replaced: analytic gate on a BC-RNN response surface

The counterfactual-training oracles (greedy ~2,650 trainings, beam ~8,650) are
deleted. Replacement, in order:

1. **BC-RNN response-surface refresh (~48 runs ≈ 24 GPU-h):** balanced-D0 /
   add-zone / starved-D0 / starved-add cells, 3 seeds, v2 recipe. Yields
   beta_own, beta_cross, concavity, AND the missing BC-RNN v2-recipe seed sd
   (fixes A1.9) in one block.
2. **Analytic oracle:** best allocation under the fitted surface + frozen cost
   model, uncertainty propagated by parametric bootstrap over the surface fit.
   This is the design-ceiling reference line in all figures.
3. **Empirical spot-check (~60 runs):** ONE oracle-allocated arm vs paired broad
   at D2 only, 6 paired histories — verifies the analytic prediction within its
   interval before the tournament is authorized. Gate: measured advantage
   consistent with prediction AND interval excluding zero.

## A1.4 Query-cost stance (fixes the mechanical predetermination)

Deployable selectors receive a frozen **inspection budget: 30 scored rollouts per
round** (hardware-plausible), not all-candidate scoring (~510 rollouts/round, which
under the §13 ledger made every rollout-based arm lose by construction and the
rollout-free coverage arm win by default). Persistent-failure rates are estimated
cluster-level from the budgeted sample. Full-cost AULC stays primary; the 30-rollout
cost is charged honestly to the active arm.

## A1.5 Clustering: natural k-means, not balanced

v0.1 §9.1 balanced clustering forces pi_c ≈ 1/K, making the prevalence factor in the
§10.4 score a constant. Natural (unbalanced) k-means with the frozen encoder; the
K=16 sensitivity analysis unchanged.

## A1.6 F̂_c estimation: frozen probe bank, not the shrinking candidate pool

Buying failures depletes them from the candidate bank, so a bank-estimated failure
rate falls mechanically and mimics learning. F̂_c is estimated on a frozen 40-scene
probe bank disjoint from pool and exams, within the A1.4 rollout budget.

## A1.7 Honest-unit language

Histories drawn from a 200-demo pool overlap (~40% of the pool each); they are
exchangeable draws of the acquisition process on THIS pool, and all claims are
pool-conditional. The word "independent" is removed from §3. Can-MH (300 demos)
becomes the pre-registered robustness pool.

## A1.8 Compute-scale fixes

- Scoring and reporting models merged: the 3 per-round scoring models (which never
  touch exam data) are the reporting models. Halves training count. Caveat recorded:
  selection luck and evaluation luck become correlated within an arm; pairing keeps
  it symmetric across arms.
- Confirmation n raised to 24 paired histories × 3 seeds (floor 2.6–3.3 pp), priced:
  2 arms × 24 histories × 7 rounds × 3 models ≈ 1,000 RNN runs ≈ 500 GPU-h ≈ 10
  days on 2 GPUs — affordable because A1.3 freed the oracle budget.
- Futility stop: after 12 histories, if the one-sided 95% upper bound of the D2
  advantage < floor, stop the branch.
- Revised total program (surface refresh + spot-check + tournament at D1/D2 +
  confirmation + Square transfer): ≈ 2,600 RNN runs ≈ 1,300 GPU-h ≈ 27 days on 2
  GPUs — vs 4–9 GPU-months for v0.1 Stage 2 alone.

## A1.9 Floor inputs

The §14.2 floor must use BC-RNN v2-recipe variance. Interim value: 6.27 pp seed sd
(CAP block, last-checkpoint, 15 df) — conservative. Replaced by the A1.3 refresh
measurement before any confirmatory hash. Draw-sd for BC-RNN measured from the same
block's paired draws.

## A1.10 Claim 2.2 ablation or deletion

"Persistent failure beats single-model uncertainty" is currently not isolated (the
two arms differ in three design factors). Either add one ablation arm (persistent
failure at candidate level, cost-blind) in the tournament, or strike the claim.

## Effect on stages

- Stage 0: complete (this amendment is its output).
- Stage 1 rehearsal: unchanged, now also prices BC-RNN runtime exactly; explicitly
  authorized to measure noise parameters (variance is not an effect size).
- Stage 2: replaced by A1.3. Stages 3–6: machinery unchanged, run at D1/D2 with D0
  as placebo; Square transfer gate now uses the Square response surface.

## Hashes

To be computed at freeze: this amendment, ceiling_floor.py,
ceiling_floor_result.json, the D1/D2 cost models, and the A1.3 cell list.
