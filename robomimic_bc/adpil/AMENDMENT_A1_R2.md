# ADPIL Amendment A1-R2 (revision 2 — v0.2 candidate, NOT frozen)

Date: 2026-08-31. Supersedes AMENDMENT_A1_DRAFT.md (preserved verbatim; both remain
in the hash ledger). Trigger: owner review of 2026-08-31 returned eight blocking
findings and scope corrections. Each R2 clause names the finding it answers.
Findings were verified against the artifacts before revision; two are resolved
jointly (6+8) with the mechanism stated.

## R2.1 Attainability gate scoped to efficacy claims (finding 1)

The 2x-floor rule applies only to configurations supporting POSITIVE efficacy
claims. The dial becomes:

- **D0 (natural, flat measured costs):** refusal/null calibration — EXEMPT from the
  gate. Pre-registered interpretation: correct selector behavior is refusal. An
  active win at D0 with interval excluding zero is NOT dismissed as noise; it is
  evidence against the ceiling model itself and triggers a model-falsification
  investigation, not an efficacy claim.
- **D1 (moderate cost tilt, ~1.5x floor):** EXPLORATORY dose-response point.
  Excluded from efficacy claims; labeled exploratory in every figure.
- **D2 (strong cost tilt, >=2x floor by design, target 3x):** the sole
  confirmatory configuration.

## R2.2 Spot-check raised to 12 paired histories (finding 2)

The oracle spot-check uses at least 12 paired histories (11 df), the same
inferential standard as §14.1. It is the only result that can open the tournament
gate. Nothing inferential is claimed from fewer than 12 histories anywhere in the
program. (The three Stage-1 rehearsal histories remain SMOKE_NOT_EVIDENCE.)

## R2.3 Natural-Can verdict re-scoped (finding 3; scope correction 1)

Stage 0 supports exactly: *the existing four-zone additive response model, with
parameters measured on mixed learners and recipes (MLP slopes, Square cross-zone
ratio, MLP draw variance, BC-RNN last-checkpoint seed variance), predicts no
detectable natural-Can allocation opportunity.* The binding natural-Can decision
is deferred until surface-v2 (R2.4) refits the model under BC-RNN v2 with paired
draws. Per v0.1 §10.5, the ceiling is model-conditional: allocation structure
outside the four-zone partition (e.g., in the K=8 visual clustering) is untested.

## R2.4 Response surface: paired multi-draw design (finding 4)

- Surface-v1 (the 48-run block launched 2026-08-31): its slope outputs are
  DOWNGRADED to pilot status by surface/PREREG_CORRECTION_1.md, written before any
  result value was visible. Its sigma_seed (32 df, seeds within fixed cells — the
  correct design for that parameter) and sigma_draw (5 independent vd_N80 draws)
  retain registered status.
- The binding surface comes from **surface-v2** (surface/SURFACE_V2_DESIGN.md):
  12 independently drawn base datasets per principal contrast, paired add-vs-base
  construction within draw, one seed per draw nested inside it (identical seed
  within a pair, never reused across draws), targeted zones rotated by frozen
  schedule, 11 df per contrast, draw-vs-seed separation by within-draw pairing,
  plus nested-budget cells for cross-budget draw covariance (R2.8). Masks are
  constructed by a frozen, hashed script after the surface-v1 block releases the
  HDF5. Not launched until this amendment and that design are approved.

## R2.5 The dial manipulates cost only (finding 5)

D0, D1, D2 share the FIXED natural exam distribution and differ only in the
designed per-zone acquisition-cost model. Exam-prevalence tilt is removed from D2
and becomes a separate factorial secondary stress test (P1), run only if D2
confirms, under its own pre-registered analysis.

## R2.6 Scoring/reporting merge retracted for confirmation (finding 6)

The A1.8 merge claim ("pairing keeps it symmetric") was wrong: active acquisition
is adapted to its scoring seeds, broad acquisition is not, so same-seed evaluation
can reward seed-specific repair — a directional bias favoring the active arm.
Resolution:
- Confirmatory analyses restore v0.1 §7 fresh reporting seeds (three new seeds per
  acquired dataset per round). The fresh-seed estimate is the confirmatory number.
- The scoring-seed estimate is reported alongside as a pre-registered adaptivity
  diagnostic (their difference measures seed-specific repair).
- Merging remains permitted only in the exploratory tournament.

## R2.7 Failure-probe specification (finding 7)

Replaces A1.4/A1.6: six round-disjoint 30-scene query banks (180 scenes total)
drawn from the unlabeled codebook bank, disjoint from every exam, frozen and
hashed in the task manifest. Stratified by natural cluster prevalence with a
minimum of one scene per cluster; F_hat_c estimated with empirical-Bayes shrinkage
toward the global failure rate (strength alpha=2, matching §9.1 smoothing). No
scene is reused across rounds, so the selector cannot adapt to a fixed probe set.
All probe rollouts are charged to the active arm's cost ledger per §13.

## R2.8 AULC covariance (finding 8, resolved jointly with R2.6)

Cross-budget SEED independence is restored by construction: fresh reporting seeds
per round (R2.6) — it was the retracted merge that would have broken it. Cross-
budget DRAW covariance remains bracketed at rho in [0.5, 1] in the floor and is
measured by surface-v2's nested-budget cells (six N=40-within-N=80 nested draw
pairs). The floor is recomputed with measured inputs after surface-v2;
ceiling_floor.py gains a measured-rho variant before any confirmatory hash.

## R2.9 Reconciliation: the ~4 pp program figure vs the 0.39 pp Stage-0 ceiling
(scope correction 2)

Both reproduce; they answer different questions (Q17_CEILING_AUDIT.md,
2026-08-26):
- Q17's C(q) peak of ~4.0 pp is a PER-ROUND (B=24) ceiling against a
  q-proportional diverse comparator, and requires a D0 STARVED in the high-mass
  region plus measured response curvature; its linear core is C_lin = beta·B·(q_max
  − Σq²) ≤ +2.0 pp, and it collapses to ~0 at q_max=0.85. Q17's own balanced-D0,
  measured-q row is **+0.58 pp** — the same order as Stage 0's +0.39 pp endpoint
  (differences: Q17 diagonal FE slope beta=0.45 vs Stage-0 own/cross split
  beta_diff=0.30; comparator b_div(q) vs pool-proportional broad; per-round vs
  whole-trajectory dose).
- **Open conflict, recorded for surface-v2 to resolve:** Q17's bin model estimates
  the n=0 occupancy bin at −11.98 pp (large returns to filling an empty region),
  while the surface-v1 pilot cell contrast estimated the starved own-zone slope at
  0.44x the balanced slope (small returns). These come from different models over
  overlapping MLP data. Surface-v2's paired starved cells measure this directly
  under BC-RNN v2. Until then, neither the starvation lever nor its absence is a
  supported claim.

## R2.10 Remaining scope corrections

- D1/D2 are **calibrated synthetic positive controls**. Any natural SO-101 claim
  requires independently measured, prospectively frozen collection costs.
- The "no competitor, FSC included" sentence is struck. Competitive positioning
  waits for a dedicated literature audit.
- **History-generation randomization, defined:** each history h gets an
  independent RNG stream from the seed ledger; the initial 20 demos are a uniform
  draw without replacement; the broad arm's purchase stream and all seed triplets
  derive from the same ledger. Sign-flip validity: under the sharp null (the
  acquisition policy does not affect the learning-curve distribution), arm labels
  within a history are exchangeable, and the per-history label randomization is
  independent across histories by construction, so the paired sign-flip test keeps
  its level under overlapping histories; cross-history outcome dependence affects
  power and scope (claims are pool-conditional), not type-I error. This argument
  is part of the frozen analysis plan.
- **Futility rule:** binding, futility-only (no early efficacy stop). A
  futility-only stop cannot inflate type-I error of the final test; it spends
  power, which the 24-history confirmation absorbs.

## Revised pricing (BC-RNN at 0.5–1.0 GPU-h/run; exact rate from surface-v1)

| block | runs | note |
|---|---|---|
| surface-v2 | 72 | 12 paired draws × (balanced 24 + starved 36 + nested 12) |
| D2 oracle spot-check | ~504 | 2 arms × 12 histories × 7 budgets × 3 fresh seeds |
| tournament (D1+D2, merged seeds) | ~504 | 4 arms × 6 histories × 7 × 3 |
| confirmation (D2, 24 histories) | ~1,500 | active 3 scoring + 3 reporting, broad 3 reporting |
| total | ~2,600 | 55–110 GPU-days ≈ 27–55 calendar days on 2 GPUs |

Roughly double the A1 estimate — the cost of restoring fresh reporting seeds and
the 12-history spot-check. Staging caps the expected cost: surface-v2 can kill the
natural-Can question, the spot-check can kill the tournament, futility can kill
the confirmation.

## Status

NOT frozen. Blocks awaiting owner approval: this amendment, SURFACE_V2_DESIGN.md,
and the D1/D2 cost models (to be drafted only after surface-v2 fixes the response
parameters). The running surface-v1 block continues; its role is now pilot +
noise-parameter measurement per PREREG_CORRECTION_1.md.
