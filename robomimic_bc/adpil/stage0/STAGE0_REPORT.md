# ADPIL Stage 0 report — pool audit, runtime pricing, design ceiling vs detection floor

Date: 2026-08-30. Authorized by protocol ADPIL-POOL-001 v0.1 §20 (items 1, 5, and the
§10.5 oracle-affordability review). No evidence-producing runs were performed; every
number below is an audit fact or a computation over previously collected data.

## 1. Source-pool audit (§20.1) — PASS

All three candidate pools carry full simulator `states` and `model_file` per demo,
so exact initial-scene replay — the mechanism the hidden-pool query interface needs —
is supported everywhere.

| pool | file | demos | steps (min/med/max) | notes |
|---|---|---|---|---|
| Can-PH | prescribe_ph/can_ph_work.hdf5 | 200 | 82 / 115 / 151 | zone masks: exactly 50 demos per zone |
| Square-PH | prescribe_sq/square_ph_work.hdf5 | 200 | 107 / 150 / 236 | region masks present |
| Can-MH | prescribe/can_mh_work.hdf5 | 300 | 98 / 178 / 1050 | operator masks present |

Consequences for the v0.1 design:
- An acquisition history (20 initial + 60 purchased of 200) uses 40% of the pool, so
  12 "independent" histories overlap heavily pairwise. Inference is pool-conditional;
  the protocol must say so (Amendment A1.7).
- Max concentration into one Can zone is 50 demos minus what the initial 20 took
  (~45 in expectation) — the dose cap used in the ceiling computation below.

## 2. Runtime pricing (§20.5)

Measured from the vardecomp2 production block (60 BC-MLP runs of the v2 recipe —
500 epochs, checkpoint-every-50, best-of-checkpoint on a 50-scene probe — launched
2026-08-28 23:43, last log 2026-08-29 07:39, two GPUs):

- **BC-MLP v2 run (train + probe + sealed eval): ~16 GPU-minutes** (≈7.9 h wall × 2 GPUs / 60).
- BC-RNN (CAP block, 20 runs, last-checkpoint recipe): ~25–35 GPU-minutes/run
  (tail-block estimate; Stage 1 rehearsal must price this exactly).

Priced against the v0.1 Stage-2 oracle design at 0.5 GPU-h per BC-RNN run:

| stage | trainings | GPU-h | wall on 2 GPUs |
|---|---|---|---|
| greedy oracle gate (12 histories) | ~2,650 | ~1,300 | ~27 days |
| beam-4 oracle gate (12 histories) | ~8,650 | ~4,300 | ~90 days |
| tournament (5 arms × 6 histories) | ~1,260 | ~630 | ~13 days |
| confirmation (2 arms × 12) | ~1,000 | ~500 | ~10 days |

**Verdict: the §10.5 affordability clause fires.** The empirical-oracle stage as
specified (~5,600–11,300 trainings) costs 4–9 GPU-months before any deployable
selector exists. The protocol itself requires a redesigned, re-hashed oracle stage.

## 3. Design ceiling vs detection floor (script: ceiling_floor.py, result JSON hashed)

This is the check the project's MDE-before-unblinding rule requires and v0.1 omitted:
compute what the experiment COULD detect and what the treatment COULD deliver, before
running it.

### 3.1 Detection floor (what Stage 2/4 could detect)

Inputs, all measured:
- BC-RNN seed sd on the sealed exam: **6.27 pp** (pooled over 5 data conditions × 4
  seeds from the CAP block, 15 df — satisfies the ≥10 df rule). Caveat: last-checkpoint
  recipe; the v2 recipe's BC-RNN sd is unmeasured (Amendment A1.9).
- Draw sd (which demos a broad draw got): 3.6 pp (vardecomp2 v2, BC-MLP).
- ΔAULC = trapezoid over budgets 20..80, 3 reporting seeds per point, seed noise
  independent across rounds, draw noise bracketed at correlation ρ ∈ {0.5, 1} across
  nested budgets, arms additive (overlap between paired arms' purchases ignored —
  conservative).

**Floor (minimum detectable ΔAULC, 80% power, α=.05 paired t):**
- 12 histories, 3 seeds: **3.9–4.9 pp**
- 24 histories, 5 seeds: **2.5–3.2 pp** (best affordable configuration)

### 3.2 Design ceiling (what natural Can-PH can deliver)

Response model from confirmed measurements: adding a demo of zone z buys the zone
+0.657 pp/demo (C9, out-of-sample, t=5.66) and other zones +0.657/1.84 = +0.357
pp/demo (own/other response ratio 1.84, Square screen). Deployment prevalence from
2,000 natural resets: (27.6, 21.5, 26.1, 24.9)% — nearly uniform. Best possible
allocator = concentrate the 60 purchased demos on the highest-prevalence zone up to
the ~45-candidate cap, vs broad random matching pool shares.

- **Ceiling, natural Can: +0.39 pp final-budget, +0.20 pp ΔAULC.**
- Even forcing cross-zone transfer to zero (untrue, but the outer bound): +0.86 pp
  final, +0.43 pp ΔAULC.
- Histories needed to detect even that outer bound: **~1,271** (vs 12 planned).

**Ceiling is 10–25× below the floor. The natural-Can oracle gate cannot pass for a
true reason; it can only pass on noise.** Running it as designed would be a
months-long confirmation of arithmetic already available.

### 3.3 The three natural heterogeneity levers, each measured dead

1. **Prevalence tilt (designed exam):** even an exam putting 90% of its mass on one
   zone yields +3.9 pp ΔAULC = 0.8× the n=12 floor. The lever saturates below
   detectability because the response-difference coefficient (0.30 pp/demo) and the
   45-demo zone dose cap it. Prevalence design ALONE cannot rescue the study.
2. **Response heterogeneity:** zones respond alike (C9 + Square screen; the screen's
   own/other ratio 1.84 failed its pre-registered 2× gate).
3. **Concavity / starvation (new measurement, this Stage 0):** from the prescribe_ph
   raw screening scores (2–4 seeds/cell, BC-MLP, exploratory): own-zone add slope
   with a zone-starved initial dataset = +0.44 pp/demo vs +0.99 balanced — a ratio of
   **0.44, i.e. the starved slope is SMALLER, not larger**. One starved cell was
   negative (−0.46, n=2); cross-zone demos sometimes helped the starved zone more
   than own-zone demos (+1.42 vs +1.33 in the cleanest cell). No support for
   "initial-dataset holes create allocation opportunity" at this noise level.

### 3.4 What remains live

**Cost asymmetry** is the only allocation lever that is (a) not refuted by
measurement, (b) designable to any strength in sim, and (c) externally real —
SO-101 collection conditions genuinely differ in operator cost, and the Square
heterogeneity screen already concluded cost is where the allocation question moved.
Under the protocol's own cost-normalized estimand, a designed per-zone cost model
shifts the broad arm's cost curve directly; the predicted active-vs-broad advantage
scales with the cost spread and is computable in advance to sit at any multiple of
the floor. Natural (flat-cost) Can then serves as the placebo point where the
correct selector behavior is REFUSAL (§11.1) — turning the refusal mechanism from
a safety valve into a tested claim.

## 4. Stage-0 conclusions

1. Pools and replay interface: ready. Manifest fields for §18 can be filled.
2. v0.1 Stage 2 is dead as written, by the protocol's own §10.5 affordability
   clause AND by ceiling-vs-floor arithmetic (0.2 pp vs 3.9 pp).
3. The redesign must move the manipulated heterogeneity into the cost channel and
   replace the counterfactual-training oracle with an analytic gate on a measured
   response surface, refreshed on BC-RNN. Draft: ../AMENDMENT_A1_DRAFT.md.
4. Nothing in this report unblinds any future confirmatory contrast: every number
   is either an audit fact, a noise parameter, or a prior-experiment quantity.
