# PRE-REGISTRATION: Can-PH condition-region prescription benchmark
Frozen 2026-08-25, BEFORE any policy was evaluated on any fixed start.

## Objective
Test whether region-targeted data collection beats diverse collection on a
deployment-weighted objective, and whether actively-collected REMOVAL observations
improve allocation decisions (calibration) beyond dataset accounting. Task:
robosuite PickPlaceCan, robomimic Can-PH (200 single-operator demos -> quality
matched across regions by construction). Policy: robomimic BC (MLP 1024^2),
300 epochs, batch 100, EQUAL-PER-TRAJECTORY weighting ALWAYS
(justification: ../prescribe/EQ_CONTROL_VERDICT.md).

## Region grid (frozen)
2x2 median split of the can's initial (x, y) from the 200 demos:
qx=0.0986, qy=-0.2600; labels xlo_ylo, xlo_yhi, xhi_ylo, xhi_yhi; 50 demos each.
The same grid stratifies evaluation starts. Never redrawn.

## Evaluation sets (frozen; generated from fresh env resets, not demos)
- E_probe: 25 unique starts/region (100 total). Uses: pilot D0-size decision,
  policy diagnostics, ALL calibration observations, choosing allocations b,
  estimating J_hat_r for the P(fail) baseline.
- E_test (SEALED): 50 different starts/region (200 total). Uses: Gate 1,
  held-out allocation regret, final reporting. Run scripts write
  fixed_eval_test.json to disk but print only probe numbers; no analysis reads
  E_test results except the pre-registered gate analyses below. No region,
  threshold, calibration direction, or stopping rule changes after E_test is inspected.
- Assignment rule: per region, seeded reset stream order 1-25 -> probe, 26-75 -> test.
- Determinism gate PASSED: reset_to() restores state exactly; 20-step zero-action
  replay is bitwise identical => unique starts only, no repeats.
- Deployment weights q_r = natural reset shares from 2000 fresh resets
  (deploy_shares.json). Objective: J_deploy = sum_r q_r J_r.

## Pilot (seeds 100, 101 — never reused)
D0-only balanced at N in {40, 60, 80} (masks pilot_D0_N, crc32-seeded draws).
Freeze the smallest N with mean probe J_deploy in [0.40, 0.75] and no floor-dead
seed (each seed's J_deploy > 0.10). If all N saturate (>0.75): Can-PH low-dim BC
is unsuitable; STOP before building profiles.

## Profiles at frozen N (screen set; crc32-seeded draws)
- balanced: N/4 per region
- starved_r (r rotated over all 4 regions): starved region max(2, round(N/20)),
  remainder split evenly
- two_starved (confirmatory phase only): one adjacent pair, one diagonal pair
Screen uses balanced + 2 rotated starved profiles; remaining rotations + two_starved
are confirmatory.

## Arms per profile (B = 24; wells = region demos not in that profile's D0)
Forward F(b) = J(D0+A_b) - J(D0):
- 4 one-region allocations (b = all 24 from one region)
- b_div: b_r = round(B*q_r) (safe default)
- b_cov: water-filling argmin_{sum b_r=B} sum_r ((n_r+b_r)/(N+B) - q_r)^2
- b_pfail: b_r proportional to q_r*(1 - J_hat_r^probe)  [MANDATORY baseline]
- null pair: two INDEPENDENT b_div draws (different crc32 draw tags) -> paired
  null-noise scale
Removal R(b) = J(D0) - J(D0 - H_b): H_b = D0's demos in region b, all 4 regions
(on starved profiles the starved-region removal is tiny BY CONSTRUCTION; it stays
in the calibration set with its before/after counts — the response model consumes
counts, so asymmetry is information, not bias).

## MDE rule (computed BEFORE unblinding targeted arms on E_test; timestamped)
From the independent-diverse null pairs (all profiles, both seeds), estimate the
sd of paired J_deploy differences on E_test, clustered at the profile level.
X = smallest true effect with 80% power at one-sided alpha=0.05 given that sd and
the screen's n. Declared in MDE.json with a timestamp before any targeted-arm
E_test row is read.

## Gates
1. OPPORTUNITY (Gate 1, on E_test, deployment-weighted): some condition allocation
   beats b_div by > X on starved profiles, replicated in sign across both screen
   seeds. Fail after confirmation seeds => STOP (no Square, no pi0 port).
2. CALIBRATION BENEFIT (Gate 2, three separate claims, regret on held-out forward
   additions on E_test; calibration observations from E_probe only):
   (a) Regret(calibrated response model) < Regret(same meta-model, zero calibration)
   (b) Regret(calibrated) < Regret(b_cov)   [beats dataset accounting]
   (c) Regret(calibrated) < Regret(b_div)   [beats safe default]
   Raw R(b)-F(b) correlation is reported as a DIAGNOSTIC only.
3. DECISION-RELEVANCE: the chosen allocation differs across profiles (the
   instrument responds to D0 composition, not a constant answer).
4. NEUTRAL-BEHAVIOR: on the balanced profile, one-region allocations do not beat
   b_div by > X (no free lunch where coverage is already met).
5. REPLICATION: sign consistency across profile rotations; confirmatory claims
   need the third seed (seed 2) and the full rotation set. The 2-seed screen
   (seeds 0, 1) is EXPLORATORY and labeled as such in any write-up.

## Noise / inference discipline
Paired seeds throughout; profile-level clustering for all inference; no
per-comparison p-values on single seeds. Eval is deterministic given the policy,
so all variance is training-side (draw + seed) — exactly what pairing cancels.
