# PRE-REGISTRATION: Square heterogeneity confirmation (2026-08-26, ~07:20)
Written AFTER the 33-run screen was read (screen verdict: binned Wald p=0.0006 PASSED,
slope bar FAILED at 1.95x < 2x for yhi_yawhi; placebo clean at ratio 0.03) and BEFORE any
confirmation run exists. Estimator note recorded honestly: the screen script's per-region
slope fit was degenerate (run-FE saturation); the corrected estimator is the n_r x region
interaction in the joint 132-obs model, run-clustered — this is an analysis-code fix
implementing the registered intent, not a criteria change; both outputs are hashed.

## Confirmation set
Seeds 3, 4, 5 x the same 11 arms (masks unchanged) = 33 runs. No new masks, no new regions.

## Pre-registered read (frozen now, region NAMED now)
Pooled analysis over all 6 seeds (66 runs, 264 obs), corrected estimator:
PASS requires ALL of:
 1. yhi_yawhi slope > 2x the pooled slope, clustered t >= 2   [region named in advance]
 2. continuous equal-slopes Wald (3 df) p < 0.05
 3. placebo DiD ratio stays < 0.25
PASS -> Square hosts Gate 2 / GP calibration (heterogeneous response = allocation can pay);
FAIL -> heterogeneity is declared unresolved-at-this-budget; the program's remaining lever
is heterogeneous COSTS (SO-101). Either way the two-domain response-curve result
(Can slope +0.657 confirmed; Square pooled +0.430, t 4.65) stands.
