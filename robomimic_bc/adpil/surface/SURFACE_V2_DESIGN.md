# Surface-v2 design — paired multi-draw BC-RNN response surface (NOT LAUNCHED)

Date: 2026-08-31. Status: design for owner approval under Amendment A1-R2 (R2.4).
Purpose: the BINDING response surface for the analytic oracle, the recomputed
floor, and the final natural-Can verdict. Replaces the inferential role of the
surface-v1 slopes (downgraded to pilot by PREREG_CORRECTION_1.md).

## Design principles (owner findings 4 and 8)

Independent dataset draws are the experimental unit; training seeds are nested
inside draws and never treated as replicates of a causal intervention. Every
response contrast is a within-draw paired difference, which cancels draw and seed
main effects and leaves 11 df across the 12 pairs.

## Cells (72 runs, one seed per draw, seed s_d = 300+d, identical within a draw)

**C1 — balanced response (24 runs).** Draws d=1..12. Base B_d: 80 demos, 20 per
zone, uniform without replacement, fresh RNG stream per draw. Add A_d: B_d + 24
demos of zone z(d), where z(d) cycles the four zones (3 draws each). Feasible:
20+24=44 <= 50 per zone. Estimands: beta_own = mean paired diff on J_region[z(d)]
/ 24; beta_cross from the three untargeted zones of the same pairs.

**C2 — starved response (36 runs).** Draws d=13..24. Starved base S_d: 80 demos,
zone z(d) at n=0, 27/27/26 from the others (z(d) cycles all four zones, 3 draws
each — extends surface-v1's two starved zones to all four). Own-add: S_d + 24 of
z(d). Cross-add: S_d + 24 of zone z'(d) (rotated schedule). Estimands: starved
own/cross slopes; concavity ratio vs C1; resolves the R2.9 conflict (Q17 n=0 bin
−11.98 pp vs pilot ratio 0.44) under BC-RNN v2.

**C3 — cross-budget draw covariance (12 runs).** Draws d=25..30. Nested pair per
draw: N40_d (40 demos, 10/zone) and N80_d = N40_d + 40 (nested extension). Both
trained; the covariance of paired draw deviations across the two budgets feeds
the measured-rho floor variant (R2.8). 5 df — a bracket-informing measurement,
registered as such, never an MDE input on its own.

## Frozen procedures

- Mask construction: build_v2_masks.py (to be written and hashed BEFORE running),
  fixed RNG seed recorded in the manifest; masks appended to can_ph_work.hdf5 only
  after the surface-v1 pool exits (no concurrent writes to the shared file).
- Driver: run_surface.py unchanged (hash in ledger). Pools: pool_surface.py with
  new manifests.
- Analysis: analyze_surface_v2.py, written and hashed before unblinding. Paired
  t and sign-flip across 12 draws per contrast; betas with 95% CIs; concavity
  ratio with draw-level bootstrap CI; C3 covariance with its df caveat.

## What surface-v2 authorizes

- If the fitted four-zone model still predicts natural-Can ceiling < 2x the
  recomputed floor: the natural-Can branch is closed as a BINDING result (R2.3),
  and D-series cost models are drafted from the fitted betas.
- If not (e.g., the starved slope is large under BC-RNN): the natural-starved
  configuration re-enters as a candidate lever and the dial design is revisited
  before any cost model is frozen.

## Price

72 runs at the surface-v1 measured rate (expected 0.5–1.0 GPU-h/run): ~36–72
GPU-h, ~1–1.5 days on 2 GPUs.
