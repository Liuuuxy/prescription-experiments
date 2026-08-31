# PRE-REGISTRATION: Square-PH regional-response heterogeneity screen
Frozen 2026-08-26, BEFORE any Square policy is trained. Successor to the Can-PH benchmark,
whose Gate 1 was retracted (PREREG_CORRECTION_2.md) and whose corrected opportunity audit
(Q17_CEILING_AUDIT.md) showed q-concentration cannot create allocation opportunity under a
homogeneous linear response. This screen targets the one lever that can: RESPONSE
HETEROGENEITY. Advisor directive 2026-08-26: "the next positive benchmark should target
response heterogeneity, not manipulate q."

## Question
Does the per-region dose-response S_r(n_r) DIFFER across regions on Square-PH — i.e. is
there any task where the response matrix is not homogeneous-diagonal, so that allocation
can beat q-proportional diverse collection?

## Task and regions
robomimic Square-PH (200 single-operator demos; quality matched by construction). BC MLP
1024^2, 300 epochs, batch 100, EQUAL-PER-TRAJECTORY weighting always. Regions: 2x2 grid over
(nut initial y) x (nut initial yaw), dataset medians qy=0.1676, qyaw=0.084 rad; labels
ylo_yawlo, ylo_yawhi, yhi_yawlo, yhi_yawhi (counts 48/52/52/48). The nut's x position is NOT
a region axis (5mm total spread = placement jitter). Both axes are human-actionable
(near/far half of the placement zone; handle pointing left/right).

## Evaluation
Fresh-reset frozen sets, disjoint streams, validated exact/bitwise-deterministic restore:
E_probe 25/region (pilot decision, diagnostics), sealed E_test 50/region (all reported
effects). q_r = natural reset shares from 2000 resets. Unique starts only.

## Pilot (seeds 100/101; probe only)
Balanced D0 at N in {60, 80, 96} (96 = max with 24-demo wells). Rules, frozen:
- The HETEROGENEITY screen proceeds at the smallest N with mean probe J_deploy in
  [0.25, 0.75] and every seed > 0.10. (Dose-response is measurable below the allocation
  band; 0.25 is the floor at which per-region rates carry signal at 25 starts/region.)
- The C(q) OPPORTUNITY computation is additionally meaningful only if mean >= 0.40; if the
  frozen N sits in [0.25, 0.40) this is recorded and C(q) is reported as exploratory.
- All three sizes < 0.25 or floor-dead seeds at every size -> Square-PH is unsuitable at
  this budget; STOP (no threshold changes after seeing arms).

## Screen (33 runs, seeds 0/1/2 — replicates, NOT pairs: Can-PH C6 showed seed pairing
buys nothing in this domain; three seeds are for averaging)
One profile only (balanced D0 at frozen N). Arms per seed (11):
  D0 · add_<r> x4 (B=24, single region) · rm_<r> x4 · add_div1 · add_div2
(crc32-seeded draws; div1/div2 = independent q-proportional draws = the null pair.)

## Primary estimand and test (frozen)
Per-region success J_r on E_test, unit = (run, region), 132 observations.
Model M0: J_r ~ C(run) + C(region) + C(dose_bin);   dose bins [0, 1, N/4, N/4+24+1) i.e.
  {removed, base, augmented} — three doses by design.
Model M1: adds region x dose interaction.
HETEROGENEITY TEST: run-clustered Wald test of M1 vs M0 (12 - 2 interaction df). PASS =
p < 0.05 AND at least one region's own-dose slope differs from the pooled slope by more
than a factor of 2 with clustered t >= 2. Placebo discipline (Can-PH C9): every DiD
includes the untreated-region placebo; a "heterogeneity" driven by baseline region x seed
interaction (placebo DiD comparable in size) is a FAIL.
Secondary: pooled per-demo slope (the Can-PH replication read: positive, clustered t >= 2).

## Opportunity gate (advisor formulation, two thresholds SEPARATE)
C(q) = max_b q.[J(D0+b) - J(D0+b_div(q))], b_div(q) = Bq rebuilt per q (Q17 discipline),
brute force over all integer allocations, response from the fitted M1, uncertainty by
run-clustered bootstrap.
- delta (algorithmic, size-independent, frozen now) = 5.0pp: the minimum practically
  useful improvement for this program. The deployed method must abstain when
  UCB(C(q)) < delta.
- X (experimental) = computed per the C8 discipline (>= 10 df, both sd scales reported,
  refuse on >3x disagreement, lower-tail chi-square reported). The program declines a
  Gate-1-style efficacy experiment when UCB(C(q)) < X.
Changing seeds changes X, never delta.

## Multiplicity + verdicts
The heterogeneity test is ONE pre-registered test (not 18 comparisons). All verdict strings
in analysis scripts must be conditional on the data. MDE/variance discipline per C8 is
binding. Outcomes:
- PASS -> Square becomes the Gate-2/calibration benchmark; Can-PH becomes the negative
  control ("local data value does not imply allocation value under homogeneous response and
  equal costs").
- FAIL (homogeneous) -> two-domain negative control; the allocation program moves to
  heterogeneous-COST settings (SO-101 operator time) as the remaining lever.
