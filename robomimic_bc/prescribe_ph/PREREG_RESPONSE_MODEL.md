# PRE-REGISTRATION ADDENDUM: response model for Gate 2 (calibration)
Frozen 2026-08-25, BEFORE the screen unblinds. Companion to PREREG_PH_BENCHMARK.md.

## Locked family
Predicted improvement of an allocation, over the current dataset state:
  DeltaJ(x) = mu_theta(x) + g_p(x)
- x = (s_pi, q, n_before, a, c): policy state, deployment weights, per-region
  counts before, added counts a, collection cost c (constant in sim; slot kept for G1).
- EVERY observation is an addition transition. A removal calibration on subset H
  enters as n_before = n(D0 - H), a = n(H), y = J(D0) - J(D0 - H): evidence about
  adding at slightly lower coverage, not a "negative allocation".
- g_p: target-profile-specific GP residual conditioned on the (up to 3) actively
  selected calibration observations. Conformal correction on top (below).

## Amendment A — the mean is a PARAMETRIC scaling curve, not a neural net
mu_theta is the factored-scaling-curve form: per-region success follows a
saturating power law S_r(n) = A_r - C * (n + n0)^(-beta) (A_r bounded by 1), and
  mu_theta(x) = sum_r q_r * [S_r(n_r + a_r) - S_r(n_r)].
theta = (A_r ties/pooling, C, beta, n0) fit across profiles, leave-one-profile-out.
Rationale: (i) at the 40-80 observations this program will ever have, a neural
mean overfits and its uncertainty is unusable; (ii) this form IS the closest
competitor (FSC, arXiv 2505.07728) -> the "mean only" ablation doubles as the
named competitor baseline, and the paper claim becomes "active calibration
improves on scaling-curve prescription".

## Amendment B — meta-training data is robomimic-only
The pi0 intervention ledger is EXCLUDED from meta-training: measured cross-domain
transfer for cheap signals in this project is c ~ 0 (llm_borrow verdict), and the
pi0 race is a composition-null — training on it teaches the mean "allocations do
nothing". The pi0 ledger appears only as a POST-LOCK transfer test, reported
whatever it shows.

## Amendment C — shared-baseline noise structure (mandatory)
Within a profile x seed, all labels share measured endpoints (every forward y
shares J(D0); removal y shares it too). Eval is deterministic given the policy,
so label noise = training draw+seed noise, and it is EQUICORRELATED within a
profile-seed through the shared baseline. The GP likelihood must carry a
per-profile-seed shared-offset variance plus an independent nugget; both
components estimated from D0 replicates and the paired b_div nulls. Ignoring
this correlation -> overconfident posteriors -> Gate 2 optimism. (Ablation:
noise structure on/off, robustness appendix.)

## Amendment D — feature parsimony for the kernel
The GP kernel sees a predeclared <=6-dim summary only (raw per-region vectors go
into mu_theta, never the kernel):
 1. coverage-repair mass  sum_r a_r * max(0, q_r - n_r/N)
 2. failure-weighted mass sum_r a_r * q_r * (1 - Jhat_r^probe)
 3. post-allocation worst coverage  min_r (n_r + a_r) / ((N + B) * q_r)
 4. allocation entropy
 5. total budget B
 6. mean probe success (policy state)
ARD-RBF over these; hyperpriors fixed before unblinding.

## Amendment E — honest conformal arithmetic
Selection-induced optimism is handled as proposed: the conformal score for a
held-out profile is the WORST standardized prediction error over every candidate
allocation in that profile -> simultaneous one-sided bounds. But one profile
contributes ONE score, so one-sided coverage is capped at 1 - 1/(n_profiles+1):
  3 screen profiles -> 75% max; 7 profiles -> 87.5%; 90% needs >= 9.
Therefore: screen-phase intervals are GP-posterior bounds LABELED UNCALIBRATED;
conformal claims are made only at the full profile set, using leave-one-profile-
out jackknife+; the achieved coverage level is reported, not rounded up.

## Post-Gate-1 meta-training arms (so the optimizer never extrapolates)
Per surviving profile, in addition to the screen arms: all six 50/50 region
pairs and four Dirichlet(1,1,1,1) mixtures (crc32-seeded, clipped to well
feasibility), 2 paired seeds each.

## Locked ablation grid
(1) mean only (= FSC competitor); (2) count + failure heuristics (b_cov, b_pfail);
(3) GP without target calibration; (4) random vs actively selected calibration
directions; (5) 1 vs 2 vs 3 calibration directions; (6) GP bounds vs conformally
corrected bounds; (7, appendix) shared-baseline noise on/off.
