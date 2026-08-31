# PRE-REGISTRATION CORRECTION v1 (2026-08-25)
Applies to PREREG_PH_BENCHMARK.md and PREREG_RESPONSE_MODEL.md. Those documents
are NOT modified; their hashes in PREREG_SHA256.txt remain the frozen originals.
This correction is recorded BEFORE any Gate 2 result is inspected and BEFORE the
Gate 1 screen unblinds. Status of experiments at freeze time: pilot smoke run
mid-eval; no pilot, screen, or calibration result read by anyone.

## C1. Shared-baseline noise: SIGNED incidence, not equicorrelation
Amendment C of the response-model addendum described the shared-J(D0) label
noise as "equicorrelated". Wrong sign structure: J(D0) enters forward labels
y+ = J(D0+A) - J(D0) with coefficient -1 and add-back calibration labels
y- = J(D0) - J(D0-H) with +1, so Cov(y+, y-) = -sigma_D0^2 while within-type
covariances are +sigma_D0^2. CORRECTED implementation: endpoint-level noise.
Every measured endpoint J-hat(dataset; profile, seed) carries its own noise
term; a label is a signed difference of two endpoints; the label covariance is
G Sigma_endpoint G^T with G the signed incidence matrix (equivalently
y_i = f(x_i) + s_i * u_{profile,seed} + eps_i, s_i in {-1,+1}, generalized to
all shared endpoints, e.g. J(D0) reused across all removals and forwards of a
profile-seed). One further term the incidence matrix does NOT capture,
recorded as an empirical question: DISTINCT endpoints trained at the same seed
(e.g. J(D0+A_1) and J(D0+A_2) at seed 0) may be positively correlated through
a seed-level effect (the pi0 race measured seed as the dominant variance
component). Estimate this component from the D0 replicates and paired b_div
nulls; include it as a per-seed random effect if it is non-negligible.

## C2. Two separate uncertainty sources in all reporting
Deterministic rollouts eliminate ONLY repeated-rollout randomness given (start,
policy). E_probe/E_test remain finite samples of the deployment distribution.
All reported uncertainties separate:
 (a) TRAINING uncertainty: cluster over profile/seed (paired-seed differences);
 (b) DEPLOYMENT-SAMPLING uncertainty: bootstrap over unique starts.
Note (recorded, not a loosening): Gate 1 COMPARISONS are paired per-start on the
shared fixed set, where start-sampling error largely cancels; absolute deployment
J estimates carry the full bootstrap-(b) interval.

## C3. The mean is "factorized scaling-curve", NOT FSC
The additive regional decomposition mu = sum_r q_r [S_r(n_r+a_r) - S_r(n_r)] is
FSC-INSPIRED but not the published FSC construction (arXiv 2505.07728 fits
factor curves of OVERALL policy performance as factor data varies; not
equivalent to an additive per-region decomposition of a deployment mixture).
Renaming: "factorized scaling-curve mean" everywhere. The published FSC method
is implemented SEPARATELY, under the same target-fit budget, as its own
competitor baseline. The "mean only" ablation no longer doubles as the FSC
comparison; the ablation grid gains one entry: (1b) published-FSC baseline.

## C4. Conformal coverage claims require profile exchangeability we do not have
The 75%/87.5% figures in Amendment E are quantile-resolution arithmetic; they
are distribution-free coverage ONLY if calibration profiles are exchangeable
draws with the test profile. Our balanced/starved archetypes are DESIGNED, not
sampled. Corrected two-phase policy:
 - Phase 1 (screen + first calibration wave): intervals are described as
   "empirically calibrated on this benchmark's designed profiles"; NO formal
   coverage claim of any level.
 - Phase 2 (only if Gates 1-2 pass): predeclare a profile-generating
   distribution over D0 region proportions (symmetric Dirichlet mixture with
   sparse-corner mass; parameters and crc32 seeds fixed in a Phase-2 addendum
   BEFORE sampling), sample >= 10 independent profiles, and make coverage
   claims scoped to THAT distribution only. No coverage claim transfers to
   pi0, other tasks, or the real robot.

## C5. Kernel input addition (advisor): normalized allocation a/B
The <=6-dim summary of Amendment D leaves distinct region pairs with identical
entropy/coverage statistics indistinguishable. The kernel gains an ADDITIVE
LINEAR component on the 4-dim normalized allocation a/B (summaries keep their
ARD-RBF). Frozen before any Gate 2 fit.
