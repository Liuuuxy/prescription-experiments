# PRE-REGISTRATION ADDENDUM 2: calibration-direction acquisition + stopping rules
Frozen 2026-08-25, before the Gate 1 screen unblinds and before any Gate 2
computation. Companion to PREREG_RESPONSE_MODEL.md as corrected by
PREREG_CORRECTION_1.md.

## Locked acquisition rule: decision-focused integrated variance reduction
Let B_lib be the finite library of feasible forward allocations for the target
profile (one-region x4, b_div, b_cov, b_pfail, and post-Gate-1 mixtures).
The GP maintains a JOINT posterior over latent advantages relative to diverse:
  A(b) = E[DeltaJ | x_b] - E[DeltaJ | x_div]   (latent level, not measured runs).
At calibration step t (t = 0, 1, 2, 3):
 1. Contender set C_t = { b : UCB_t(A(b)) >= max_b' LCB_t(A(b')) }.
 2. For every remaining removal direction h in H_lib, score the reduction in
    contender-weighted integrated posterior variance:
      Score(h) = [ tr(W_t Sigma_t) - tr(W_t Sigma_{t|h}) ] / cost(h)
    with W_t uniform on C_t (zero elsewhere). Sigma_{t|h} is the analytic
    posterior covariance after adding h's add-back observation(s) under the
    endpoint-level signed noise model (C1) — no outcome guessing; with frozen
    hyperparameters this is exact.
 3. Evaluate the argmax direction with BOTH paired seeds (2 new runs:
    J(D0 - H_h) at seeds 0 and 1); condition the GP.

H_lib = the 4 single-region removal directions of the target profile.
cost(h) = GPU run cost; constant in sim (divisor inert here, meaningful for G1).

## Locked stopping certificates
- TARGET: stop and prescribe b* when LCB(A(b*)) > 0 AND
  LCB(A(b*)) >= max_{b != b*} UCB(A(b)) - eps, with eps = X (the Gate-1 MDE,
  already predeclared; differences below the detectable floor are
  decision-irrelevant).
- FUTILITY: stop and prescribe diverse when max_b UCB(A(b)) <= 0.
- CAP: at most 3 calibration directions. At the cap: prescribe the allocation
  with the highest positive simultaneous LCB; if none positive, diverse.
- t = 0 stopping is ALLOWED (the meta-mean alone may certify): reported as
  "calibration not needed for this profile", which feeds the 0/1/2/3-direction
  ablation honestly.

## Sharpenings locked with the rule
S1. ALL bounds (contender sets and certificates) are SIMULTANEOUS over B_lib,
    computed from joint posterior samples of {A(b)} (>= 10^4 draws), at a
    predeclared level: 90% simultaneous. Pointwise +-k*sigma is never used.
    Phase-1 language (per C4): these are stopping RULES whose operating
    characteristics are measured empirically on held-out profiles; the word
    "certificate" carries formal weight only under Phase 2's sampled-profile
    conformal calibration.
S2. Hyperparameters (kernel + all noise components incl. the seed-level random
    effect of C1) are fitted on meta-training profiles and FROZEN before target
    calibration. Target observations condition the posterior only; no retuning.
S3. Acquisition ablation (extends the locked grid): decision-focused (this
    rule) vs maximum raw posterior variance vs uniformly random direction.
    HONESTY CAVEAT, registered now: with |H_lib| = 4 and a cap of 3, the
    acquisition rule's headroom over random is largest only when stopping
    early; we therefore report the realized share of t<=1 stops, and if most
    profiles run to the cap, the acquisition comparison is declared
    underpowered rather than spun.
S4. Every calibration observation enters the GP with the endpoint-level signed
    covariance of C1; J(D0) endpoints are shared with the forward observations,
    and the certificates therefore tighten from calibration through BOTH the
    residual update and the baseline-noise cancellation.
