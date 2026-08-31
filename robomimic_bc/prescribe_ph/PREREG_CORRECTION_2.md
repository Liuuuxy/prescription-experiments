# PRE-REGISTRATION CORRECTION v2 (2026-08-26)

**WRITTEN AFTER E_test UNBLINDING.** This is the material fact and is stated first: the
Gate-1 analysis was run, its output preserved verbatim (GATE1_RAW_OUTPUT.txt, hashed), and
this correction was written afterwards in response to an adversarial re-analysis (4 independent
probes + 4 skeptics + synthesis; full transcript in the session workflow journal). It therefore
has LOWER evidentiary standing than PREREG_PH_BENCHMARK.md / _RESPONSE_MODEL.md /
_CORRECTION_1.md / _ACQUISITION.md, all of which were frozen before any result existed.
Originals are unmodified; their hashes still verify.

## C4 — GATE 1 IS VOID
MDE.json (X = 0.042381) and gate1_result.json (7 "hits") are RETRACTED.
Cause: X was computed from a variance estimated on **2 degrees of freedom**. Realized
sd_profile = 0.01844 against the 0.18833 that exchangeability implies from sd_per_seed
= 0.26633 (ratio 0.098). Lower-tail chi2_2 P = 0.0095; exact ANOVA F(2,3) = 0.005775,
P = 0.0058; and the assumption-free version: re-running the entire frozen pipeline over all
56^3 = 175,616 relabelings of which two of the eight 104-demo add arms play div1/div2 puts the
realized X at the **1.23rd percentile** of its own null (median X = 30.1pp).
Honest noise from an INDEPENDENT source (39 same-mask seed pairs, byte-identical data):
sigma_run = 16.73pp; SE of the registered contrast = 14.5-17.9pp; **honest 80%-power
MDE = 36-45pp**, i.e. X was 8.5-10.5x too permissive.
The 7 hits: honest z = 1.09-1.72, best raw one-sided p = 0.043, Bonferroni over 18
comparisons >= 0.77. Exact permutation at the frozen X expects **4.75 hits**, P(>=7) = 0.187;
P(max contrast >= observed +24.86pp | H0) = 0.32-0.41. The hits are what the broken
threshold predicts, not evidence.
NOTE (do not overstate): the prereg FORMULA is not broken. Its /sqrt(3) grand-mean divisor and
its missing sqrt(1.5) null-level factor nearly cancel (ratio 1.067), and under H0 the rule is a
correctly-sized ~5% per-comparison screen in expectation. 100% of the failure is one flukey
2-df variance draw. Do not write this up as "two compounding errors".

## C5 — THE RULE AS CODED
gate1_analyze.py declares "OPPORTUNITY DETECTED" on `if gate1_hits:` — i.e. **>= 1 hit** out of
18 comparisons, with no multiplicity control registered anywhere. Type-I error under the global
null: **0.318** (0.989 at the frozen X). Any future gate must register its multiplicity rule.

## C6 — THE NOISE MODEL IS RETRACTED
PREREG_PH_BENCHMARK.md's "Noise / inference discipline" paragraph claimed "all variance is
training-side - exactly what pairing cancels". First clause TRUE (eval-side sd <= 1.6-3.5pp,
<= 5% of Var(J_test); corr(J_probe, J_test) = 0.943 over 78 runs). Second clause FALSE: there is
no seed MAIN effect to cancel. corr(J_s0, J_s1) over 39 identical-mask pairs = **-0.076**;
profile x seed cell means explain R^2 = 0.041; matched-seed contrast sd (0.1886) is no better
than crossed-seed (0.1831). **Pairing on training seed buys nothing in this domain.** (Contrast
with the pi0 domain, where pairing demonstrably worked - this is a domain property, not a law.)

## C7 — b_pfail IS NOT A DISTINCT ARM
Its allocation is IDENTICAL to b_div on the balanced profile (6,6,5,7) and within 1-2 demos on
the starved profiles. It was therefore a third null draw, not a baseline. Four free null
contrasts existed and went unused; n = 10 instead of 6 would have exposed the variance fluke
before unblinding. Any reuse of this design must replace b_pfail with an allocation that
actually differs from b_div.

## C8 — VARIANCE-ESTIMATOR DISCIPLINE (binding on all future MDEs in this program)
1. Never estimate an MDE from fewer than **10 df** when per-unit dispersion is available.
2. Always report BOTH the clustered sd and the per-unit sd, and **refuse to proceed when they
   disagree by more than 3x**. Here they disagreed by 14x, and that was visible in MDE.json
   BEFORE unblinding.
3. Always report the lower-tail chi-square probability of the script's own sd estimate.

## C9 — NEW PRIMARY ESTIMAND (registered BEFORE the confirmation runs below)
The allocation screen is abandoned; the regional RESPONSE CURVE becomes the estimand.
Specification, frozen now:
- Unit = (run, region). Outcome = J_r (per-region success on E_test).
- Model: J_r ~ C(run) + C(region) + C(bin(n_r)), bins [0, 1, 7, 16, 24, 36, 61],
  reference bin 16-23, SEs clustered by run.
- Reported effect: per-demo slope of J_r on n_r with total dataset size held constant.
- **Every difference-in-differences MUST include a zero-dose placebo arm on untreated
  regions** (the uncorrected DiD produced a spurious region-heterogeneity pattern; placebo
  DiD on untreated regions returns +9.4/-17.4/+8.8/-13.9pp, and after correction the four
  regions are homogeneous, Wald chi2(3) = 1.96, p = 0.58).
- Duplicate-allocation arms (C7) are ineligible as baselines.
- CONFIRMATION SET, frozen before launch: **2 fresh training seeds (2, 3) x 12 existing masks**
  = 24 runs. Masks: for each of the 3 profiles, {D0, rm_xlo_ylo, add_xlo_ylo, add_xhi_yhi}.
  These span n_xlo_ylo = 0 / 4-20 / 28-44 without building any new mask.
  Pre-registered read: the out-of-sample slope must be positive with a run-clustered
  t >= 2 and a point estimate within a factor of 2 of the in-sample +0.44pp/demo.
  A negative or null slope retracts C9's finding as well.

## C10 — EVAL REPRODUCIBILITY (latent defect, contributed 0pp here)
`reset_to({"states": ...})` restores only [time, qpos, qvel]; `sim.data.qacc_warmstart` leaks
from the previous episode, and the OSC nullspace reference is never refreshed - so an episode's
outcome is formally a function of the preceding episode sequence, not of (start, policy) alone.
MEASURED IMPACT ON THIS EXPERIMENT: **zero.** A stored checkpoint re-evaluated under the
original seeding order reproduced its successes bit-for-bit (0/200 flips); the eval-side term is
bounded at sd <= 1.6pp. An earlier claim of 1.5-5pp eval scatter came from an audit harness that
loaded checkpoints without training and so never seeded numpy - that harness was wrong, not the
experiment. The defect is nonetheless REAL and must be fixed before any checkpoint is
re-evaluated in a fresh process.

## What survives, and is the actual scientific content
Regional data reliably and CAUSALLY buys regional competence:
  own-region +24 demos = **+9.8pp** (p = 0.019); own-region removal = **-14.4pp**
  (t = -4.07, p = 4.7e-4); slope **+0.447pp per demo** with total size held at 104
  (t_cluster = 5.87), +0.439pp/demo on the full set (t = 9.32).
And prescription still cannot pay on this benchmark, for a reason that is structural rather
than statistical: deployment weights span only q_r = 0.215-0.2755, so a pure reallocation
enters J_deploy multiplied by (q_r - 0.25) in [-0.035, +0.026]. Brute force over EVERY
allocation of 104 demos (50/region cap) gives a **design ceiling of 6.48pp total spread**
(placebo-corrected model-free estimate: 0.79pp). The declared hits were 3-10x larger than the
largest effect physically available. **More seeds cannot rescue this design; only concentrating
deployment weight can** (ceiling 9.98pp at q = 0.50, 13.74pp at q = 0.70, 16.56pp at q = 0.85).
