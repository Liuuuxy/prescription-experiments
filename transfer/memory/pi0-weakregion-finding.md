---
name: pi0-weakregion-finding
description: "pi0's weak-region result on PickPlaceCounterToSink — fails at grasp on tall/awkward objects; the concrete targeting signal for the experiment"
metadata: 
  node_type: memory
  type: project
  originSessionId: 4f31bea1-2bdd-4e89-9f93-63656e3f0e12
---

First weak-region result (2026-06-10), pi0 on PickPlaceCounterToSink, n=50, via `~/robocasa_experiments/policy_analysis/analyze_pi0_weakregions.py`. Output `~/robocasa_experiments/weakregion/pi0_PickPlaceCounterToSink/`.

**pi0 overall ~52-58% (this run 26/50). Dominant failure mode: `fail_no_grasp` = 96% of failures** — pi0 almost never fails at transport/placement; it fails at the GRASP.

**CORRECTION (A2 coverage/geometry analysis, 200 resets + the 50 eps):** the apparent "fails on tall/cylindrical objects" pattern was **small-sample noise** (1-2 eps/category). Aggregate median-splits show **NO effect**: tall 54.5% vs short 52.2%; wide 54.5% vs narrow 52.2%; rare (data-sparse) 56.8% vs common 38.5% (rare succeeds MORE — opposite of the coverage hypothesis). So at n=50 the object/region localization is NOT reliable; do not target on it. Script: `policy_analysis/coverage_geometry.py`, output `weakregion/coverage_geometry.json`.

**What IS robust:** the failure MODE — `fail_no_grasp` dominates (n=50: 96%; n=150: 86% no-grasp + 13% grasped-no-transport) → pi0 fails mostly at GRASPING.

**RESOLVED with n=150 + per-episode geometry + a numpy logistic predictor (`policy_analysis/predict_failure.py`):** pi0 overall 52.7% (79/150, CI 45-61%). **Object HEIGHT is a real but MODERATE predictor of failure** (5-fold CV AUC = 0.628; standardized coeff height = -0.57, all others ~0). Univariate height bins: short (<6cm) 67% > medium 55% > tall (>11cm) 36% — a clean monotonic ~31pt gap. Width, lateral position, depth are NOT predictive (coeffs ~0). Lowest-predicted-success episodes are all tall objects (reamer h=0.28, wine h=0.26, cheese_grater h=0.24).

Scientific arc (why rigor mattered): naive eyeball@n=50 said "tall fail 0%, short 100%" (overconfident); A2 median-split@n=50 with category-avg geometry said "NO effect" (false null, under-powered + crude features); n=150 + per-episode geometry + continuous predictor gives the truth: **height matters, moderately (AUC 0.63), tall ~36% vs short ~67%.**

**EMBODIMENT-LIMIT TEST (`policy_analysis/embodiment_test.py`, gating question: is the failure data-fixable or a physical limit?):** Panda gripper max aperture = **0.08m (8cm)**. 137/150 objects have bounding-diameter > 8cm, **YET they still succeed 52%** → bounding-width is NOT a hard limit (gripper grasps the narrow side / a graspable part). **=> NO aperture/embodiment limit; the objects ARE graspable → pi0's height-driven failures are a SKILL/grasp-strategy gap (EPISTEMIC, data-addressable), not aleatoric/physical.** This is the GOOD outcome: the targeted-data experiment has a valid premise. Caveat: doesn't 100% rule out a height-specific physical effect (grasp stability for tall objects), but width/aperture is cleared. (Note: obj_width = max horizontal extent, which over-estimates ungraspability — the script's first auto-verdict was wrong for this reason; corrected to use the empirical over-aperture success rate.)

**FAILURE FORENSICS + CONTINUOUS-PROGRESS ANALYSIS (`policy_analysis/progress_analysis.py`, n=150):** Of 71 failures, **76% are "never_touched"** (object moves <1cm) — the bottleneck is grasp INITIATION/approach, not transport/placement (only 14% ever grasped). Height effect confirmed 3 ways but WEAK: corr(height, success)=-0.28, corr(height, progress)=-0.29, corr(height, max_lift)=-0.27; all 4 geometric features together explain only **R^2≈0.08-0.09**. Continuous progress barely beat binary (failures are bimodal) — so a continuous signal does NOT rescue localization here. **KEY STRATEGIC TAKEAWAY: simple geometry is a WEAK targeting axis (<10% of variance); ~90% is object-INSTANCE-specific (which objaverse mesh/affordances) or stochastic. => Target on POLICY UNCERTAINTY (diffusion action-variance) / CROSS-POLICY DISAGREEMENT / per-instance failure, NOT on hand-picked geometric features.** (This is the active-learning paper's lesson, confirmed empirically.)

**Targeting implication for [[first-experiment-pickplace-sink]]:** ~~the targeting axis is object HEIGHT~~ (superseded — height is too weak); use uncertainty/instance-level signals. Earlier height note kept for context: (tall objects = the weak region), NOT position/width. But it's a moderate signal — AUC 0.63 means ~much failure variance is unexplained (grasp failures also hit short objects), so set expectations: targeted-on-tall-objects data should help partially, not fully. The experiment needs enough statistical power to detect a partial effect.

**Implication for [[first-experiment-pickplace-sink]]:** before defining a "targeted region," need a much larger weak-region run (>=150-300 eps) with **per-episode geometry logged directly** (not category-avg) + Wilson CIs per bucket. The current signal is too noisy to target on. This is design-concern B5 (sample size) confirmed empirically. Next: bigger pi0 run w/ geometry; GR00T cross-policy (universally-hard check).
