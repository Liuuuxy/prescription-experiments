# Equal-weighting control verdict (Can-MH, 2026-08-25)

**Question.** Was the +23.3pp better-vs-worse operator gap (B=25, seeds 0-2, frame-uniform
sampling) a real data-quality effect, or an artifact of frame-uniform sampling giving long
"worse" demos (304 vs 143 frames) ~2.1x the optimizer weight?

**Design.** 16 runs: eq_D0only x3, eq_{better,worse,random}_B25 x3 seeds (equal-per-trajectory
WeightedRandomSampler), plus standard-weighting better/worse at 2 extra seeds (s3, s4).

**Result.**
| | better-worse paired gaps | mean |
|---|---|---|
| standard, seeds 0-2 (original claim) | +0.24 / +0.41 / +0.05 | **+23.3pp** |
| equal-weighted, seeds 0-2 | -0.24 / +0.23 / -0.10 | **-3.7pp** |
| standard, extra seeds 3-4 | +0.15 / -0.22 | (5-seed mean +12.6pp, sd 24pp, t=1.2) |

Under equal weighting, "worse" additions are not harmful (delta vs eq_D0only: mean +9.3pp).

**Verdict.** The MH operator-quality ordering is NOT established: it fails to replicate under
equal weighting AND erodes under extended seeds. Some mix of (a) weighting artifact and
(b) seed luck at n=3 with a ~20-25pp per-seed noise floor. Consequences:
1. The "worse demos are causally harmful" premise behind the length-based quality gate is
   unsupported on this task; the two-signal quality-filter claim stays UNLOCKED (as agreed).
2. Any future robomimic experiment must (i) weight trajectories equally, (ii) compute its
   minimum detectable effect from null pairs BEFORE reading treatment arms. Can-MH BC at
   3 paired seeds cannot resolve differences < ~20pp.
3. The Can-PH region benchmark (prescribe_ph/) adopts both rules by construction.
