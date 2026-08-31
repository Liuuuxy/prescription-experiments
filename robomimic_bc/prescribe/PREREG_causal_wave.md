# Pre-registered predictions — score-arm causal wave (written 2026-08-25, before results)
Arms (B=50 fixed sets + D0, paired seeds 0/1/2 vs existing runs):
- predT_bad  (longest-duration 50; 86% worse-operator): EXPECT harm ≈ the labeled worse arm
  (worse−random at B50 was −5.7pp): predT_bad − random_B50 < 0 in ≥2/3 seeds, mean ≤ −4pp.
- predT_good (shortest 50; 4% worse-op): EXPECT ≈ better/okay arms: predT_good − predT_bad > 0
  in 3/3 paired seeds — THE headline claim (zero-label, zero-model duration filter causally
  steers collection value).
- predL_bad  (top-50 clean-reference loss; 60% worse-op): intermediate harm; predL_bad < random
  in ≥2/3 seeds validates the model-side (RHO-style) score causally.
Kill criterion: predT_good − predT_bad ≤ 0 on ≥2 seeds ⇒ duration filter NOT causal despite
0.886 detection AUC (would imply worse-op harm is not mediated by what duration measures).

## SUPERSEDED — CANCELLED 2026-08-25, ZERO RUNS COMPLETED
The wave launched twice and was stopped both times before any of the 9 runs finished
(results.json contains no run_pred* entries). It is cancelled, not merely interrupted:
(1) the equal-weighting control (EQ_CONTROL_VERDICT.md) retracted the better>worse
ordering this wave was built to explain — the positive control no longer exists;
(2) these runs would have used the same frame-uniform weighting shown to be the artifact;
(3) the honest per-seed noise floor (~20pp at n=3 paired seeds) cannot resolve the
predicted effect. The kill criterion in the original text is therefore moot. The 9
manifest entries and the three fixed masks (predT_good/predT_bad/predL_bad in
can_mh_work.hdf5) are left in place for the record but MUST NOT be run as designed.
