# DataMIL-proxy datamodel on the existing corpus (2026-08-27): NOT FITTABLE

Fit: X[run,demo]=membership (108 Can-PH runs x 200 demos, mask sizes 40-104), y=J_deploy
on sealed E_test, ridge (lam=5) + total-count covariate. In-sample R2 0.451.

VERDICT: per-demo value estimates are noise. Split-half reliability (54/54 runs):
pearson -0.318, spearman -0.282; within-region (removing the only identified structure):
-0.356. Region-mean coefficients span only 0.62pp; within-region coefficient sd 1.0pp is
pure overfit. Cause: sigma_run = 16.7pp across-run noise + 200 coefficients from 108
region-STRUCTURED (not iid) masks.

FAIRNESS CAVEAT (must accompany any use of this as a baseline): DataMIL prescribes many
hundreds of policies trained on iid random subsets; our corpus is both ~5x too small and
adversarially structured for demo-level identification. This result says "a datamodel
cannot be extracted from the corpus a realistic prescription study already owns," NOT
"DataMIL fails." A faithful test needs ~500 random-subset runs (~35 GPU-h) — priced, not run.
