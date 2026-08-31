# PRE-REGISTRATION: convergence-timing sweep (2026-08-30, before any run)
Question (advisor's open hypothesis): do policies trained on fewer demos converge in fewer
GRADIENT STEPS? The 250-500 checkpoint window could not see this (everything converged
earlier); this sweep watches the climb itself.

Design: plain BC-MLP, v2 conventions otherwise (equal-traj weighting, horizon 500).
N in {10,40,80,120} x draws {0,1,2} (reusing vd_N*_d* masks) x seeds {0,1} = 24 runs.
Train 250 epochs, checkpoint every 10; evaluate checkpoints [10,20,30,50,75,100,150,200,250]
on the stratified 50-scene probe subset (sealed E_test untouched).

Primary estimand: plateau epoch per run = first checkpoint whose score >= 90% of that run's
maximum across the 9. Read: Spearman correlation of plateau epoch with N over 24 runs +
per-N medians. Hypothesis supported if plateau epoch increases with N (rho > 0, p < 0.05);
refuted if flat or negative. Success-curve shapes reported descriptively either way.
