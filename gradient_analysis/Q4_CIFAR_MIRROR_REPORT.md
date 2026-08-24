# Q4: CIFAR arm-race mirror — the positive control for the whole gradient program (2026-08-08)

**Question.** The RoboCasa bandit found no arm effects anywhere (12 condition-arm pulls, 4
gradient-cluster pulls, no predictive gradient statistic). Is that because (a) the robot's
flow-matching training loss does not encode the weak region — the gradient-encoding
hypothesis — or (b) the arm-race design itself cannot resolve composition effects at
budget B=200? Q4 answers this by running the *same design* in a domain where the loss
provably encodes the weak region.

**Setup** (`/data/xinyua11/xgradtest/xarm_race.py`, results in `xgradtest/armrace/`).
Base model: ResNet-18 trained on CIFAR-100 with 20 classes made rare (10% of their data) —
a weak region (base rare-class accuracy 14.6% vs 43.0% common) whose cause is known to be
missing data, under a cross-entropy loss that encodes class. Arms (each "pull" = draw 200
pool examples, fine-tune base+draw from the same base checkpoint for 2000 steps, evaluate
on held-out test): **null** (nothing added — noise floor), **rare** (200 weak-region
examples — the analog of the tall arm), **random** (200 uniform — control), **easy**
(200 well-learned-class examples), **gradarm_a/b** (the two most-separated k-means
clusters of the pool's whitened per-example gradient sketches — identical recipe to the
robot gradarm test). 4 paired-seed rounds each; 24 fine-tunes; ~30 min total.

## Results

Δ rare-class accuracy (the target stratum), per round and mean:

| arm | 4 rounds | mean | Δcommon | Δoverall |
|---|---|---|---|---|
| null | −0.3 −0.2 −0.4 −1.0 pp | **−0.5** | −0.3 | −0.3 |
| **rare** | +3.7 +3.3 +4.3 +3.9 | **+3.8** | **−1.0** | −0.0 |
| random | +1.3 +0.6 +1.1 +0.6 | **+0.9** | −0.4 | −0.1 |
| easy | −0.2 −0.5 −0.4 −0.5 | **−0.4** | −0.2 | −0.2 |
| gradarm_a | −0.5 +0.5 +0.7 +0.5 | +0.3 | −1.1 | −0.8 |
| gradarm_b | +1.9 +0.8 +0.1 +0.7 | +0.9 | −0.6 | −0.3 |

Batch-level gradient separation (whitened batch-mean distance, random-vs-random null =
0.099): rare 1.4× null, gradarms 2.6–3.5× null. Statistic-vs-outcome correlations over
the 20 non-null pulls: **cos-to-rare-direction vs Δrare ρ=+0.70; mean grad-norm vs Δrare
ρ=+0.81**; unsupervised batch stats (coherence, whitened separation, self-similarity)
ρ≈0.16 — non-predictive.

## Four conclusions

1. **The design works when the loss encodes the target.** The rare arm separates from
   random decisively — +3.8 vs +0.9pp on the target stratum, consistent in all four
   rounds, against a null floor of ±0.4pp — with the same B=200, 4-pulls-per-arm race
   that produced pure noise on the robot. The robot nulls are a *domain property*, not a
   design failure.
2. **The gradient does provide the information — when encoding holds.** The targeted
   statistic (a draw's mean cosine to the weak-region gradient direction) predicts the
   realized target-stratum improvement at ρ=0.70 (grad-norm ρ=0.81). The identical
   statistics on the robot predicted nothing (|ρ|≤0.23, Q3). This is the experiment-level
   version of the gradient-encoding principle, upgraded from AUC-level evidence.
3. **Gradient-cluster arms fail even on CIFAR.** The gradarm clusters are the most
   gradient-distinct batches available (2.6–3.5× null separation — larger than the robot
   gradarms' 2.4×) yet their outcomes ≈ random, and unsupervised batch statistics predict
   nothing here either. Batch *distinctiveness* is the wrong object everywhere; what
   matters is alignment with a *target* direction the loss encodes. (This retroactively
   explains the robot gradarm sign-flip as expected behavior, not just noise.)
4. **Even the retention constraint echoes.** The rare arm costs −1.0pp on common classes
   — the CIFAR miniature of the forgetting/retention constraint that bound every robot
   selection arm.

## Caveat

The CIFAR noise floor (±0.4pp) is ~8× tighter than the robot's (σ_e=3.3pp). The CIFAR
rare-vs-random gap (2.9pp) would sit *at the edge of one-pull detectability under robot
noise* — so the robot program's problem is compounded: the loss doesn't encode the
region, AND its noise floor would hide a CIFAR-sized effect anyway. Both facts belong in
the paper; the first is the mechanism, the second is the power wall.

## Paper placement

This is the missing positive control for the teachability-gate story: same race, same
budget, same statistics — signal present where the loss encodes the target, absent where
it doesn't. Pairs directly with the ClusterUCB/LESS contrast (their influence-proxy
reward assumes encoding; CIFAR is their regime, VLA flow-matching is not).
