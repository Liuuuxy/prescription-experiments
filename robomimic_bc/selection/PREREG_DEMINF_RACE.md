# PRE-REGISTRATION: DemInf selection race on Can-MH (2026-08-27, before scores or runs exist)

## Question
Does the best published demo-quality score (DemInf, mutual-information scoring — which lists
robomimic Can-MH among its own benchmarks) predict CAUSAL demo value on Can-MH, where our own
cheap scores (loss 0.578, gradient 0.499) failed and length (0.945) only identifies operators?

## Scoring (deminf_score.py, running)
Two beta-VAEs (state->8, action->4, beta=0.02) + KSG-1 MI (k=5, batches 1024, 4 passes,
1-99% clipping); demo score = mean over its steps. Raw-vector ablation for the
"your VAE isn't their VAE" objection. Scores frozen to deminf_scores.json BEFORE any race run.

## Race
Masks: deminf_top100 / deminf_bottom100 / sel_random100 (crc32 draw) over the 300 MH demos.
Learner: BC-RNN (seq 10, 300 epochs, equal-per-trajectory weighting) — LEARNER GATE: race
launches only if tonight's small-N BC-RNN runs show mid-band sigma_run <= 4pp; else the race
moves to tomorrow at 5 seeds. Eval: the frozen Can-PH exam (SAME PickPlaceCan env, kwargs
verified identical). E_probe only tonight; E_test written per-run but SEALED (pool harvests
probe only) for tomorrow's confirmation of whichever contrast wins.
Queue order (any prefix valid): s1: bottom,top,random; s2: same; s3: same. 9 runs.

## Read (frozen)
Primary: random - bottom (harm contrast), mean over paired seeds.
Secondary: monotone ordering top > random > bottom.
MDE stated at analysis time from the measured RNN sigma at the relevant success band, per the
C8 discipline (both sd scales, >=10 df or the race is declared screening-only). If top-100
saturates near ceiling, the top-vs-random read is censored and reported as such; the harm
read survives. Effects below the stated MDE are reported as "unresolved at our power",
never as method failure. Fairness: this is a pilot-only PROSPECTIVE protocol of our
construction; a null bounds where the score's signal exists, it does not refute DemInf's
published retrospective setting.
