# Q2 (absorption dynamics) + Q3 (16-pull predictive ablation) — 2026-08-03

Companion to `Q0_REPORT.md`. Data: `q2_absorption/<pull>/<step>_{norms,sketches}.npy`
(6 trajectories × 4 checkpoints × 360 demos), `q3_pull_ablation.json` (16 pulls).

## Q2: every pull absorbed its demos by step 5000 — then specialized away from the pool

Mean LoRA-grad norm along each pull's own fine-tune (own = its 200-demo draw;
ctrl = 100 never-trained pool demos; d0 = 60 demos of the always-included base set).
Reference levels at π₀ before fine-tuning: own/ctrl ≈ 3.1–3.3, d0 ≈ 0.55.

| step | own | ctrl | d0 |
|---|---|---|---|
| 5000 | **0.38–0.41** | **0.43–0.47** | 0.36–0.40 |
| 10000 | 0.55–0.60 | 1.11–1.17 | 0.52–0.60 |
| 15000 | 0.61–0.66 | 1.67–1.72 | 0.67–0.74 |
| 19999 | 0.64–0.68 | **2.37–2.45** | 0.60–0.64 |

(Ranges span all six trajectories — tall_j3/j4, gradarm_a/b × j3/j4. The curves are
near-identical across them.)

Three findings:

1. **"Learned the demos, demos lack the skill" is nailed.** Own-draw norms collapse
   from ~3.2 to ~0.4 (D0 level) by step 5000 and stay there. The tall pulls fully
   absorbed their 200 in-region demos while the hard stratum never reliably moved;
   the failure is not under-training — the demos do not contain the missing skill.
2. **The 20k-step recipe specializes hard after step 5000.** At 5000 the model fits
   even never-seen pool demos (ctrl 0.45 — generalization to the shared
   distribution). From there, ctrl norms climb monotonically (1.15 → 1.70 → 2.43,
   ~75% of the way back to the π₀ level and still rising) while own stays flat:
   the fine-tune increasingly memorizes its exact 600-demo mix and drifts away
   from the exchangeable pool distribution. This is the gradient-space signature
   of the draw lottery, and the most plausible amplifier of the ±3.3pp pull noise:
   15k of 20k steps are spent amplifying draw idiosyncrasies.
3. **Best and worst pulls have indistinguishable training dynamics.** gradarm_b_j3
   (+7.8pp, best ever) vs gradarm_a_j3 (−2.7pp, worst ever): own 0.638 vs 0.638,
   ctrl 2.37 vs 2.43 at the end. Outcome differences do not show up anywhere in the
   training signal — they emerge downstream, in closed-loop rollout compounding.

**Actionable hypothesis (testable for free with retained checkpoints):** early
stopping at ~5000 steps should shrink pull-to-pull variance (and, per the
dose-response knee, likely keeps most of the gain). Every pull's 5000-step
checkpoint was retained; evaluating a few of them on frozen E (~5h/eval) would
test it directly. If it holds, the paper gains a practical recipe note: the
noise floor everyone fights is partly self-inflicted by over-long fine-tunes.

## Q2b addendum (2026-08-11): encoding does NOT emerge along the trajectory — LESS closed

The last loophole in the Q0 verdict was that encoding was only measured at the trajectory
START (π₀). `grad_gate_traj.py` re-ran the region gate at steps 5000/10000/19999 of the
tall-arm trajectory (a model trained on 200 in-region demos) and a random-arm control.
Result (`gate_traj/report.json`): split-half contrast AUC stays 0.54–0.59 at every
checkpoint of both trajectories (tall @19999: 0.577 ≈ π₀'s 0.572); best-mode stays at its
shuffle-null floor throughout. The region distinction never becomes gradient-visible —
not at the start, not mid-training, not after the model has fully absorbed the region's
own data. Trajectory-LESS has no signal to exploit at any checkpoint; the 10000/15000
checkpoints carry no further gradient-analysis value.

## Q2c addendum (2026-08-11): style arms — the first gradient-visible arm difference

Absorption archive for the execution-quality arms (style_hi_j3 delta +5.8pp, style_lo_j3
−3.1pp): style_lo's own demos absorb markedly WORSE (residual grad-norm 0.52→0.83 across
training vs style_hi's 0.35→0.57, +46% at the end; ctrl curves identical). After 16
content-arm pulls with indistinguishable dynamics, the quality axis is the first arm
whose treatment differs in the training signal itself — low-quality (inconsistent)
executions are harder to fit, and that arm hurt. One pull per arm → suggestive, but it
matches the G1 SREE finding (inconsistent demos → mode collapse) and marks quality, not
content, as the axis the loss can actually see.

## Q3: no π₀-time statistic of a draw predicts its realized Δ (n=16)

Per-draw stats (mean grad norm, batch coherence, whitened batch separation,
self-similarity, region-cosine) vs realized delta over all 16 completed pulls:
best |Spearman| = 0.23; best within-round Pearson = 0.47 (wsep, driven by the j3
round alone). Sharpest negative: the gradarm clusters are *reproducibly* different
in gradient space (wsep ≈ 0.208 (a) vs ≈ 0.235 (b) in both rounds — the engineered
2.4×-null contrast is real and stable) yet their deltas flipped sign across rounds
(a: −2.7/+1.1; b: +7.8/−1.1). Even a real, replicated batch-level gradient
difference does not survive pull-level noise at B=200.

## Program-level summary (Q0+Q2+Q3 + race + gradarm)

- Q0: the training gradient does not encode the weak region → composition can't
  be targeted through this loss.
- Race (12 pulls): no condition-space arm separates from random.
- Gradarm (4 pulls): maximal gradient-space arm contrast doesn't separate either.
- Q3: nothing measurable at π₀ predicts a draw's outcome.
- Q2: every draw is fully absorbed by 5k steps; late training specializes to the
  draw; best/worst pulls have identical dynamics.

Conclusion: at B=200 retrieval on this task, WHAT you add is irrelevant within the
pool — by mechanism, not just by null result. The remaining levers are quantity,
dose/duration (early stop), and *source* (new data with different content), which
is exactly the prescription-of-new-data direction. For the paper: Q0 is the
teachability gate, Q2 is the absorption exhibit, Q3+gradarm is the "we tried the
learner's own space" control that closes the selection story.
