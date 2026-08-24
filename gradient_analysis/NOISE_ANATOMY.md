# Anatomy of the ±3.3pp noise floor (2026-08-14)

Every component of this project's per-pull noise is now measured, all from artifacts that
already existed. Terms: a **pull** = fine-tune the frozen base policy π₀ on its 400 base
demonstrations plus 200 drawn ones, then evaluate on the frozen eval set **E** (150 saved
scene states × 3 repeats = 450 rollouts); **Δ** = that policy's success rate minus the
baseline 51.33%; **σ_e** = 3.33pp, the per-pull standard deviation measured from two null
pulls (identical data, different seed).

## The four layers

| layer | what is held fixed | measured spread | source |
|---|---|---|---|
| 1. rollout / eval stochasticity | **everything**, including the weights (bit-identical policies) | **0.88pp** in Δ; per-episode success agreement only **0.671** | determinism replica: `replica_random_j106` vs `random_j106`, same 200 demo ids and same seed 1106 (`replica_test.py`, `replica_anchor.json`) — the two weight updates are cosine **1.0000**, ‖Δθ‖ 108.037 both |
| 2. checkpoint choice within one run | data, seed, recipe — only the stopping point differs | up to **2.4pp**, non-monotone | training-length ablation (below) |
| 3. training seed | data, recipe, checkpoint | **3.3pp** (null pulls +2.44 / −0.89) | the two null pulls; weight updates only 0.505 cosine-aligned (Q5) |
| 4. data composition (what we were trying to measure) | seed, recipe, checkpoint | ~**1.5%** of the weight update; arm means spread 1.6pp against ±1.9pp standard errors | Q5 weight-space decomposition; the 12-pull race |

**The ordering is the finding: the thing we wanted to measure (layer 4) is the smallest
term, and it sits underneath three sources of nuisance variation, two of which are
introduced by the experimental protocol itself rather than by the world.**

## Layer 2 in detail: the same run scores differently depending on where you stop

Two pulls had their 10000- and 15000-step checkpoints evaluated on the same frozen E
(450 rollouts each). Same data, same seed, same recipe — only the stopping point differs.

| pull | @10000 | @15000 | @19999 (final) |
|---|---|---|---|
| mid_band_j5 | +2.45pp | +1.11pp | +3.11pp |
| random_j5 | +2.00pp | −0.44pp | +0.45pp |

Paired per-start comparison of final vs 10000 steps, bootstrap over the 150 shared scene
states: mid_band_j5 **+0.67pp [95% CI −4.89, +6.44]**, random_j5 **−1.56pp [−7.11,
+3.78]**. Both intervals contain zero by a wide margin: **the second half of training buys
nothing measurable**, which is why the production recipe moved to 10k steps. But the
non-monotone wandering (+2.45 → +1.11 → +3.11) is the sharper observation: **choosing a
different checkpoint of the *same* run moves the reported effect by about as much as
choosing a different arm.**

## Consequences

1. **A single-run, single-checkpoint arm comparison is uninterpretable in this regime.**
   Layers 1–3 are all comparable to or larger than the effect being estimated. The race's
   paired-seed design cancels layer 3 within a round; nothing in the protocol cancels
   layer 2 except fixing the checkpoint by convention (which we do — always 19999, or 9999
   for the 10k recipe).
2. **Halving training cost is free.** 10k steps is statistically indistinguishable from
   20k on both pulls tested, and halves the 9h training bill.
3. **Evaluation stochasticity is not the bottleneck.** Layer 1 (0.88pp) is the smallest
   nuisance term, so buying more rollouts per policy has limited return; the leverage is
   in more seeds, not more episodes — the opposite of the field's usual instinct.
4. **For the paper:** report σ_e with this decomposition rather than as a single opaque
   number. It converts "our effects were within noise" from an apology into a measured
   statement about where the noise comes from — and layers 1 and 2 are protocol facts that
   apply to every VLA fine-tuning comparison, not just ours.

## Provenance

Layer 1: `/data/xinyua11/tmp/replica_test.log`, `gradient_analysis/replica_test.py`,
`gradient_analysis/replica_anchor.json`. Layer 2: ledger `episodes` rows with policy_id
`ablate_{mid_band,random}_j5_ck{10000,15000}` (450 episodes each), analyzed here for the
first time. Layer 3: `null_j1`/`null_j2` ledger rows; Q5 weight-space analysis
(`theory_demo_report.json`). Layer 4: `Q2_REPORT.md`, `q3_pull_ablation.json`, the 12-pull
race table.
