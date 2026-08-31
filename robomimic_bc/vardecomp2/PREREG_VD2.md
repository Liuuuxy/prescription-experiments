# PRE-REGISTRATION: variance decomposition under the robomimic-paper recipe (2026-08-28)
Owner-specified rerun of vardecomp with the paper-style training/eval protocol:
500 epochs; checkpoints every 50; from epoch 250 on, each checkpoint is evaluated on a
50-scene selection set; the BEST checkpoint (highest selection score; tie -> earliest)
is the run's model, instead of the final one.

## Protocol details (frozen)
- Same 20 masks (N in {10,40,80,120} x 5 draws) and same seeds (0,1,2) as vardecomp v1 ->
  all 60 cells PAIRED with v1 for a direct recipe-effect readout.
- Learner: plain BC-MLP, equal-per-trajectory weighting, horizon 500, unchanged.
- Selection set: a stratified 50-scene subset of E_probe (first 12/13 per zone) — the
  sealed E_test is NEVER used for selection. Both selection scores (all 6 checkpoints)
  and the best checkpoint's full probe+sealed evaluations are recorded.

## Analyses (frozen)
1. Same nested decomposition as v1 (sigma_seed, sigma_draw, pooled F) on the sealed score
   of the best checkpoint. Question: does the paper recipe change the verdict?
2. Paired recipe effect per cell: best-of-500 minus last-of-300 (v1), clustered by draw.
3. Selection inflation: best checkpoint's 50-scene selection score minus its own sealed
   200-scene score, distribution over 60 runs — a direct measurement of how much
   best-checkpoint reporting flatters, under otherwise identical conditions.
