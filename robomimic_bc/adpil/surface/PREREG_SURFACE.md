# PRE-REGISTRATION: ADPIL A1.3 BC-RNN response-surface refresh (2026-08-31)

Purpose: design-input measurement block (NOT confirmatory evidence). Re-grounds the
Can-PH response model on the learner the protocol actually mandates (BC-RNN) under
the v2 recipe, and measures the two noise parameters the §14.2 floor is missing.
Authorized by the owner's continuation directive of 2026-08-31 (see
../AUTHORIZATION_LOG.md); scope limited to this block.

## Design

Learner/recipe: BC-RNN (rnn.enabled, seq_length 10, actor_layer_dims ()), v2 recipe
(500 epochs, ckpt every 50, best-of from epoch 250 on the stratified 50-scene
E_probe subset, evaluated on E_probe + E_test at h500/h400, equal-per-trajectory
weighting). Driver run_surface.py = vardecomp2/run_vd2.py + capability/run_cap.py
RNN config, output under adpil/surface/out.

16 cells x 3 seeds (200, 201, 202) = 48 runs, all masks pre-existing in
can_ph_work.hdf5 (no new data construction):

- Response surface (11 cells): balanced_D0; balanced_add_{xhi_yhi, xhi_ylo,
  xlo_yhi, xlo_ylo}; starved_{xhi_ylo, xlo_yhi}_D0; starved_xhi_ylo_add_xhi_ylo;
  starved_xlo_yhi_add_xlo_yhi; starved_xhi_ylo_add_xlo_ylo (cross);
  starved_xlo_yhi_add_xhi_yhi (cross).
- Draw variance (5 cells): vd_N80_d{0..4} (five independent 80-demo draws).

## Frozen quantities to extract (analysis: analyze_surface.py, written before unblinding)

1. beta_own(balanced): mean over 4 zones of [J_region(add_z) - J_region(D0)] / 24,
   E_test h500.
2. beta_cross(balanced): mean cross-zone response from the same add cells.
3. beta_own(starved) and beta_cross->starved: same contrasts on the starved cells;
   concavity ratio = beta_own(starved) / beta_own(balanced).
4. sigma_seed(RNN, v2): pooled within-cell seed sd over all 16 cells (32 df),
   J_deploy on E_test h500.
5. sigma_draw(RNN, v2): sd over the 5 vd_N80 cell means minus seed-noise share
   (nested ANOVA as in vardecomp2), 4 df on draws — reported with that df caveat,
   used alongside (not replacing) the MLP 3.6 pp figure.

These five numbers feed the A1 analytic oracle and the recomputed floor. No
selector, allocation, or exam-tilt decision is taken from any other statistic of
this block. Per-cell J values will be reported in full regardless of outcome.

## What this block may NOT do

No MDE below 10 df from the draw estimate; no selector tuning; no comparison
labeled confirmatory; no touching of any future confirmatory sealed exam
(E_test here is the PRIOR program's sealed set, already unblinded in earlier
experiments and used strictly as a design-input instrument).
