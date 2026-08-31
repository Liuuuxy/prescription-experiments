# robomimic BC experiments

Scripts for the robomimic behavior-cloning (BC / BC-RNN) experiment line —
data-prescription questions tested on robomimic tasks (Lift, Can, Square),
where training is cheap enough for many seeds.

**These are COPIES, snapshotted 2026-08-31.** The working directory — with
datasets, checkpoints, run outputs, logs, and any live runs — is
`/data/xinyua11/robomimic_runs/`. Scripts there remain the source of truth for
anything still executing; re-snapshot before committing if they change. Only
`*.py`, `*.sh`, `*.md`, and `bc_lift_config.json` were copied (run output
directories `out/`, `run_*/`, `srf_*/` excluded).

## Layout

- `run_bc.py` + `bc_lift_config.json` — the base BC training entrypoint
  (Lift smoke test that validated the stack).
- `prescribe/` — Can-MH (multi-human) prescription sandbox: operator-quality
  splits, scoring, refinement. NOTE: its "better>worse ordering" claim was
  RETRACTED 2026-08-25 (weighting artifact + seed luck).
- `prescribe_ph/` — Can-PH (proficient-human) region benchmark: region-targeted
  vs random demo addition, MDE computation, C9 confirmation, and `nullcheck/`
  (the 13-script refutation battery that retracted Gate 1).
- `prescribe_sq/` — Square heterogeneity screen (CLOSED: pooled 6-seed fail;
  response slope +0.35pp/demo replicates).
- `selection/` — demo-selection baselines: DemInf scoring, CUPID
  (`cupid/record_rollouts.py`, `cupid_score.py`), DPP selection.
- `vardecomp/`, `vardecomp2/` — draw-effect variance decomposition v1/v2
  (seed vs draw vs recipe; v2 = paired paper-recipe version that found
  recipe +7.7pp and made checkpoint selection the cheapest fix).
- `capability/` — capability probes (BC-RNN instrument validation).
- `convergence/` — training-convergence checks.
- `variance/` — rollout/seed variance measurement.
- `xtask/` — cross-task merge probe.
- `adpil/` — the ADPIL active-learning protocol: protocol abstract, amendment
  drafts, authorization log, `stage0/` (ceiling/floor arithmetic), and
  `surface/` (the balanced-D0 response-surface runs).
