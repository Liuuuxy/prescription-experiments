# Gradient-Analysis Handoff (written 2026-07-29, mid-race)

For a fresh session doing gradient/influence analysis on the Bandit v1 artifacts. Everything here was
learned the hard way during the race; read the Gotchas before touching anything.

## 1. What exists for you (retained BY DESIGN for gradient analysis)

Every finetune of the experiment kept everything needed to recompute per-demo gradients later:

- **Checkpoints, all save intervals** (5000/10000/15000/19999), never deleted:
  `/data/xinyua11/openpi/checkpoints/pi0_ppc2sink_bandit_{a,b}/<pull_id>/<step>/params` (orbax).
  `pull_id` examples: `null_j1`, `tall_vessel_grasp_fail_j3`, `mid_band_j4`. π₀ itself:
  `pi0_ppc2sink_pi0base/pi0_v1/19999`. Slot a vs b is bookkeeping only — find a pull's slot from its
  ledger row (`training_artifacts.ckpt_root`), don't guess (slots swapped once mid-race; see Gotchas).
- **The exact training datasets**, materialized per pull: `/data/xinyua11/ft_arms/ppc2sink_bandit_<pull_id>/`
  (LeRobot; D0's 400 episodes + the pull's 200 drawn demos). D0-only: `ppc2sink_base_only`.
- **The ledger** (single source of truth): `/data/xinyua11/robocasa/bandit_v1/ledger/`
  - `pulls.parquet`: one row per pull attempt. Use `status=='ok'` rows only. Columns you need:
    `pull_id, arm, round_j, seed` (training seed = 1000+round, paired across arms within a round),
    `demo_ids` (the 200 pool episode indices), `delta` (success − baseline 0.5133),
    `training_artifacts` (dict: ckpt_root, ckpt_steps_present, train_log_path, wandb_dir, recipe_seed).
  - `episodes.parquet`: every rollout ever (diagnosis 2400, evals 450/checkpoint) with per-episode
    success, 5-stage failure signature, condition features. `phase` ∈ diag/eval; join eval rows to
    strata via `E_manifest.parquet`.
  - `pool_demos.parquet`: all 9,885 pool demos with recovered knobs (category canonicalized, h, w,
    x_rel, y_rel, side:int) + `in_d0`.
  - Frozen: `map_models.joblib` (p̂/p_stage; load via `bandit_v1.map_fit.load()` — NOT raw joblib, see
    Gotchas), `arms.yaml` (behavior-only k=3 + z_spec + hashes), `E_manifest.parquet`.
- **Read the ledger** with `import sys; sys.path.insert(0,'/data/xinyua11/robocasa'); from bandit_v1 import ledger; ledger.read('pulls')`
  (env: `/data/xinyua11/conda/envs/robocasa/bin/python`).

## 2. Prior gradient/influence art in this repo (start here, don't reinvent)

- `policy_analysis/influence_score.py` — the LESS/TracIn-style machinery from the influence arms.
  Validated recipe: **LoRA-restricted gradients + contrastive score (mean_hard − mean_ref)** passed
  smoke (AUC 0.63); plain last-layer and per-instance variants FAILED. Core principle (validated 3×,
  see memory `influence-gated-by-gradient-encoding`): influence selection works iff the loss gradient
  encodes the target — check that FIRST for any new gradient signal.
- Loading a checkpoint for gradient compute: openpi env (`/data/xinyua11/conda/envs/openpi/bin/python`),
  TrainConfig `pi0_ppc2sink_bandit_a` / `_b` / `_pi0base` (identical LoRA recipe; only data_dirs differ),
  weight init irrelevant — load the orbax `params` dir directly.

## 3. Conventions this project enforces (violating them corrupts the live race)

1. **The race may still be RUNNING** (runner `pgrep -xf ".../python -u -m bandit_v1.run_race"`; check
   before any GPU work). Both H100s are usually busy (training 0.9 fraction, or eval servers 0.25 +
   workers). Gradient jobs: use `XLA_PYTHON_CLIENT_MEM_FRACTION=0.15–0.2`, prefer idle windows, and
   NEVER kill processes you didn't start.
2. **Ledger is append-only via `ledger.append_rows` ONLY** (it holds a cross-process flock;
   `ledger/<table>.lock`). Never write parquets in `bandit_v1/ledger/` directly — a raw write races the
   runner and can silently lose its rows (this bit us; the lock landed after). Analysis output goes in
   your OWN directory (e.g. `gradient_analysis/`), not the ledger.
3. **Tests must never touch the real ledger/config.yaml** — a conftest tripwire fails the suite if they
   do. Pass tmp paths for everything (six tests once wrote demo values into the real config repeatedly;
   it looked like sabotage until root-caused).
4. Frozen artifacts (map, arms.yaml, E, D0, recipe) are load-only. Re-fitting/re-clustering mid-race
   invalidates the experiment.

## 4. Gotchas (each one cost us hours; read all)

- **`map_models.joblib` must be loaded via `bandit_v1.map_fit.load()`** — the pickle references classes
  by module path; raw `joblib.load` from a script may hit `__main__.MapModels` issues (fixed once; if you
  re-save anything, save from an imported-module context, never from a `__main__` CLI).
- **GL env leakage**: robosuite/mujoco imports export `PYOPENGL_PLATFORM` into `os.environ`; any child
  process you launch into the *openpi* env must scrub it (`env.pop("PYOPENGL_PLATFORM")`) or mujoco's
  import raises. `MUJOCO_GL=egl` for anything rendering; training/serving children get it set explicitly.
- **The mujoco EGLError at process exit is DESTRUCTOR NOISE**, not a render failure. Trust the
  "RENDER OK"/result line, never the stderr tail. (We mis-diagnosed EGL as broken for a day.)
- **JAX preallocation fails fast** — a train/serve that doesn't fit its memory fraction dies in the first
  minutes, not at hour 5. Use that: launch and watch 10 min. `BANDIT_TRAIN_MEM_FRACTION` overrides the
  train fraction (default 0.9); serving uses 0.25.
- **`setsid nohup ... &` wrapper PIDs lie** — `$!` is the dead wrapper; find the real process with
  `pgrep -xf "<full python cmdline>"`. This burned us 4 times (watchdogs watching corpses, signals to
  ghosts).
- **Bracket your pgrep/pkill patterns** (`pgrep -f "[s]erve_policy"`) — unbracketed patterns match your
  own shell/monitor command line; one unbracketed pkill killed our own shell mid-operation.
- **Thread oversubscription**: unset OMP/MKL/etc. defaults spawn ~322 threads/process on this 128-core
  box. Export `OMP_NUM_THREADS=4` (and friends) in anything you fan out; `parallel_eval._worker_env()`
  shows the full set.
- **Never wait forever, never fail silently**: every poll loop needs a timeout and a log line. Two
  multi-hour stalls came from silent infinite waits (`wait_for_checkpoint` pre-timeout; a claimability
  check reading one stale glance). For stacks from a live process: it must have been launched with
  `PYTHONFAULTHANDLER=1`, then `kill -ABRT` (py-spy is blocked by ptrace perms w/o sudo).
- **openpi `--overwrite` rmtree-deletes existing checkpoint dirs.** Never launch a training whose
  exp_name matches an existing dir unless you MEAN to destroy it. `pull.checkpoint_looks_complete()`
  (`_CHECKPOINT_METADATA` + non-empty `params/`) is the completeness test.
- **GPU claimability right after killing a process is stale** for ~10–30 s (memory still releasing) —
  don't make one-shot decisions on it.
- **Parallel rollout workers**: 4-worker evals work NOW (thread caps + unbuffered logging + ledger lock
  + a supervised pass), but the mode hung mysteriously twice before those fixes — if you fan out env
  workers, keep per-worker timeouts and progress lines, and treat "no progress lines in 12 min" as hung.
- `nvidia-smi` works (post-reboot). If NVML ever mismatches again (userspace update under a running
  kernel module), CUDA/JAX keep working; guards must tolerate nvidia-smi failure; reboot fixes it.

## 5. Current race state (as of writing — check the ledger for truth)

Round 3 complete: mid_band +4.0pp, random +4.0pp (control TIED the best selected arm), tall_vessel
+2.2pp, easy_band +0.9pp vs baseline 51.3%, noise floor σ_e=3.3pp (nulls +2.4/−0.9). Round 4 training.
The interesting gradient question this raises: WHY did 200 tall-vessel grasp-failure demos not move the
hard stratum (−2.7pp on its own target)? Per-demo gradient alignment between arm demos and the target
stratum's failure modes is exactly the analysis the retained artifacts enable.

## 6. Pointers

- Design: `weakregion/BANDIT_V1_DESIGN.md` · Walkthrough: `weakregion/BANDIT_V1_WALKTHROUGH.md`
- Engineering postmortems: `.superpowers/sdd/task-*-report.md` (gitignored, on disk)
- Memory index entries: `bandit-v1-race-state`, `influence-gated-by-gradient-encoding`,
  `pi0-influence-arm`, `pi0-value-influence-arm`
