# CIFAR experiments

All CIFAR-100 sandbox work in one place: the experiments that mirror the robot
data-prescription questions in a domain where fine-tune pulls are cheap enough
to afford real statistics. Collected here 2026-08-31 from `sandbox_regions/`
(repo root), `gradient_analysis/`, `policy_analysis/`, and
`/data/xinyua11/xgradtest/`.

## Layout

### `sandbox_regions/` (moved from repo root; hardcoded paths updated)
The CIFAR-100 accuracy-region sandbox: an imbalanced base model creates a
per-class accuracy spread (easy / medium / hard bands), then arms add
budget-matched images and we measure per-band deltas.

- `region_sandbox.py` — the base sandbox: band-targeted vs random vs null arms,
  6 seeds each; also saves lossless per-sample last-layer gradients for the pool.
- `mixture_sweep.py` — band-mixture ratio ladders (is there a mix that beats random?).
- `sandbox_ucb.py` — UCB1 bandit over gradient-cluster arms with five different
  rewards (raw / balanced / hard / loss-surrogate / composite).
- `objtest_cifar.py` — CIFAR analogue of the robot clean-object test
  (add images of worst classes vs best classes vs random).
- `proxy_continuum.py` — allocator simulation: best-arm identification with a
  free-but-possibly-corrupted proxy (built on measured task-1 landscapes).
- `kappa_frontier.py` — kappa sweep for the calibrated proxy gate (frozen
  allocators; reporting only).
- Untracked artifacts (`*.npz`, `*.pt`, `*.parquet` — see `.gitignore`):
  `base_ckpt.pt`, `pool_grads.npz`, results parquets.

### `robot_mirrors/` (moved from `gradient_analysis/` and `policy_analysis/`)
CIFAR mirrors of robot-side analyses — same statistics code, run where ground
truth is knowable.

- `cifar_gate.py` → `cifar_gate_report.json` — the Q0 gradient-encoding-gate
  statistics computed on the CIFAR sandbox (imports `analyze_gate.py`, which
  stays in `gradient_analysis/`). Reads gradlogs from `/data/xinyua11/xgradtest/`.
- `whiten_cifar.py` — whitening k-sweep on the CIFAR gradlog (generalization
  check of the robocasa whitening win; imports `influence_offline.py`, which
  stays in `policy_analysis/`).
- `plot_cluster_class_race.py` + `cifar_cluster_class_race.png`,
  `cifar_cluster_class_table.csv/.png`, `cifar_per_class_acc.csv` — the
  cluster-vs-class race figure (Q4 mirror).

### `xgradtest_runs/` (COPIES — originals stay at `/data/xinyua11/xgradtest/`)
The run-side scripts of the Q4 CIFAR mirror (arm race with 20 artificially-rare
classes as the planted "weak region"). Only scripts are copied; the data,
checkpoints, gradlogs, and armrace results live in `/data/xinyua11/xgradtest/`,
which remains the working directory for these.

- `xgrad.py`, `grad_geometry.py` — gradient logging / geometry on the testbed.
- `xarm_race.py` — the arm race that produced the Q4 positive control
  (rare +3.8pp vs random +0.9pp vs null ±0.4pp).
- `cifar_control.py`, `verify_cifar_control.py`, `verify_null.py`,
  `verify_robust.py`, `cluster_improvement.py` — controls and verification.

## What deliberately stays elsewhere

- `xgradtest/` at the repo root — the collaborator's standalone clone of
  RansML/xgradtest (gitignored, not part of this repo). `xgradtest_runs/` above
  are the scripts that ran *on* that testbed.
- `gradient_analysis/Q4_CIFAR_MIRROR_REPORT.md` — stays with the Q0–Q4 report
  series in `gradient_analysis/`.
- `gradient_analysis/llm_borrow/*cifar*` — the LLM-borrow wind tunnel runs on
  this sandbox but is a coherent suite of its own; left intact.
