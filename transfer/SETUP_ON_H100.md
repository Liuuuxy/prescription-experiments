# Bootstrap on the H100 (no shared filesystem)

This `robocasa_experiments/` repo carries everything *small* (code, results, patches,
memory, the weak-region finding, the H100 handoff). The big stuff (checkpoints/assets/
datasets) is re-downloaded on the H100 — it's all public on HuggingFace / via robocasa
scripts, so don't transfer 28GB.

## On the H100
```sh
git clone <your-private-repo-url> robocasa_experiments
cd robocasa_experiments
bash transfer/apply_patches.sh      # clones the public forks into ~ and applies the 3 patch sets,
                                    # places the standalone scripts, and installs the Claude memory
```
Then follow **`H100_HANDOFF.md`** (env setup + downloads + the training/experiment plan).
A fresh Claude Code session on the H100 will pick up the memory installed by the script and
have full context of today's work.

## What's in here
- `H100_HANDOFF.md` — the plan + DP training setup + experiment steps (read this).
- `results_log.md`, `model_triage.md` — results.
- `policy_analysis/` — the weak-region detector + triage tool (analyze_pi0_weakregions.py, compare_models.py, analysis.py, …).
- `weakregion/` — pi0's weak-region report (the concrete targeting signal).
- `mimicgen_src/` — example pi0 source demos + the validated LeRobot dataset (regenerable).
- `transfer/patches/` — the 5 fork patches (diffusion_policy ×3, robomimic ×1, mimicgen ×2 geom fixes).
- `transfer/new_files/` — standalone scripts that live inside the forks.
- `transfer/memory/` — copy of the Claude memory (installed by apply_patches.sh).

## Alternatives to GitHub (if the two boxes can reach each other)
- `rsync -avz ~/robocasa_experiments/ user@h100:~/robocasa_experiments/` (and the forks + memory) — simplest if SSH between boxes works; handles big files too.
- HuggingFace Hub for the LeRobot datasets (native robocasa/lerobot workflow).
- Shared cluster `/scratch` or `/project` storage, if your HPC has one even when `$HOME` isn't shared.
