#!/usr/bin/env bash
# bandit_v1 Task 5: launch the real (~8h) pi_0 fine-tune -- D0-only, full recipe, seed 1000.
#
# The 2 shared H100s were holding other users' jobs at write time (~71GB+77GB used).
# This script polls nvidia-smi every 300s and launches on whichever GPU frees first
# (>=70,000 MiB free), rather than blocking a fixed GPU index. Mirrors the wait-for-GPU
# pattern in /data/xinyua11/tmp/run_coverage_arm.sh; env activation + MUJOCO_GL/WANDB_MODE/
# XLA_PYTHON_CLIENT_MEM_FRACTION copied from /data/xinyua11/tmp/ft.sh. ft.sh does not forward
# --seed, so the training command is invoked directly here (not via ft.sh) with --seed 1000.
#
# Config: pi0_ppc2sink_pi0base (bandit_v1/config.py D0_DATASET ==
# /data/xinyua11/ft_arms/ppc2sink_base_only; TrainConfig added in openpi commit 6e21f87,
# literal clone of pi0_ppc2sink_core with only name+data_dirs changed).
# Checkpoint convention: /data/xinyua11/openpi/checkpoints/pi0_ppc2sink_pi0base/pi0_v1/<step>/
# Final step dir name: see bandit_v1/config.py FINAL_CKPT_STEP (19999).
#
# Run under nohup so it survives session boundaries:
#   nohup bash /data/xinyua11/robocasa/bandit_v1/launch_pi0.sh > /data/xinyua11/tmp/ft_pi0base.log 2>&1 &
# The training subprocess is NOT given its own separate redirect (that would open the same
# ft_pi0base.log path a second time and risk truncating/clobbering the outer nohup redirect's
# fd); it simply inherits this script's stdout/stderr, which the launch command above already
# points at ft_pi0base.log. Orchestrator-only lines (the GPU-wait poll) go to a separate
# launch_pi0.log so the two streams don't interleave.
set -uo pipefail
source /data/xinyua11/conda/etc/profile.d/conda.sh && conda activate openpi
export TMPDIR=/data/xinyua11/tmp HF_HOME=/data/xinyua11/.cache/huggingface MUJOCO_GL=egl
export WANDB_MODE=offline WANDB_DIR=/data/xinyua11/wandb

CONFIG=pi0_ppc2sink_pi0base
EXP_NAME=pi0_v1
SEED=1000
GPUS=(0 1)
TRAIN_NEED=70000   # MiB free required to launch (adjustment #2: poll for >=70,000 MiB free)
POLL_SECS=300

ORCH_LOG=/data/xinyua11/tmp/launch_pi0.log
FT_LOG=/data/xinyua11/tmp/ft_pi0base.log

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$ORCH_LOG"; }
free_mb(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' '; }

log "=== pi0base launcher START (pid $$) ==="

# 1) poll every 300s across both GPUs; pick whichever frees first (>=70,000 MiB free)
GPU=-1
while [ "$GPU" -lt 0 ]; do
  for g in "${GPUS[@]}"; do
    f=$(free_mb "$g")
    log "gpu${g} free=${f:-0}MB (need >=${TRAIN_NEED})"
    if [ "${f:-0}" -ge "$TRAIN_NEED" ]; then GPU=$g; break; fi
  done
  [ "$GPU" -lt 0 ] && sleep "$POLL_SECS"
done
log ">>> gpu${GPU} has >=${TRAIN_NEED}MiB free -- launching TRAINING on gpu${GPU} (output -> $FT_LOG via the outer nohup redirect)"

# 2) launch the real training run on the freed GPU (inherits this script's stdout/stderr)
cd /data/xinyua11/openpi
CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  python scripts/train.py "$CONFIG" --exp-name "$EXP_NAME" --seed "$SEED" --overwrite
rc=$?
log "=== FT DONE $CONFIG $EXP_NAME EXIT $rc ==="
