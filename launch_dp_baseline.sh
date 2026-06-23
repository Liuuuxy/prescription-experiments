#!/usr/bin/env bash
# Launch ONE Diffusion-Policy baseline seed for PickPlaceCounterToSink (single-task,
# 108 human pretrain demos). Built 2026-06-11; see Claude memory h100-setup-complete
# and first-experiment-pickplace-sink. Plan: 3 seeds x 500 epochs to measure sigma_seed.
#
# Usage:   bash launch_dp_baseline.sh <seed> <gpu_index> [num_epochs=500]
# Example: bash launch_dp_baseline.sh 0 0          # seed 0 on physical GPU 0
#          bash launch_dp_baseline.sh 1 1          # seed 1 on physical GPU 1
#
# Refuses to start unless the target GPU has >= MINFREE_MIB free (default 40GB),
# because the box is shared (ssagar6's VLLM jobs can saturate both H100s).
set -euo pipefail

SEED="${1:?usage: $0 <seed> <gpu_index> [num_epochs]}"
GPU="${2:?usage: $0 <seed> <gpu_index> [num_epochs]}"
EPOCHS="${3:-500}"
MINFREE_MIB="${MINFREE_MIB:-40000}"

source /data/xinyua11/conda/etc/profile.d/conda.sh && conda activate robocasa
export TMPDIR=/data/xinyua11/tmp
export HF_HOME=/data/xinyua11/.cache/huggingface
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")
if [ "$free" -lt "$MINFREE_MIB" ]; then
  echo "ABORT: GPU $GPU has only ${free} MiB free (< ${MINFREE_MIB}); not launching seed $SEED." >&2
  echo "       (set MINFREE_MIB lower to override, but DP bs192 wants ~24GB+)" >&2
  exit 3
fi
echo ">>> GPU $GPU has ${free} MiB free -> launching baseline seed $SEED, ${EPOCHS} epochs"

OUT="/data/xinyua11/dp_runs/pickplacesink_human/seed${SEED}"
mkdir -p "$OUT"
cd /data/xinyua11/diffusion_policy
CUDA_VISIBLE_DEVICES="$GPU" python train.py \
  --config-name=train_diffusion_transformer_bs192 \
  task=robocasa/pickplacesink_human \
  training.seed="$SEED" \
  training.num_epochs="$EPOCHS" \
  training.device=cuda:0 \
  dataloader.num_workers=16 val_dataloader.num_workers=8 \
  dataloader.persistent_workers=True val_dataloader.persistent_workers=True \
  +dataloader.prefetch_factor=4 \
  logging.mode=offline \
  logging.name="dp_pickplacesink_seed${SEED}" \
  hydra.run.dir="$OUT" \
  > "$OUT/train.log" 2>&1
echo ">>> seed $SEED finished. Outputs + checkpoints under $OUT/"
