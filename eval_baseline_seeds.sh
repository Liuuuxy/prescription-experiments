#!/usr/bin/env bash
# Offline-eval the baseline DP seeds to get per-seed success rate -> sigma_seed.
# Runs n=50 rollouts on PickPlaceCounterToSink (pretrain split) for each seed's
# checkpoint at a common epoch. Seeds occupy both GPUs while training, so run
# this only AFTER training finishes. Usage: bash eval_baseline_seeds.sh [epoch=0400]
set -uo pipefail
EPOCH="${1:-0400}"
N="${2:-50}"
source /data/xinyua11/conda/etc/profile.d/conda.sh && conda activate robocasa
export TMPDIR=/data/xinyua11/tmp HF_HOME=/data/xinyua11/.cache/huggingface
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/xinyua11/diffusion_policy

BASE=/data/xinyua11/dp_runs/pickplacesink_human
declare -A DIRS=(
  [0]="$BASE/seed0_noncached_partial_ep471"
  [1]="$BASE/seed1"
  [2]="$BASE/seed2"
)
GPU=0
for s in 0 1 2; do
  ckpt=$(ls "${DIRS[$s]}/checkpoints/"epoch=${EPOCH}*.ckpt 2>/dev/null | head -1)
  [ -z "$ckpt" ] && ckpt="${DIRS[$s]}/checkpoints/latest.ckpt"
  if [ ! -f "$ckpt" ]; then echo "seed $s: NO checkpoint for epoch $EPOCH (skipping)"; continue; fi
  out="$BASE/eval/seed${s}_ep${EPOCH}"
  mkdir -p "$out"
  echo ">>> eval seed $s on GPU $GPU: $ckpt -> $out"
  CUDA_VISIBLE_DEVICES=$GPU python run_dp_smoketest.py \
    -c "$ckpt" -t PickPlaceCounterToSink -s pretrain -n "$N" -e 5 -d cuda:0 -o "$out" \
    > "$out/eval.log" 2>&1 &
  GPU=$((1-GPU))   # alternate the two GPUs
  sleep 30         # stagger startup
done
wait
echo "=== ALL EVALS DONE ==="
for s in 0 1 2; do
  j=$(find "$BASE/eval/seed${s}_ep${EPOCH}" -name eval_log.json 2>/dev/null | head -1)
  [ -n "$j" ] && echo "seed $s:" && grep -E "success_rate|mean_score" "$j"
done
