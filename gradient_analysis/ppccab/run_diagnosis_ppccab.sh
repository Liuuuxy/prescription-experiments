#!/usr/bin/env bash
# Task-2 (PickPlaceCounterToCabinet) diagnosis orchestrator — clone of
# bandit_v1/run_diagnosis.sh with: ppccab checkpoint/config/parquet/logs,
# port 8134, BANDIT_TASK_PROFILE=ppccab on every bandit_v1-importing step,
# and the parallel rollout driver (gradient_analysis/ppccab/run_diag_parallel.py,
# workers=4) instead of the serial chunked module. Waiting mode: polls for
# BOTH the base checkpoint (training on GPU0) and the 300-row
# diag_conditions.parquet (condition scan), then serves + rolls out.
#
#   nohup bash /data/xinyua11/robocasa/gradient_analysis/ppccab/run_diagnosis_ppccab.sh \
#     > /data/xinyua11/tmp/ppccab_diag_orch.log 2>&1 &
set -uo pipefail

REPO=/data/xinyua11/robocasa
OPENPI_DIR=/data/xinyua11/openpi
OPENPI_PY=/data/xinyua11/conda/envs/openpi/bin/python
ROBOCASA_PY=/data/xinyua11/conda/envs/robocasa/bin/python

CKPT_DIR=/data/xinyua11/openpi/checkpoints/pi0_ppccab_bandit_a/ppccab_base/9999
PARQUET="$REPO/bandit_v1/ledger_ppccab/diag_conditions.parquet"
N_DIAG=300

PORT=8134
POLL_SECS=120
PORT_WAIT_TRIES=90
PORT_WAIT_SLEEP=5

SERVE_MAX_ATTEMPTS=6
SERVE_RETRY_SLEEP=300
GPU0_MIN_FREE_MB=20000
GPU_WAIT_POLL_SECS=60
GPU_WAIT_MAX_SECS=14400

ORCH_LOG=/data/xinyua11/tmp/ppccab_diag_orch.log
SERVE_LOG=/data/xinyua11/tmp/ppccab_diag_serve.log
ROLLOUT_LOG=/data/xinyua11/tmp/ppccab_diag_rollouts.log

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$ORCH_LOG"; }
free_mb(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' '; }

log "=== ppccab diagnosis orchestrator START (pid $$) -- waiting mode ==="

parquet_has_300_rows(){
  [ -f "$PARQUET" ] || return 1
  "$ROBOCASA_PY" -c "
import sys
import pandas as pd
df = pd.read_parquet('$PARQUET')
sys.exit(0 if len(df) == $N_DIAG else 1)
" 2>/dev/null
}

while :; do
  ck=no; [ -d "$CKPT_DIR" ] && ck=yes
  pq=no; parquet_has_300_rows && pq=yes
  log "waiting: checkpoint=${ck}  diag_conditions(==${N_DIAG} rows)=${pq}"
  if [ "$ck" = yes ] && [ "$pq" = yes ]; then break; fi
  sleep "$POLL_SECS"
done
log ">>> both preconditions satisfied -- serving ppccab base"

cd "$OPENPI_DIR" || { log "FATAL cd $OPENPI_DIR"; exit 4; }

served=no
gpu_wait_elapsed=0
attempt=1
while [ "$attempt" -le "$SERVE_MAX_ATTEMPTS" ]; do
  while :; do
    f=$(free_mb 0)
    if [ "${f:-0}" -ge "$GPU0_MIN_FREE_MB" ]; then break; fi
    log "gpu0 free=${f:-0}MiB < ${GPU0_MIN_FREE_MB}MiB -- waiting (elapsed ${gpu_wait_elapsed}s / cap ${GPU_WAIT_MAX_SECS}s)"
    if [ "$gpu_wait_elapsed" -ge "$GPU_WAIT_MAX_SECS" ]; then
      log "!!! GPU0 memory wait exceeded cap -- aborting"; exit 2
    fi
    sleep "$GPU_WAIT_POLL_SECS"
    gpu_wait_elapsed=$((gpu_wait_elapsed + GPU_WAIT_POLL_SECS))
  done

  log ">>> serve attempt ${attempt}/${SERVE_MAX_ATTEMPTS}: pi0_ppccab_bandit_a from $CKPT_DIR on :$PORT (gpu0 free=${f}MiB)"
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 MUJOCO_GL=egl \
    TMPDIR=/data/xinyua11/tmp HF_HOME=/data/xinyua11/.cache/huggingface \
    "$OPENPI_PY" scripts/serve_policy.py --port "$PORT" policy:checkpoint \
    --policy.config pi0_ppccab_bandit_a --policy.dir "$CKPT_DIR" \
    > "$SERVE_LOG" 2>&1 &
  SV=$!
  log ">>> server pid=$SV, waiting for :$PORT"

  up=no
  for i in $(seq 1 "$PORT_WAIT_TRIES"); do
    if "$ROBOCASA_PY" -c "
import socket
s = socket.socket(); s.settimeout(1)
s.connect(('127.0.0.1', $PORT)); s.close()
" 2>/dev/null; then
      up=yes; log ">>> server up after ${i}x${PORT_WAIT_SLEEP}s"; break
    fi
    if ! kill -0 "$SV" 2>/dev/null; then
      log "!!! SERVER DIED waiting for port (attempt ${attempt}) -- tail of $SERVE_LOG:"
      tail -30 "$SERVE_LOG" >> "$ORCH_LOG"
      break
    fi
    sleep "$PORT_WAIT_SLEEP"
  done

  if [ "$up" = yes ]; then served=yes; break; fi
  kill "$SV" 2>/dev/null; sleep 2; kill -9 "$SV" 2>/dev/null || true
  attempt=$((attempt + 1))
  if [ "$attempt" -le "$SERVE_MAX_ATTEMPTS" ]; then
    log "!!! serve attempt failed -- retrying in ${SERVE_RETRY_SLEEP}s"
    sleep "$SERVE_RETRY_SLEEP"
  fi
done

if [ "$served" != yes ]; then
  log "!!! all serve attempts failed -- aborting"; exit 2
fi

cd "$REPO" || { log "FATAL cd $REPO"; exit 4; }
log ">>> rollouts: run_diag_parallel --port $PORT (progress -> $ROLLOUT_LOG)"
BANDIT_TASK_PROFILE=ppccab MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  "$ROBOCASA_PY" -u gradient_analysis/ppccab/run_diag_parallel.py \
  --host 127.0.0.1 --port "$PORT" >> "$ROLLOUT_LOG" 2>&1
RC=$?
log ">>> rollout driver exited rc=$RC"

log ">>> killing server pid=$SV"
kill "$SV" 2>/dev/null; sleep 2; kill -9 "$SV" 2>/dev/null || true

if [ "$RC" -ne 0 ]; then
  log "!!! PPCCAB DIAGNOSIS FAILED (rc=$RC) -- tail of $ROLLOUT_LOG:"
  tail -30 "$ROLLOUT_LOG" >> "$ORCH_LOG"
  exit "$RC"
fi
log "=== DONE: ppccab diagnosis batch complete ==="
