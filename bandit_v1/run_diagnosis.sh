#!/usr/bin/env bash
# bandit_v1 Task 7: diagnosis-batch orchestrator -- 2,400 rollouts (300 starts x
# config.M_DIAG=8 repeats) of the freshly-trained pi_0 (pi0_ppc2sink_pi0base /
# pi0_v1), phase="diag" policy_id="pi0" in ledger table "episodes".
#
# Launched in WAITING mode: at start time neither precondition is met yet --
# pi_0 is still training on GPU0 (final checkpoint not yet at .../19999) and the
# 300-condition capture (bandit_v1/diagnosis.py, PID 2996254) is still scanning
# toward diag_conditions.parquet's 300 rows. This script polls both every 120s
# and does nothing else until BOTH are true, so it is safe to launch immediately
# and long before either precondition is satisfied.
#
# Known gotcha (Task 6, see task-6-report.md): `conda run -n <env> python ...`
# pipe-buffers a long-running child's stdout through a non-line-flushed relay,
# so log files can sit empty for many minutes even with real progress on disk.
# Fix (used everywhere below): invoke the conda env's python BINARY directly
# (/data/xinyua11/conda/envs/openpi/bin/python,
# /data/xinyua11/conda/envs/robocasa/bin/python), never `conda run`/`conda
# activate` for the long-running server or rollout processes.
#
# Run detached so it survives this session:
#   nohup bash /data/xinyua11/robocasa/bandit_v1/run_diagnosis.sh \
#     > /data/xinyua11/tmp/bandit_diag_orch.log 2>&1 &
set -uo pipefail

REPO=/data/xinyua11/robocasa
OPENPI_DIR=/data/xinyua11/openpi
OPENPI_PY=/data/xinyua11/conda/envs/openpi/bin/python
ROBOCASA_PY=/data/xinyua11/conda/envs/robocasa/bin/python

CKPT_DIR=/data/xinyua11/openpi/checkpoints/pi0_ppc2sink_pi0base/pi0_v1/19999
PARQUET="$REPO/bandit_v1/ledger/diag_conditions.parquet"
N_DIAG=300

PORT=8124
POLL_SECS=120          # precondition-wait poll interval (brief's Step 1)
PORT_WAIT_TRIES=90     # 90 x 5s = 7.5 min max wait for the server socket to accept
PORT_WAIT_SLEEP=5

SERVE_MAX_ATTEMPTS=6      # max real serve attempts before giving up (Task 7 hardening)
SERVE_RETRY_SLEEP=300     # seconds between serve attempts
GPU0_MIN_FREE_MB=20000    # required free MiB on GPU0 before each serve attempt
GPU_WAIT_POLL_SECS=60     # how often to re-check GPU0 free mem while waiting (does not burn an attempt)
GPU_WAIT_MAX_SECS=14400   # cap total pre-serve GPU-memory waiting at 4h, then exit 2 loudly

ORCH_LOG=/data/xinyua11/tmp/bandit_diag_orch.log
SERVE_LOG=/data/xinyua11/tmp/bandit_diag_serve.log
ROLLOUT_LOG=/data/xinyua11/tmp/bandit_diag_rollouts.log

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$ORCH_LOG"; }
free_mb(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' '; }

log "=== diagnosis orchestrator START (pid $$) -- waiting mode ==="

# --- Step 1: wait for BOTH preconditions --------------------------------
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
  log "waiting: checkpoint(${CKPT_DIR})=${ck}  diag_conditions(==${N_DIAG} rows)=${pq}"
  if [ "$ck" = yes ] && [ "$pq" = yes ]; then break; fi
  sleep "$POLL_SECS"
done
log ">>> both preconditions satisfied -- proceeding to serve pi_0"

# --- Step 2: serve pi_0, claim GPU0 promptly (it may be grabbed by others) --
# Hardening (Task 7 postmortem): GPU0 can be grabbed by another user's job
# between training finishing and this step claiming it. Retry the whole
# serve+port-wait attempt up to SERVE_MAX_ATTEMPTS times, SERVE_RETRY_SLEEP
# apart, gated on GPU0 having >=GPU0_MIN_FREE_MB free before each attempt
# (mirrors launch_pi0.sh's free_mb() parsing). Waiting for memory does not
# burn an attempt, but total memory-wait time is capped at GPU_WAIT_MAX_SECS.
cd "$OPENPI_DIR" || { log "FATAL cd $OPENPI_DIR"; exit 4; }

served=no
gpu_wait_elapsed=0
attempt=1
while [ "$attempt" -le "$SERVE_MAX_ATTEMPTS" ]; do
  # gate on GPU0 memory before burning a serve attempt
  while :; do
    f=$(free_mb 0)
    if [ "${f:-0}" -ge "$GPU0_MIN_FREE_MB" ]; then
      break
    fi
    log "gpu0 free=${f:-0}MiB < ${GPU0_MIN_FREE_MB}MiB needed -- waiting before serve attempt ${attempt}/${SERVE_MAX_ATTEMPTS} (mem-wait elapsed ${gpu_wait_elapsed}s / cap ${GPU_WAIT_MAX_SECS}s)"
    if [ "$gpu_wait_elapsed" -ge "$GPU_WAIT_MAX_SECS" ]; then
      log "!!! GPU0 memory wait exceeded cap of ${GPU_WAIT_MAX_SECS}s -- aborting"
      exit 2
    fi
    sleep "$GPU_WAIT_POLL_SECS"
    gpu_wait_elapsed=$((gpu_wait_elapsed + GPU_WAIT_POLL_SECS))
  done

  log ">>> serve attempt ${attempt}/${SERVE_MAX_ATTEMPTS}: serving pi0_ppc2sink_pi0base from $CKPT_DIR on :$PORT (gpu0 free=${f}MiB)"
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 MUJOCO_GL=egl \
    TMPDIR=/data/xinyua11/tmp HF_HOME=/data/xinyua11/.cache/huggingface \
    "$OPENPI_PY" scripts/serve_policy.py --port "$PORT" policy:checkpoint \
    --policy.config pi0_ppc2sink_pi0base --policy.dir "$CKPT_DIR" \
    > "$SERVE_LOG" 2>&1 &
  SV=$!
  log ">>> server pid=$SV, waiting for it to accept connections on :$PORT"

  up=no
  for i in $(seq 1 "$PORT_WAIT_TRIES"); do
    if "$ROBOCASA_PY" -c "
import socket
s = socket.socket(); s.settimeout(1)
s.connect(('127.0.0.1', $PORT)); s.close()
" 2>/dev/null; then
      up=yes
      log ">>> server up after ${i}x${PORT_WAIT_SLEEP}s"
      break
    fi
    if ! kill -0 "$SV" 2>/dev/null; then
      log "!!! SERVER DIED while waiting for port (attempt ${attempt}/${SERVE_MAX_ATTEMPTS}) -- tail of $SERVE_LOG:"
      tail -30 "$SERVE_LOG" >> "$ORCH_LOG"
      break
    fi
    sleep "$PORT_WAIT_SLEEP"
  done

  if [ "$up" = yes ]; then
    served=yes
    break
  fi

  if kill -0 "$SV" 2>/dev/null; then
    log "!!! server never accepted connections after $((PORT_WAIT_TRIES * PORT_WAIT_SLEEP))s (attempt ${attempt}/${SERVE_MAX_ATTEMPTS}) -- tail of $SERVE_LOG:"
    tail -30 "$SERVE_LOG" >> "$ORCH_LOG"
  fi
  kill "$SV" 2>/dev/null; sleep 2; kill -9 "$SV" 2>/dev/null || true

  attempt=$((attempt + 1))
  if [ "$attempt" -le "$SERVE_MAX_ATTEMPTS" ]; then
    log "!!! serve attempt failed -- retrying in ${SERVE_RETRY_SLEEP}s (next attempt ${attempt}/${SERVE_MAX_ATTEMPTS})"
    sleep "$SERVE_RETRY_SLEEP"
  fi
done

if [ "$served" != yes ]; then
  log "!!! all ${SERVE_MAX_ATTEMPTS} serve attempts failed -- aborting"
  exit 2
fi

# --- Step 3: chunked, resumable rollouts (300 starts x M_DIAG=8 repeats) ----
cd "$REPO" || { log "FATAL cd $REPO"; exit 4; }
log ">>> starting rollouts: bandit_v1.run_diagnosis --host 127.0.0.1 --port $PORT"
log ">>> rollout progress -> $ROLLOUT_LOG"
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$ROBOCASA_PY" -u -m bandit_v1.run_diagnosis \
  --host 127.0.0.1 --port "$PORT" >> "$ROLLOUT_LOG" 2>&1
RC=$?
log ">>> rollout driver exited rc=$RC"

# --- Step 4: kill the server, log final counts + success rate --------------
log ">>> killing server pid=$SV"
kill "$SV" 2>/dev/null; sleep 2; kill -9 "$SV" 2>/dev/null || true

if [ "$RC" -ne 0 ]; then
  log "!!! DIAGNOSIS BATCH FAILED (rollout driver rc=$RC) -- tail of $ROLLOUT_LOG:"
  tail -30 "$ROLLOUT_LOG" >> "$ORCH_LOG"
  exit "$RC"
fi

log ">>> computing final per-stage counts + success rate from the ledger"
"$ROBOCASA_PY" -c "
from bandit_v1 import ledger
df = ledger.read('episodes')
d = df[(df['phase'] == 'diag') & (df['policy_id'] == 'pi0')]
print('DIAG_TOTAL', len(d))
print('SUCCESS_RATE', round(d['success'].mean(), 4) if len(d) else float('nan'))
print('STAGE_COUNTS', d['failure_stage'].value_counts().to_dict())
" >> "$ORCH_LOG" 2>&1

log "=== DONE: diagnosis batch complete ==="
