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
cd "$OPENPI_DIR"
log ">>> serving pi0_ppc2sink_pi0base from $CKPT_DIR on :$PORT (gpu0)"
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
    log "!!! SERVER DIED while waiting for port -- tail of $SERVE_LOG:"
    tail -30 "$SERVE_LOG" >> "$ORCH_LOG"
    exit 2
  fi
  sleep "$PORT_WAIT_SLEEP"
done
if [ "$up" != yes ]; then
  log "!!! server never accepted connections after $((PORT_WAIT_TRIES * PORT_WAIT_SLEEP))s -- aborting"
  kill "$SV" 2>/dev/null; sleep 2; kill -9 "$SV" 2>/dev/null || true
  exit 3
fi

# --- Step 3: chunked, resumable rollouts (300 starts x M_DIAG=8 repeats) ----
cd "$REPO"
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
