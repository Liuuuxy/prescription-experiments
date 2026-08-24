#!/usr/bin/env bash
# Task-2 baseline eval orchestrator — waits for ledger_ppccab/E_manifest.parquet
# (150 rows, written by `eval_set build-e`), then serves ppccab_base@9999 on
# :8135 and runs `eval_set eval-baseline` (workers=4) under the ppccab profile.
# Records b / per-stratum b / sigma_e_eval into ledger_ppccab.
#   nohup bash run_baseline_ppccab.sh > /data/xinyua11/tmp/ppccab_baseline_orch.log 2>&1 &
set -uo pipefail

REPO=/data/xinyua11/robocasa
OPENPI_DIR=/data/xinyua11/openpi
OPENPI_PY=/data/xinyua11/conda/envs/openpi/bin/python
ROBOCASA_PY=/data/xinyua11/conda/envs/robocasa/bin/python

CKPT_DIR=/data/xinyua11/openpi/checkpoints/pi0_ppccab_bandit_a/ppccab_base/9999
MANIFEST="$REPO/bandit_v1/ledger_ppccab/E_manifest.parquet"
PORT=8135
ORCH_LOG=/data/xinyua11/tmp/ppccab_baseline_orch.log
SERVE_LOG=/data/xinyua11/tmp/ppccab_baseline_serve.log
EVAL_LOG=/data/xinyua11/tmp/ppccab_baseline_eval.log

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$ORCH_LOG"; }
free_mb(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' '; }

log "=== ppccab baseline orchestrator START (pid $$) ==="

while :; do
  ok=no
  "$ROBOCASA_PY" -c "
import sys, pandas as pd
try:
    sys.exit(0 if len(pd.read_parquet('$MANIFEST')) == 150 else 1)
except Exception:
    sys.exit(1)
" 2>/dev/null && ok=yes
  log "waiting: E_manifest(150 rows)=$ok"
  [ "$ok" = yes ] && break
  sleep 120
done
log ">>> E manifest ready -- serving"

cd "$OPENPI_DIR" || { log "FATAL cd"; exit 4; }
GPU=0
f=$(free_mb 0); [ "${f:-0}" -lt 20000 ] && { f=$(free_mb 1); [ "${f:-0}" -ge 20000 ] && GPU=1 || { log "!!! no GPU with 20GB free"; exit 2; }; }
log ">>> serving on GPU$GPU :$PORT"
CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 MUJOCO_GL=egl \
  TMPDIR=/data/xinyua11/tmp HF_HOME=/data/xinyua11/.cache/huggingface \
  "$OPENPI_PY" scripts/serve_policy.py --port "$PORT" policy:checkpoint \
  --policy.config pi0_ppccab_bandit_a --policy.dir "$CKPT_DIR" > "$SERVE_LOG" 2>&1 &
SV=$!
up=no
for i in $(seq 1 90); do
  "$ROBOCASA_PY" -c "
import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1',$PORT)); s.close()" 2>/dev/null && { up=yes; break; }
  kill -0 "$SV" 2>/dev/null || { log "!!! SERVER DIED"; tail -20 "$SERVE_LOG" >> "$ORCH_LOG"; exit 2; }
  sleep 5
done
[ "$up" = yes ] || { log "!!! server never came up"; kill "$SV" 2>/dev/null; exit 2; }
log ">>> server up (pid $SV)"

cd "$REPO" || exit 4
BANDIT_TASK_PROFILE=ppccab MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  "$ROBOCASA_PY" -u -m bandit_v1.eval_set eval-baseline \
  --host 127.0.0.1 --port "$PORT" --workers 4 > "$EVAL_LOG" 2>&1
RC=$?
log ">>> eval-baseline exited rc=$RC"
kill "$SV" 2>/dev/null; sleep 2; kill -9 "$SV" 2>/dev/null || true
if [ "$RC" -ne 0 ]; then
  log "!!! PPCCAB BASELINE FAILED"; tail -25 "$EVAL_LOG" >> "$ORCH_LOG"; exit "$RC"
fi
tail -12 "$EVAL_LOG" >> "$ORCH_LOG"
log "=== DONE: ppccab baseline complete ==="
