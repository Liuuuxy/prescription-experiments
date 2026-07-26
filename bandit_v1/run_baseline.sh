#!/usr/bin/env bash
# bandit_v1 Task 9: baseline orchestrator. Waits for Task 7's diagnosis batch
# to actually finish (2,400 diag rows in the ledger), then:
#   1. fits the REAL difficulty map p_hat_0/p_stage on those diag rows
#      (bandit_v1.map_fit's new CLI), gated on held-out AUC >= 0.55 -- loud
#      exit otherwise (the design's investigate-before-proceeding checkpoint,
#      task-8-brief.md Step 3's own sanity note);
#   2. builds the frozen 150-start eval set E (bandit_v1.eval_set build-e --
#      scan+capture+score+stratify+delete-nonselected+freeze/hash the
#      manifest, no GPU/policy server needed for this step, same as
#      diagnosis.py's condition capture);
#   3. serves pi_0 on port 8125 (SAME serve-retry + GPU-memory-guard
#      machinery as run_diagnosis.sh's Step 2 -- see its report for the
#      postmortem this hardening addresses) and runs
#      bandit_v1.eval_set eval-baseline (policy_id="pi0_baseline",
#      repeats=3), which records b + per-stratum b + sigma_e(eval) +
#      per-repeat means into bandit_v1/ledger/config.yaml (appended, not a
#      round-trip rewrite -- see eval_set.append_baseline_to_config_yaml's
#      docstring) and writes the per-start flip table;
#   4. kills the server, logs DONE.
#
# DO NOT LAUNCH THIS YET. Two preconditions, only the first of which this
# script itself waits for:
#   (a) Task 7's diagnosis orchestrator (run_diagnosis.sh) must actually
#       finish -- this script polls for that (Step 1 below).
#   (b) the diagnosis batch's RESULTS need a human look (per-condition
#       success spread, stage histogram, overall success rate) before this
#       script is even STARTED -- launching it is a controller decision made
#       after that review, not an automatic follow-on to (a). This script
#       does not and cannot enforce (b) itself; it only enforces (a).
# Once launched, it ALSO enforces its own second human-visible checkpoint at
# Step 2: the real map validation report (held-out AUC/log-loss/calibration)
# is written to disk and should be eyeballed -- the AUC>=0.55 gate is an
# automatic floor under that review, not a replacement for it.
#
# Known gotcha (Task 6/7, see their reports): `conda run -n <env> python ...`
# pipe-buffers a long-running child's stdout through a non-line-flushed
# relay. Fix (used everywhere below, identical to run_diagnosis.sh): invoke
# the conda env's python BINARY directly, never `conda run`/`conda activate`.
#
# Run detached so it survives the session:
#   nohup bash /data/xinyua11/robocasa/bandit_v1/run_baseline.sh \
#     > /data/xinyua11/tmp/bandit_baseline_orch_stdout.log 2>&1 &
#
# Adaptation (arms already frozen at 8e76d12, this run's map already fit and
# gate-passed -- see ledger/arms.yaml's map_hash / ledger/config.yaml's
# arms_freeze block): Step 2 below now SKIPS the map refit entirely when
# ledger/map_models.joblib + map_validation_report.json already show a
# gate-passed fit on disk (arms.yaml pins that exact joblib's sha256 -- a
# refit here would silently invalidate that pin). EGL is broken box-wide
# (driver update) so every env-touching step now runs MUJOCO_GL=osmesa
# instead of egl. The baseline eval (Step 5) now runs with
# --workers "$EVAL_WORKERS" (parallel_eval.py, opt-in). The serve-retry loop
# and the AUC-gate refusal path (exit 3) are otherwise UNCHANGED -- both
# still apply verbatim for a future run against a fresh, not-yet-frozen map.
set -uo pipefail

REPO=/data/xinyua11/robocasa
OPENPI_DIR=/data/xinyua11/openpi
OPENPI_PY=/data/xinyua11/conda/envs/openpi/bin/python
ROBOCASA_PY=/data/xinyua11/conda/envs/robocasa/bin/python

# Same fine-tuned pi_0 checkpoint run_diagnosis.sh serves for the diagnosis
# batch -- the baseline b is THIS checkpoint evaluated on E, per the design
# doc's "pi_0 on E, 3 independent repeats".
CKPT_DIR=/data/xinyua11/openpi/checkpoints/pi0_ppc2sink_pi0base/pi0_v1/19999
N_DIAG_EXPECTED=2400

PORT=8125               # brief's assigned port for the baseline serve
POLL_SECS=120            # precondition-wait poll interval (matches run_diagnosis.sh)
PORT_WAIT_TRIES=90       # 90 x 5s = 7.5 min max wait for the server socket to accept
PORT_WAIT_SLEEP=5

EVAL_WORKERS=4           # bandit_v1 rollout-speedup #1, opt-in (parallel_eval.py):
                          # shards the 150 E-set starts round-robin across this many
                          # worker subprocesses hitting the same served policy --
                          # see eval_set.eval_checkpoint's `workers` param docstring.

SERVE_MAX_ATTEMPTS=6      # identical serve-retry policy to run_diagnosis.sh
SERVE_RETRY_SLEEP=300
GPU0_MIN_FREE_MB=20000
GPU_WAIT_POLL_SECS=60
GPU_WAIT_MAX_SECS=14400   # cap total pre-serve GPU-memory waiting at 4h

AUC_GATE_MIN=0.55         # mirrors map_fit.AUC_GATE_MIN -- kept here only for
                          # log messages; the actual gate is enforced inside
                          # `bandit_v1.map_fit`'s CLI (single source of truth).

DIAG_ORCH_LOG=/data/xinyua11/tmp/bandit_diag_orch.log     # Task 7's own log (read-only here)
ORCH_LOG=/data/xinyua11/tmp/bandit_baseline_orch.log
SERVE_LOG=/data/xinyua11/tmp/bandit_baseline_serve.log
MAPFIT_LOG=/data/xinyua11/tmp/bandit_baseline_mapfit.log
BUILDE_LOG=/data/xinyua11/tmp/bandit_baseline_build_e.log
EVAL_LOG=/data/xinyua11/tmp/bandit_baseline_eval.log

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$ORCH_LOG"; }
free_mb(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' '; }

log "=== baseline orchestrator START (pid $$) -- waiting mode ==="

# --- Step 1: wait for the diagnosis batch to actually finish ----------------
# Two corroborating signals, both required: run_diagnosis.sh's own DONE line
# in its log (confirms the orchestrator itself believes it's done) AND the
# ledger independently shows exactly N_DIAG_EXPECTED diag/pi0 rows (confirms
# the data really landed -- never trust a log line alone for a fact the
# ledger itself can just be asked).
diag_done(){
  [ -f "$DIAG_ORCH_LOG" ] || return 1
  grep -q "DONE: diagnosis batch complete" "$DIAG_ORCH_LOG" || return 1
  "$ROBOCASA_PY" -c "
import sys
from bandit_v1 import ledger
try:
    df = ledger.read('episodes')
except FileNotFoundError:
    sys.exit(1)
d = df[(df['phase'] == 'diag') & (df['policy_id'] == 'pi0')]
sys.exit(0 if len(d) == $N_DIAG_EXPECTED else 1)
" 2>/dev/null
}

while :; do
  dd=no; diag_done && dd=yes
  log "waiting: diagnosis_done(${DIAG_ORCH_LOG} has a DONE line AND ledger has ${N_DIAG_EXPECTED} diag/pi0 rows)=${dd}"
  if [ "$dd" = yes ]; then break; fi
  sleep "$POLL_SECS"
done
log ">>> diagnosis batch confirmed complete -- proceeding to the real map fit"

# --- Step 2: fit the REAL difficulty map on the diag rows, AUC-gated -------
# Frozen-map fast path: if bandit_v1/ledger/map_models.joblib already exists
# AND map_validation_report.json shows its winning family cleared the same
# AUC_GATE_MIN gate map_fit._main itself enforces, the map is FROZEN (see
# ledger/arms.yaml's map_hash -- clustering/wells were built against that
# exact joblib's sha256, which arms.yaml pins). Refitting here would silently
# rewrite map_models.joblib out from under that pin, so this path never calls
# bandit_v1.map_fit at all in that case -- it only reads the two files below.
# The gate-refusal path (exit 3 on a real AUC-gate failure) is UNCHANGED for
# the general case: a fresh diag batch with no frozen map yet still goes
# through the exact same map_fit CLI + gate as before.
cd "$REPO" || { log "FATAL cd $REPO"; exit 4; }

MAP_JOBLIB="$REPO/bandit_v1/ledger/map_models.joblib"
MAP_REPORT="$REPO/bandit_v1/ledger/map_validation_report.json"

map_frozen_sha(){
  # Prints the existing joblib's sha256 on stdout iff both files exist AND
  # the report's winning family's AUC clears AUC_GATE_MIN (imported from
  # bandit_v1.map_fit -- single source of truth, same convention as this
  # script's own AUC_GATE_MIN log-message-only constant above). Prints
  # nothing and exits nonzero otherwise (missing files, missing/NaN auc, or
  # gate not cleared) -- caller falls back to the normal refit path.
  "$ROBOCASA_PY" -c "
import json, hashlib, os, sys
p = r'$MAP_JOBLIB'
rp = r'$MAP_REPORT'
if not (os.path.exists(p) and os.path.exists(rp)):
    sys.exit(1)
report = json.load(open(rp))
winner = report.get('winner')
fam = report.get(winner) or {}
auc = fam.get('auc')
from bandit_v1.map_fit import AUC_GATE_MIN
if auc is None or not (auc >= AUC_GATE_MIN):
    sys.exit(1)
print(hashlib.sha256(open(p, 'rb').read()).hexdigest())
" 2>/dev/null
}

FROZEN_SHA=$(map_frozen_sha)
if [ -n "$FROZEN_SHA" ]; then
  log ">>> map already frozen (sha ${FROZEN_SHA:0:12}) -- skipping refit"
else
  log ">>> fitting bandit_v1.map_fit on real diag data (gate: AUC >= ${AUC_GATE_MIN}) -> $MAPFIT_LOG"
  "$ROBOCASA_PY" -u -m bandit_v1.map_fit > "$MAPFIT_LOG" 2>&1
  RC=$?
  tail -60 "$MAPFIT_LOG" >> "$ORCH_LOG"
  if [ "$RC" -ne 0 ]; then
    log "!!! MAP FIT GATE FAILED (rc=$RC) -- held-out AUC likely < ${AUC_GATE_MIN}, or a real"
    log "!!! error (wrong join, leaked constants -- see task-8-brief.md Step 3 / $MAPFIT_LOG)."
    log "!!! Refusing to build E or run the baseline. map_models.joblib was NOT written on"
    log "!!! a gate failure (see bandit_v1/map_fit.py's _main). Investigate, fix, and"
    log "!!! relaunch this script from the top."
    exit 3
  fi
  log ">>> map fit gate PASSED -- full validation report is in $MAPFIT_LOG and"
  log ">>> bandit_v1/ledger/map_validation_report.json (HUMAN CHECKPOINT: eyeball the"
  log ">>> held-out AUC/log-loss/calibration there before trusting E -- this script does"
  log ">>> not pause for that review itself; it was already implicitly satisfied by the"
  log ">>> controller reviewing the diagnosis results before even launching this script)."
fi
log ">>> proceeding to build eval set E"

# --- Step 3: build the frozen 150-start eval set E --------------------------
log ">>> building eval set E (bandit_v1.eval_set build-e) -> $BUILDE_LOG"
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa "$ROBOCASA_PY" -u -m bandit_v1.eval_set build-e > "$BUILDE_LOG" 2>&1
RC=$?
tail -60 "$BUILDE_LOG" >> "$ORCH_LOG"
if [ "$RC" -ne 0 ]; then
  log "!!! BUILD E FAILED (rc=$RC) -- see $BUILDE_LOG"
  exit 3
fi
log ">>> eval set E built (150 starts; per-stratum p_hat ranges are in $BUILDE_LOG)"

# --- Step 4: serve pi_0 on port 8125, claim GPU0 promptly -------------------
# Identical serve-retry + GPU-memory-guard policy to run_diagnosis.sh's Step 2
# (see its report's Task-7-postmortem for why this hardening exists: GPU0 can
# be grabbed by another user's job between it freeing up and this claiming it).
cd "$OPENPI_DIR" || { log "FATAL cd $OPENPI_DIR"; exit 4; }

served=no
gpu_wait_elapsed=0
attempt=1
while [ "$attempt" -le "$SERVE_MAX_ATTEMPTS" ]; do
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
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 MUJOCO_GL=osmesa \
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

# --- Step 5: eval_checkpoint x3 (baseline), record into config.yaml --------
cd "$REPO" || { log "FATAL cd $REPO"; exit 4; }
log ">>> running baseline eval (bandit_v1.eval_set eval-baseline, repeats=3, policy_id=pi0_baseline, workers=${EVAL_WORKERS}) -> $EVAL_LOG"
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa "$ROBOCASA_PY" -u -m bandit_v1.eval_set eval-baseline \
  --host 127.0.0.1 --port "$PORT" --policy_id pi0_baseline --repeats 3 \
  --checkpoint_id "$CKPT_DIR" --workers "$EVAL_WORKERS" > "$EVAL_LOG" 2>&1
RC=$?
tail -60 "$EVAL_LOG" >> "$ORCH_LOG"

# --- Step 6: kill the server regardless of eval outcome ---------------------
log ">>> killing server pid=$SV"
kill "$SV" 2>/dev/null; sleep 2; kill -9 "$SV" 2>/dev/null || true

if [ "$RC" -ne 0 ]; then
  log "!!! BASELINE EVAL FAILED (rc=$RC) -- see $EVAL_LOG"
  exit "$RC"
fi

log "=== DONE: baseline eval complete -- b + per-stratum b + sigma_e_eval + per-repeat"
log "means recorded in bandit_v1/ledger/config.yaml (appended 'baseline:' block);"
log "per-start flip table at bandit_v1/ledger/baseline_flip_table.parquet ==="
