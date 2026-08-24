#!/usr/bin/env bash
# Rounds 2-3 auto-chain for the task-2 gradient-arm race (frozen protocol:
# uniform allocation, gate in shadow). Waits for round-1 drivers, retro-probes
# round-1's kept checkpoints (old driver code didn't probe inline), deletes
# them, then launches rounds 2 (j=141) and 3 (j=142) with the probing driver,
# running the shadow cmeter between rounds. No allocation decisions anywhere.
set -u
REPO=/data/xinyua11/robocasa
GA=$REPO/gradient_analysis/ppccab
RPY=/data/xinyua11/conda/envs/robocasa/bin/python
OPY=/data/xinyua11/conda/envs/openpi/bin/python
CK=/data/xinyua11/openpi/checkpoints
LOG=/data/xinyua11/tmp/ppccab_race_chain.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

wait_complete(){  # $1 $2 = the two driver logs to wait on
  while :; do
    a=$(grep -c "RUNG3PPC QUEUE COMPLETE" "$1" 2>/dev/null || true)
    b=$(grep -c "RUNG3PPC QUEUE COMPLETE" "$2" 2>/dev/null || true)
    [ "${a:-0}" -ge 1 ] && [ "${b:-0}" -ge 1 ] && return
    sleep 300
  done
}

launch_round(){  # $1 = j
  J=$1
  log ">>> launching round j=$J"
  setsid nohup env -u PYOPENGL_PLATFORM BANDIT_TASK_PROFILE=ppccab MUJOCO_GL=egl \
    TMPDIR=/data/xinyua11/tmp DRIVER_ARMS=gc0,gc1,gc2,random DRIVER_SLOT=a DRIVER_GPU=0 DRIVER_J=$J \
    $RPY -u $GA/ppccab_rung3_driver.py > /data/xinyua11/tmp/ppccab_rung3_slotA_j$J.log 2>&1 < /dev/null &
  setsid nohup env -u PYOPENGL_PLATFORM BANDIT_TASK_PROFILE=ppccab MUJOCO_GL=egl \
    TMPDIR=/data/xinyua11/tmp DRIVER_ARMS=gc3,gc4,gc5 DRIVER_SLOT=b DRIVER_GPU=1 DRIVER_J=$J \
    $RPY -u $GA/ppccab_rung3_driver.py > /data/xinyua11/tmp/ppccab_rung3_slotB_j$J.log 2>&1 < /dev/null &
}

log "=== race chain START (pid $$) ==="
wait_complete /data/xinyua11/tmp/ppccab_rung3_slotA.log /data/xinyua11/tmp/ppccab_rung3_slotB.log
log ">>> round 1 complete -- retro-probing kept j140 checkpoints"

mkdir -p "$GA/ucb_robot/shadow"
for d in "$CK"/pi0_ppccab_bandit_*/*_j140; do
  [ -d "$d/9999" ] || continue
  pid=$(basename "$d")
  out="$GA/ucb_robot/shadow/${pid}_probe9999.json"
  if [ ! -f "$out" ]; then
    log "probe $pid"
    env -u PYOPENGL_PLATFORM BANDIT_TASK_PROFILE=ppccab CUDA_VISIBLE_DEVICES=0 \
      XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 OMP_NUM_THREADS=4 \
      TMPDIR=/data/xinyua11/tmp HF_HOME=/data/xinyua11/.cache/huggingface \
      $OPY -u $REPO/gradient_analysis/ppccab/ppccab_probe_ckpt.py \
      --ckpt_root "$d" --step 9999 --out "$out" >> "$LOG" 2>&1
  fi
  [ -f "$out" ] && { rm -rf "$d"; log "deleted ckpt $pid (probed)"; }
done

log ">>> shadow cmeter after round 1"
cd "$REPO" && BANDIT_TASK_PROFILE=ppccab $RPY -u $GA/ppccab_cmeter.py >> "$LOG" 2>&1

launch_round 141
sleep 60
wait_complete /data/xinyua11/tmp/ppccab_rung3_slotA_j141.log /data/xinyua11/tmp/ppccab_rung3_slotB_j141.log
log ">>> round 2 complete -- shadow cmeter"
cd "$REPO" && BANDIT_TASK_PROFILE=ppccab $RPY -u $GA/ppccab_cmeter.py >> "$LOG" 2>&1

launch_round 142
sleep 60
wait_complete /data/xinyua11/tmp/ppccab_rung3_slotA_j142.log /data/xinyua11/tmp/ppccab_rung3_slotB_j142.log
log ">>> round 3 complete -- final shadow cmeter"
cd "$REPO" && BANDIT_TASK_PROFILE=ppccab $RPY -u $GA/ppccab_cmeter.py >> "$LOG" 2>&1
log "=== RACE 3 ROUNDS COMPLETE ==="
