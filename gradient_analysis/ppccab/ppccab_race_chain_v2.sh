#!/usr/bin/env bash
# Race chain v2 (2026-08-19) — takes over from ppccab_race_chain.sh after it
# launches round 2. Adds the round-1 control MAKEUP (random_j140 failed on the
# regions=None driver bug, fixed in ppccab_rung3_driver.py) in the GPU1 gap
# between slot-B and slot-A round-2 completion, then serializes round 3 so the
# makeup never collides with round-3 training. Protocol content unchanged:
# uniform allocation, frozen shadow gate, cmeter between rounds.
set -u
REPO=/data/xinyua11/robocasa
GA=$REPO/gradient_analysis/ppccab
RPY=/data/xinyua11/conda/envs/robocasa/bin/python
LOG=/data/xinyua11/tmp/ppccab_race_chain_v2.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

wait_log(){ while ! grep -q "RUNG3PPC QUEUE COMPLETE" "$1" 2>/dev/null; do sleep 300; done; }

launch_round(){
  J=$1
  log ">>> launching round j=$J"
  setsid nohup env -u PYOPENGL_PLATFORM BANDIT_TASK_PROFILE=ppccab MUJOCO_GL=egl \
    TMPDIR=/data/xinyua11/tmp DRIVER_ARMS=gc0,gc1,gc2,random DRIVER_SLOT=a DRIVER_GPU=0 DRIVER_J=$J \
    $RPY -u $GA/ppccab_rung3_driver.py > /data/xinyua11/tmp/ppccab_rung3_slotA_j$J.log 2>&1 < /dev/null &
  setsid nohup env -u PYOPENGL_PLATFORM BANDIT_TASK_PROFILE=ppccab MUJOCO_GL=egl \
    TMPDIR=/data/xinyua11/tmp DRIVER_ARMS=gc3,gc4,gc5 DRIVER_SLOT=b DRIVER_GPU=1 DRIVER_J=$J \
    $RPY -u $GA/ppccab_rung3_driver.py > /data/xinyua11/tmp/ppccab_rung3_slotB_j$J.log 2>&1 < /dev/null &
}

log "=== race chain v2 START (pid $$) ==="

log "waiting for slot-B round 2 (j=141) to finish -> GPU1 gap for the makeup"
wait_log /data/xinyua11/tmp/ppccab_rung3_slotB_j141.log
log ">>> MAKEUP: random_j140 on GPU1 slot b (round-1 control)"
env -u PYOPENGL_PLATFORM BANDIT_TASK_PROFILE=ppccab MUJOCO_GL=egl \
  TMPDIR=/data/xinyua11/tmp DRIVER_ARMS=random DRIVER_SLOT=b DRIVER_GPU=1 DRIVER_J=140 \
  $RPY -u $GA/ppccab_rung3_driver.py > /data/xinyua11/tmp/ppccab_rung3_makeup.log 2>&1
log ">>> makeup driver exited (see ppccab_rung3_makeup.log)"

log "waiting for slot-A round 2 (j=141)"
wait_log /data/xinyua11/tmp/ppccab_rung3_slotA_j141.log
log ">>> round 2 complete -- cmeter (now includes round-1 pairs via makeup)"
cd "$REPO" && BANDIT_TASK_PROFILE=ppccab $RPY -u $GA/ppccab_cmeter.py >> "$LOG" 2>&1

launch_round 142
sleep 60
wait_log /data/xinyua11/tmp/ppccab_rung3_slotA_j142.log
wait_log /data/xinyua11/tmp/ppccab_rung3_slotB_j142.log
log ">>> round 3 complete -- final cmeter"
cd "$REPO" && BANDIT_TASK_PROFILE=ppccab $RPY -u $GA/ppccab_cmeter.py >> "$LOG" 2>&1
log "=== RACE 3 ROUNDS COMPLETE (v2) ==="
