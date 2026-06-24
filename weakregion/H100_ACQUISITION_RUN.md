# H100 task: per-category STUDENT (DP) estimates for the core algorithm

Goal: produce the trained DP's per-scene **success + epistemic uncertainty** over
`gym.make` seeds **0–499** (the same scenes the pi0 teacher is evaluated on, on the
4070), so `run_acquisition.py` can score every object category and allocate the
generation budget. This is the *student* + *uncertainty* half of the acquisition
inputs; the 4070 produces the *teacher* (pi0) half.

## Why seeds 0–499 and this script
- `gym.make(robocasa/PickPlaceCounterToSink, split=pretrain, seed=S)` is deterministic
  and is what every eval/training run uses. seed S == the same scene on both boxes.
- The DP fork's `eval_robocasa` only labels rollouts with cosmetic seeds (100000+) and
  seeds the env once at creation — it CANNOT target specific scenes. So we use
  `policy_analysis/eval_dp_weakregion.py`, which drives one env per seed via `gym.make`
  and reuses the fork's `RobomimicImageWrapper`+`MultiStepWrapper` (byte-identical DP
  inference) while logging object category + success + K-sample uncertainty.

## Run (on the H100, in the env that runs eval_robocasa)
```sh
git pull                                  # get policy_analysis/eval_dp_weakregion.py + run_acquisition.py
CK=/data/xinyua11/dp_runs/pickplacesink_human/seed0_noncached_partial_ep471/epoch=0400-*.ckpt
MUJOCO_GL=egl python policy_analysis/eval_dp_weakregion.py \
    -c "$CK" --pool_start 0 --pool_n 500 -k 8 \
    --out weakregion/dp_seed0_ep400_pool500
# n_envs=1 here (one env per seed). ~500 rollouts; parallelize across both GPUs by
# splitting the range (e.g. --pool_start 0 --pool_n 250 on GPU0, 250..500 on GPU1)
# and merging the two dp_eval.json result lists.
```
Output: `weakregion/dp_seed0_ep400_pool500/dp_eval.json` — per seed:
`{seed, object_category, obj_height, success, uncertainty}`.

## Then (once BOTH halves exist)
- 4070 produces `weakregion/pi0_PickPlaceCounterToSink_n500/weakregion.json` (teacher).
- Join + allocate:
```sh
python policy_analysis/run_acquisition.py \
    --teacher weakregion/pi0_PickPlaceCounterToSink_n500/weakregion.json \
    --student weakregion/dp_seed0_ep400_pool500/dp_eval.json \
    --budget 200 --min_n 3
```
→ `weakregion/acquisition_plan.json` = score + attempt allocation per category
(P(teacher)×P(student_fails)×uncertainty, Wilson-LCB). That allocation is the
**core-algo arm's collection spec.**

## Notes
- Optionally eval seed0 at ep200/300 too (the 2% at ep400 may be overfit) to pick the
  baseline operating point with the most headroom.
- The eval also yields the protocol-correct overall + tall/short success (bucketed by
  OBSERVED height) — i.e. the real baseline number on gym.make scenes, replacing the
  off-protocol 2% (which used eval_robocasa's 100000+ seeds).
