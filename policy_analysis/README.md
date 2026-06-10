# RoboCasa policy-analysis harness

Quantitative + qualitative triage of a policy on a RoboCasa task. Turns "the
model is bad" into "it fails on **these objects / in this spatial region / these
scenes**" — the weak-region signal that drives *targeted* data collection.

This is step 5.4 of the project (see project memory): get a quant+qual read so
you don't waste time on very-bad or very-good models, and identify what data to
add next.

## Files
- `rollout_eval.py` — run a policy, log per-episode success + metadata, save videos, write `metrics.json` + `report.txt`.
- `analysis.py` — pure aggregation logic (overall + by object category / spatial region / layout / style, Wilson CIs, weakest-bucket detection). Simulator-independent.
- `policies.py` — `RandomPolicy` (self-test) + `CheckpointPolicyStub` (adapter template for a real model).
- `test_analysis.py` — unit tests for the analysis logic.

## Quick self-test (no checkpoint, runs on the 4070 box)
```sh
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  python rollout_eval.py --task PickPlaceCounterToSink \
  --n 3 --steps 25 --policy random --out-dir ./results/smoke
python test_analysis.py            # 5/5 should pass
```

## Real evaluation run
```sh
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  python rollout_eval.py --task PickPlaceCounterToSink \
  --n 100 --steps 500 --policy random --save-videos failure \
  --out-dir ./results/baseline
```
Use enough rollouts that each bucket has ≥3 samples (weakest-bucket detection
abstains below that). `--steps` should be the task horizon for a fair read.

Outputs:
- `metrics.json` — full machine-readable results + every episode record.
- `report.txt` — human-readable success table, weakest first.
- `videos/{success,failure}/epNNN_<category>.mp4` — qualitative triage.

## Plugging in a real model (Diffusion Policy / GR00T / pi0) on the H100
1. Implement `CheckpointPolicyStub._load` and `__call__` in `policies.py`:
   map RoboCasa obs (camera images named in `camera_names` + proprio) to your
   model's input, run it, return a 12-D action in the env action space.
2. Run with `--policy checkpoint --checkpoint <path>`.

The action layout (12-D) is: `[eef_pos(3), eef_rot(3), gripper(1), base_motion(4), control_mode(1)]`
(see `robocasa.utils.env_utils.convert_action`).

## Notes / caveats
- Spatial regions are tertiles of the object's init position **relative to the
  robot base**, so they're roughly comparable across scenes. For cross-run
  comparison, pass fixed `x_edges`/`y_edges` to `analysis.assign_regions`.
- The official DP fork's `eval_robocasa.py` + `get_eval_stats.py` give overall
  success + videos; this harness adds the **per-bucket breakdown** on top, which
  is what the targeted-data experiment needs.
- `--save-videos failure` keeps disk use down on big runs; use `all` to inspect
  successes too, `none` to skip rendering for speed.
