"""Shared rollout engine (bandit_v1 Task 4): the ONE code path diagnosis, baseline,
and pull evals all run through. Rolls a served pi0 policy (openpi_client websocket)
out on saved-state starts (states.py) and appends every episode to the shared
ledger (ledger.py, table "episodes").

Reconciling the two env code paths (see states.py's make_env vs
make_env_for_rollout docstrings): analyze_pi0_weakregions.py -- the pattern this
file's obs->policy mapping, action-application loop, horizon, and failure-stage
thresholds are all copied from -- drives the env through
robocasa.wrappers.gym_wrapper.RoboCasaGymEnv (gym.make(f"robocasa/{task}")). Our
env instead comes from states.make_env_for_rollout() (the robomimic
`create_env_for_data_processing` path), so it composes with states.restore()'s
reset_to()-based saved-state restoration, which the gym env does not support. The
two wrappers expose different obs key names for the SAME underlying robosuite
quantities:
  - images:   gym "video.robot0_agentview_left"           <-> robomimic "robot0_agentview_left_image"
  - state:    gym "state.end_effector_position_relative"  <-> robomimic "robot0_base_to_eef_pos" (etc.)
  - language: gym "annotation.human.task_description"      <-> env.env.get_ep_meta()["lang"]
_policy_element below reads the robomimic-style keys but builds the EXACT same
"observation/image" / "observation/wrist_image" / "observation/right_image" /
"observation/state" / "prompt" dict analyze_pi0_weakregions.py's
build_pi0_element sends -- the server contract is fixed, only the obs source
differs.

Actions are the other half of the reconciliation. analyze_pi0_weakregions.py calls
`env.step(convert_action(a))` because RoboCasaGymEnv.step() takes an ACTION DICT
and internally reassembles it into the flat robosuite action array via
PandaOmronKeyConverter.unmap_action() + a per-robot composite-controller loop
(robocasa/wrappers/gym_wrapper.py:355-372) -- notably discretizing
gripper_close/control_mode to +-1, not a plain slice passthrough. Our robomimic
env's step() takes that flat array directly (no dict). _apply_action replicates
gym_wrapper.py's exact assembly logic (same helper functions, same thresholding)
directly against env.env (the same underlying robosuite object the gym wrapper
would also hold), then calls env.step() with the result: semantically identical
actions, different plumbing. Verified against a live env (see task-4-report.md)
that this reproduces gym_wrapper.py's per-part action_split_indexes exactly.
"""
import collections
import time
import uuid
from pathlib import Path

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _wcp
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.utils.env_utils import convert_action
from robocasa.wrappers.gym_wrapper import PandaOmronKeyConverter
from robosuite.controllers.composite.composite_controller import HybridMobileBase

from . import config, ledger, states

RESIZE_SIZE = 224   # pi0 image side, matches analyze_pi0_weakregions.py's --resize_size default
REPLAN_STEPS = 5    # openpi_client action-chunk replan cadence (brief's interface)

GRASP_LIFT_M = 0.05   # max_lift threshold for "grasped" (analyze_pi0_weakregions.py:186)
SINK_DIST_M = 0.15    # min_sink_dist threshold for "near_sink" (analyze_pi0_weakregions.py:187)
EE_REACH_M = 0.10     # ee_min_dist threshold splitting never_reached / reached_no_grasp (brief)

TARGET = states.TARGET


def _policy_element(obs, lang, resize):
    """pi0 server input dict -- same key set/order as analyze_pi0_weakregions.py's
    build_pi0_element, sourced from the robomimic env's raw robosuite-style obs
    keys (see module docstring) instead of the gym wrapper's "video."/"state."
    keys."""
    def prep(k):
        return image_tools.convert_to_uint8(
            image_tools.resize_with_pad(np.ascontiguousarray(obs[k]), resize, resize))
    return {
        "observation/image": prep("robot0_agentview_left_image"),
        "observation/wrist_image": prep("robot0_eye_in_hand_image"),
        "observation/right_image": prep("robot0_agentview_right_image"),
        "observation/state": np.concatenate((
            obs["robot0_base_to_eef_pos"],
            obs["robot0_base_to_eef_quat"],
            obs["robot0_base_pos"],
            obs["robot0_base_quat"],
            obs["robot0_gripper_qpos"]), axis=0).astype(np.float64),
        "prompt": lang,
    }


def _policy_action(client, obs, plan, lang, resize, replan):
    if not plan:
        plan.extend(client.infer(_policy_element(obs, lang, resize))["actions"][:replan])
    return plan.popleft()


def _apply_action(env, a):
    """Flat 12-dim pi0 action -> robosuite composite-controller action array,
    replicating RoboCasaGymEnv.step()'s dict reassembly (convert_action +
    PandaOmronKeyConverter.unmap_action + per-robot part-controller loop) directly
    against env.env, since our robomimic env.step() takes the flat array, not the
    action dict the gym wrapper's step() accepts. Returns env.step()'s result."""
    action_dict = PandaOmronKeyConverter.unmap_action(convert_action(a))
    raw_env = env.env
    env_action = []
    for robot in raw_env.robots:
        cc = robot.composite_controller
        pf = robot.robot_model.naming_prefix
        part_action = np.zeros(cc.action_limits[0].shape)
        for part_name in cc.part_controllers:
            start_idx, end_idx = cc._action_split_indexes[part_name]
            part_action[start_idx:end_idx] = action_dict.pop(f"{pf}{part_name}")
        if isinstance(cc, HybridMobileBase):
            part_action[-1] = action_dict.pop(f"{pf}base_mode")
        env_action.append(part_action)
    assert not action_dict, f"unmapped action keys left over: {action_dict}"
    return env.step(np.concatenate(env_action))


def _failure_stage(success, grasped, near_sink, ee_min_dist):
    """5-stage failure signature: analyze_pi0_weakregions.py's 4-way classifier
    (success / fail_no_grasp / fail_grasped_no_transport / fail_reached_sink_no_place)
    with fail_no_grasp split by ee_min_dist into never_reached / reached_no_grasp
    (brief's Task 4 interface)."""
    if success:
        return "success"
    if not grasped:
        return "never_reached" if ee_min_dist > EE_REACH_M else "reached_no_grasp"
    if not near_sink:
        return "fail_grasped_no_transport"
    return "fail_reached_sink_no_place"


def _rollout_one(env, client, start_dir, horizon):
    """Restore `start_dir` into `env` and roll the served policy out for up to
    `horizon` steps. Returns the per-episode outcome dict (no identity/ledger
    fields -- run() adds those)."""
    obs = states.restore(env, start_dir)
    rs = env.env
    lang = rs.get_ep_meta().get("lang", "")

    def obj_xyz():
        return np.array(rs.sim.data.body_xpos[rs.obj_body_id[TARGET]])
    sink_xy = (np.array(rs.sim.data.body_xpos[rs.obj_body_id["distr_sink"]][:2])
               if "distr_sink" in getattr(rs, "obj_body_id", {}) else None)

    init_z = float(obj_xyz()[2])
    max_lift, min_sink_dist, ee_min_dist = 0.0, float("inf"), float("inf")
    plan = collections.deque()
    success, steps = False, horizon

    for t in range(horizon):
        a = _policy_action(client, obs, plan, lang, RESIZE_SIZE, REPLAN_STEPS)
        obs, _r, _done, info = _apply_action(env, a)
        pz = obj_xyz()
        max_lift = max(max_lift, float(pz[2]) - init_z)
        ee_pos = np.asarray(obs["robot0_eef_pos"], dtype=float)
        ee_min_dist = min(ee_min_dist, float(np.linalg.norm(ee_pos - pz)))
        if sink_xy is not None:
            min_sink_dist = min(min_sink_dist, float(np.linalg.norm(pz[:2] - sink_xy)))
        if info["is_success"]["task"]:
            success, steps = True, t + 1
            break

    grasped = max_lift > GRASP_LIFT_M
    near_sink = sink_xy is not None and min_sink_dist < SINK_DIST_M
    stage = _failure_stage(success, grasped, near_sink, ee_min_dist)
    return dict(
        success=bool(success),
        failure_stage=stage,
        ee_min_dist=(round(ee_min_dist, 4) if ee_min_dist != float("inf") else None),
        max_lift=round(max_lift, 4),
        min_sink_dist=(round(min_sink_dist, 4) if min_sink_dist != float("inf") else None),
        steps=int(steps),
    )


def run(policy_host, policy_port, start_dirs, repeats, phase, policy_id, arm=None,
        pull_id=None, skip_pairs=None):
    """Roll the policy served at (policy_host, policy_port) out on every start in
    `start_dirs`, `repeats` times each. Appends every episode to ledger table
    "episodes" as it completes (crash-safe -- mirrors the rest of bandit_v1's
    per-item ledger writes, e.g. validate_reset.py's per-start gate) and returns
    the same rows as a list of dicts.

    `skip_pairs` (optional, default None -- identical behavior to before this
    parameter existed): a container supporting `in` of (start_id, repeat_idx)
    tuples (start_id == Path(start_dir).name, matching the ledger's own
    start_id column) to skip entirely -- no `_rollout_one` call, no row built,
    no ledger append. This is bandit_v1's resume mechanism (run_diagnosis.py):
    the caller queries the ledger for already-completed (start_id, repeat_idx)
    pairs before each chunk and passes them here so a rerun after a crash
    never redoes a completed episode, at per-episode (not just per-chunk)
    granularity."""
    horizon = int(get_task_horizon(config.TASK))
    client = _wcp.WebsocketClientPolicy(policy_host, policy_port)
    env = states.make_env_for_rollout()
    rows = []
    try:
        env.reset()  # env must have been reset once before the first restore (states.restore also resets itself)
        for start_dir in start_dirs:
            start_dir = Path(start_dir)
            start_id = start_dir.name
            feats = states.start_features(start_dir)
            for repeat_idx in range(repeats):
                if skip_pairs is not None and (start_id, repeat_idx) in skip_pairs:
                    continue
                t0 = time.time()
                result = _rollout_one(env, client, start_dir, horizon)
                wall_time_s = round(time.time() - t0, 3)
                row = {
                    "episode_id": f"{phase}_{policy_id}_{start_id}_r{repeat_idx}_{uuid.uuid4().hex[:8]}",
                    "phase": phase,
                    "arm": arm,
                    "pull_id": pull_id,
                    "policy_id": policy_id,
                    "start_id": start_id,
                    "repeat_idx": repeat_idx,
                    **result,
                    "wall_time_s": wall_time_s,
                    **feats,
                }
                rows.append(row)
                ledger.append_rows("episodes", [row])
    finally:
        states.close_env(env)
    return rows
