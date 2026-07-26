"""Saved-state capture/restore for robocasa env starts (bandit_v1 Task 3).

Design: weakregion/BANDIT_V1_DESIGN.md section 1 item 4 (Eval set E) + the day-one
gate. Mirrors policy_analysis/check_train_eval_disjoint.py's env construction (the
robomimic `create_env_for_data_processing` path) and
policy_analysis/analyze_pi0_weakregions.py's episode-feature extraction.

A "start" is a directory holding everything needed to deterministically restore a
robocasa env to one particular init scene via `env.reset_to`, INDEPENDENT of
reproducing the scene by re-seeding. This distinction matters: per the design doc,
same-seed scene replay across processes only reproduces the exact scene ~2/3 of the
time (recurring ~1.17 m left/right-of-sink offsets, occasional category flips), so a
seed list alone is not a frozen eval set. `validate_reset.py` is the gate that
verifies `reset_to` restoration does not have the same problem.

Layout of a start directory (out_dir passed to capture_start):
    state.npz              -- flattened mujoco state vector, key "states"
    model.xml               -- mujoco scene xml (str)
    ep_meta.json             -- robocasa episode metadata (dict, json-serialized)
    fingerprint.json          -- fingerprint(env) snapshot taken right after capture

TARGET is always "obj": bandit_v1's single task (PickPlaceCounterToSink) names its
one manipulated object "obj" in every ep_meta / obj_body_id (see
check_train_eval_disjoint.py, analyze_pi0_weakregions.py).

Category aliasing (task 3 fix, see .superpowers/sdd/task-3-report.md): 17/1516
robocasa mjcf instances are registered under two overlapping category names, so
capture's forward-sampled `category` can disagree with reset_to's reverse-lookup
`category` for the SAME instance. `instance` (the mjcf path) is the identity
ground truth; `category` is canonicalized via categories.canonical_category
everywhere it is compared or emitted (fingerprint_diff, start_features) so this
no longer registers as a mismatch.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

import robocasa.utils.robomimic.robomimic_dataset_utils as DatasetUtils
import robocasa.utils.robomimic.robomimic_env_utils as EnvUtils

from . import categories, config

TARGET = "obj"


def make_env(seed=None):
    """One canonical env constructor: robomimic `create_env_for_data_processing`,
    camera/obs config copied verbatim from check_train_eval_disjoint.py. No cameras
    are requested (camera_names=[]) -- fingerprint/start_features never touch
    pixels, only sim state, so rendering would be pure overhead.

    `seed` is forwarded into env_kwargs["seed"] by EnvUtils.create_env_for_data_processing
    and controls robosuite's scene-randomization RNG (object placement, layout/style
    pick, robot base pose) for the FIRST reset() after construction.
    """
    env_meta = DatasetUtils.get_env_metadata_from_dataset(
        dataset_path=str(config.ENV_ARGS_HDF5))
    return EnvUtils.create_env_for_data_processing(
        env_meta=env_meta, camera_names=[], camera_height=84, camera_width=84,
        reward_shaping=False, seed=seed)


ROLLOUT_CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
ROLLOUT_CAMERA_SIZE = 256  # matches robocasa.wrappers.gym_wrapper.PandaOmronKeyConverter.get_camera_config


def make_env_for_rollout(seed=None):
    """Second env constructor (bandit_v1 Task 4), used only by rollout.py's
    policy-serving loop. make_env() passes camera_names=[] because fingerprint /
    start_features never touch pixels (pure overhead there); a policy rollout needs
    the pi0 websocket server's 3 cameras instead -- robot0_agentview_left/right +
    robot0_eye_in_hand at 256x256, the exact rig
    robocasa.wrappers.gym_wrapper.PandaOmronKeyConverter.get_camera_config uses for
    the gym env pi0 is normally served from (analyze_pi0_weakregions.py's path).

    Still built on the robomimic `create_env_for_data_processing` path (not
    gym.make), NOT the gym wrapper, so restore()'s reset_to()-based saved-state
    restoration and the env.env / get_ep_meta() / obj_body_id access states.py and
    rollout.py rely on keep working unchanged; only the returned obs dict gains
    image keys plus the raw robosuite-style low-dim keys (robot0_eef_pos,
    robot0_gripper_qpos, robot0_base_to_eef_pos, ...) that rollout.py maps into the
    pi0 server's expected observation/state contract (the gym wrapper's obs uses
    differently-named "video.*"/"state.*" keys for the same underlying
    quantities -- see rollout.py's module docstring)."""
    env_meta = DatasetUtils.get_env_metadata_from_dataset(
        dataset_path=str(config.ENV_ARGS_HDF5))
    return EnvUtils.create_env_for_data_processing(
        env_meta=env_meta, camera_names=list(ROLLOUT_CAMERAS),
        camera_height=ROLLOUT_CAMERA_SIZE, camera_width=ROLLOUT_CAMERA_SIZE,
        reward_shaping=False, seed=seed)


def close_env(env):
    """EnvRobocasa (the robomimic wrapper) has no .close(); the underlying
    robosuite env does."""
    raw = getattr(env, "env", env)
    if hasattr(raw, "close"):
        raw.close()


def _target_cfg_info(ep_meta):
    """(category, mjcf_path) for the TARGET object, from an ep_meta dict (works for
    both a live env's ep_meta dict and one parsed back from ep_meta.json)."""
    for cfg in ep_meta.get("object_cfgs", []):
        if cfg.get("name") == TARGET:
            info = cfg.get("info", {}) or {}
            return info.get("cat", "unknown"), info.get("mjcf_path", "unknown")
    return "unknown", "unknown"


def fingerprint(env):
    """Snapshot of the identifying features of the CURRENT scene in `env` (call
    right after reset() or reset_to()). Keys: category, instance, layout_id,
    style_id, obj_xyz (rounded 1e-4), base_xy."""
    rs = env.env  # unwrap robomimic EnvRobocasa -> underlying robosuite env
    ep = rs.get_ep_meta()
    cat, instance = _target_cfg_info(ep)
    obj_xyz = np.asarray(rs.sim.data.body_xpos[rs.obj_body_id[TARGET]], dtype=float)
    base_xy = np.asarray(ep.get("init_robot_base_pos", [0.0, 0.0, 0.0])[:2], dtype=float)
    return {
        "category": cat,
        "instance": instance,
        "layout_id": ep.get("layout_id"),
        "style_id": ep.get("style_id"),
        "obj_xyz": [round(float(v), 4) for v in obj_xyz],
        "base_xy": [round(float(v), 4) for v in base_xy],
    }


def fingerprint_diff(a, b):
    """List of keys where fingerprints `a` and `b` disagree (empty == exact match).
    Used by validate_reset.py's cross-process gate to report exactly which fields
    mismatched, not just pass/fail.

    `instance` (the mjcf path) is the primary identity signal and is always
    compared verbatim. `category` is compared after canonicalizing both sides via
    categories.canonical_category: forward sampling (capture_start's env.reset())
    and env.reset_to()'s reverse mjcf_path->category lookup (restore) can
    legitimately disagree on the category LABEL for the ~1% of instances
    registered under two overlapping category names (config.CATEGORY_ALIASES) --
    the mjcf path is the ground-truth identity, not either code path's raw label,
    so an alias-only disagreement here is not a real mismatch. Every other key
    (layout_id, style_id, obj_xyz, base_xy, ...) compares as-is, unchanged."""
    keys = sorted(set(a) | set(b))
    diffs = []
    for k in keys:
        va, vb = a.get(k), b.get(k)
        if k == "category":
            va = categories.canonical_category(va)
            vb = categories.canonical_category(vb)
        if va != vb:
            diffs.append(k)
    return diffs


def capture_start(seed, out_dir):
    """Create env at `seed` (robomimic path, see make_env), reset once, serialize
    state+model+ep_meta+fingerprint to out_dir. Returns the fingerprint dict."""
    out_dir = Path(out_dir)
    env = make_env(seed=seed)
    try:
        env.reset()
        state = env.get_state()  # {"model": xml str, "states": ndarray, "ep_meta": json str}
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "state.npz", states=state["states"])
        (out_dir / "model.xml").write_text(state["model"])
        ep_meta = json.loads(state["ep_meta"])
        (out_dir / "ep_meta.json").write_text(json.dumps(ep_meta, indent=2, default=str))
        fp = fingerprint(env)
        (out_dir / "fingerprint.json").write_text(json.dumps(fp, indent=2))
        return fp
    finally:
        close_env(env)


def _model_hash(model_xml):
    """sha256 of a model.xml string's exact bytes. Used to detect "this start's
    compiled scene is byte-identical to what's already loaded in env.sim.model" --
    the condition restore()'s warm-model fast path requires."""
    return hashlib.sha256(model_xml.encode("utf-8")).hexdigest()


def _restore_plan(env, model_hash, warm):
    """Pure decision logic for restore()'s two speedups (bandit_v1 rollout-speedup
    task): given the env object's own bookkeeping attributes (`_bandit_initialized`,
    `_bandit_model_hash`, both absent on a never-restored env) and the caller's
    `warm` flag, decide which of three restore strategies to use. Kept separate
    from restore() itself (which does the actual mujoco/robomimic calls) so this
    decision can be unit-tested against a bare monkeypatched stand-in for `env` --
    no live env, no mujoco -- covering exactly the three cases that matter:
      - "full_with_prereset": a fresh/never-restored env (no `_bandit_initialized`),
        or warm=False (the gate's cold-restore cross-check, and reset_to's/rollout
        speedup (2)'s explicit fallback) -- always pays the explicit throwaway
        env.reset() before env.reset_to(), exactly the pre-existing behavior.
      - "full_no_prereset": env already initialized, warm=True, but the incoming
        start's model.xml hash does not match the last-restored one (a genuinely
        different scene) -- skips the throwaway pre-reset (speedup (2): reset_to's
        own internal reset(unset_ep_meta=False) already suffices once *some* reset
        has happened on this env, per the original restore() docstring/gate
        history) but still does the full env.reset_to() (both of its internal
        model recompiles), since the compiled model must actually change.
      - "fast": env already initialized, warm=True, and the incoming start's
        model.xml hash matches the last-restored one -- i.e. this is a REPEAT of
        the same start (or, in principle, coincidentally-identical scenes). No
        recompile is needed at all; see _restore_warm's docstring for exactly what
        this path does and does not reproduce from a full reset_to().
    """
    initialized = getattr(env, "_bandit_initialized", False)
    if warm and initialized and getattr(env, "_bandit_model_hash", None) == model_hash:
        return "fast"
    if (not warm) or (not initialized):
        return "full_with_prereset"
    return "full_no_prereset"


def _restore_full(env, states_arr, model_xml, ep_meta_json):
    """The original (pre-speedup) reset_to()-based restore, unchanged: both of
    reset_to's internal model recompiles (the ep_meta-driven `_load_model` +
    `_initialize_sim` inside its `self.reset(unset_ep_meta=False)` call, and the
    exact-xml `reset_from_xml_string`) run unconditionally. Always correct;
    used whenever no cached compiled model can be reused."""
    return env.reset_to({"states": states_arr, "model": model_xml, "ep_meta": ep_meta_json})


def _restore_warm(env, states_arr, ep_meta_json):
    """Warm-model fast path (rollout-speedup (3)): used only when the incoming
    start's model.xml is byte-identical (by sha256) to the one already compiled
    into `env.env.sim.model` -- i.e. a repeat of the same start. Skips BOTH of
    reset_to's model recompiles entirely and instead:

      1. Reapplies ep_meta (set_attrs_from_ep_meta / set_ep_meta) -- lang,
         layout/style ids, gen_textures record, saved robot base target, etc.
         (mirrors reset_to's first step).
      2. Runs ONE soft (non-recompiling) reset by forcing
         `env.env.deterministic_reset = True` around a normal
         `env.reset(unset_ep_meta=False)` call, then setting it back to False
         in a `finally` (review fix: if `env.reset()` itself raises -- a bad
         ep_meta, a controller error, anything -- the flag must not leak
         `True` forever; a stuck `deterministic_reset=True` would silently
         force EVERY subsequent reset() on this same env, warm or cold, real
         pull or eval, onto the soft/no-recompile branch, corrupting every
         restore after the exception until the process is restarted)
         (exactly mirroring reset_from_xml_string's own use of this flag, see
         robosuite/environments/base.py). With `deterministic_reset=True` and
         `sim` already built, robosuite's own reset() takes its "soft" branch
         (`self.sim.reset()`, i.e. mj_resetData -- no `_load_model()` /
         `_initialize_sim()` recompile) but still runs `_reset_internal()`
         unconditionally. This step is NOT optional: `_reset_internal()` is what
         resets each robot's composite-controller internal state
         (robosuite/robots/robot.py: `robot.reset()` calls
         `composite_controller.update_state()` + `.reset()` -- interpolators,
         ramped/filtered goals, etc.) and repositions the robot base from
         `ep_meta["init_robot_base_pos"/"init_robot_base_ori"]`. None of that is
         touched by `set_state_from_flattened`, which only overwrites MuJoCo's
         raw (time, qpos, qvel) -- so a bare
         `sim.reset() + set_state_from_flattened + forward` with NO reset() call
         at all (the naive reading of "state-set only") would leave a REPEAT's
         controller state stale from wherever the previous rollout's real policy
         actions left it, instead of matching the freshly-restored pose. This was
         found by inspection (robot.reset() is unconditionally invoked from
         `_reset_internal()`, itself unconditionally invoked from every reset()
         variant regardless of hard/soft branching) and is the "dependency the
         recompile provides that state-set alone does not" the design flagged as
         a risk to check for. Using robosuite's own deterministic_reset soft-reset
         mechanism (rather than hand-calling `robot.reset()` directly) reuses
         already-tested upstream code for this instead of re-implementing it.
         Nothing here recompiles the mujoco model, so per-episode textures /
         object id mappings / body addresses (baked into the compiled model,
         unchanged since we are, by construction, restoring the SAME model.xml
         that's already loaded) are untouched and still correct.
      3. Applies the exact captured state via set_state_from_flattened + forward,
         overwriting whatever step 2's soft reset put in qpos/qvel (matching the
         saved snapshot exactly, same as reset_to's own tail).
      4. update_sites/update_state + get_observation, matching reset_to's tail.
    """
    raw = env.env
    ep_meta = json.loads(ep_meta_json)
    if hasattr(raw, "set_attrs_from_ep_meta"):
        raw.set_attrs_from_ep_meta(ep_meta)
    elif hasattr(raw, "set_ep_meta"):
        raw.set_ep_meta(ep_meta)

    raw.deterministic_reset = True
    try:
        env.reset(unset_ep_meta=False)
    finally:
        raw.deterministic_reset = False

    raw.sim.set_state_from_flattened(states_arr)
    raw.sim.forward()
    if hasattr(raw, "update_sites"):
        raw.update_sites()
    if hasattr(raw, "update_state"):
        raw.update_state()
    return env.get_observation()


def restore(env, start_dir, warm=True):
    """Restore the saved start in `start_dir` into `env`. Mirrors
    check_train_eval_disjoint.py's `env.reset(); env.reset_to(initial_state)`
    pattern (reset_to's own internal reset(unset_ep_meta=False) call is not a
    substitute for the explicit pre-reset on a freshly-constructed, never-reset
    env) -- EXCEPT that, per the ledger cost analysis (each episode was paying
    this throwaway pre-reset PLUS reset_to's own two internal recompiles, ~16.2s
    fixed cost/episode), two speedups are applied once it is safe to do so:

      (2) The explicit pre-reset above is skipped once `env` has completed at
          least one reset ever (tracked via `env._bandit_initialized`, set by
          this function) -- the pre-reset is genuinely required only for a
          never-reset env, per the original gate history; it is pure waste on
          every subsequent call, since reset_to's own internal reset already
          establishes a valid scene before overwriting it.
      (3) If, in addition, the incoming start's model.xml hash matches the last
          -restored one (`env._bandit_model_hash`, also set by this function) --
          i.e. this is a repeat of the same start -- reset_to's two internal
          model recompiles are skipped entirely via `_restore_warm` (see its
          docstring for exactly what is and is not reproduced).

    `warm=False` disables both speedups unconditionally (always pre-reset, always
    the full reset_to path) -- used by validate_reset.py's --warm-check gate as
    the "cold restore" comparison arm, and to preserve the original behavior for
    any caller that needs it. `warm=True` (the default) is what rollout.py uses.

    Returns the post-restore observation dict."""
    start_dir = Path(start_dir)
    states_arr = np.load(start_dir / "state.npz")["states"]
    model_xml = (start_dir / "model.xml").read_text()
    ep_meta_json = (start_dir / "ep_meta.json").read_text()  # reset_to() json.loads's this itself
    model_hash = _model_hash(model_xml)

    plan = _restore_plan(env, model_hash, warm)
    if plan == "fast":
        obs = _restore_warm(env, states_arr, ep_meta_json)
    else:
        if plan == "full_with_prereset":
            env.reset()
        obs = _restore_full(env, states_arr, model_xml, ep_meta_json)
        env._bandit_model_hash = model_hash

    env._bandit_initialized = True
    return obs


_CATEGORY_HW = None


def _category_hw(category):
    """(h, w) for `category`, sourced from FX_POOL_JSON's per-category table -- the
    SAME per-category h/w used by pool.py's pool table (see its docstring), so
    eval-side and pool-side features join on identical (category -> h, w). Not
    recomputed from live object geometry: a start_dir has no live env to query, and
    the design's clustering descriptor is explicitly the eval/pool feature
    intersection, not per-instance geometry."""
    global _CATEGORY_HW
    if _CATEGORY_HW is None:
        cats = json.load(open(config.FX_POOL_JSON))["cats"]
        _CATEGORY_HW = {c["name"]: (c["h"], c["w"]) for c in cats}
    return _CATEGORY_HW.get(category, (None, None))


def _side(x_rel, y_rel):
    """Canonical side encoding (int in {-1, +1}), dominant-axis rule per
    bandit_v1/pool.py's docstring: side == sign(x_rel) if |x_rel| >= |y_rel| else
    sign(y_rel) -- NOT sign(x_rel) or sign(y_rel) alone."""
    v = x_rel if abs(x_rel) >= abs(y_rel) else y_rel
    return 1 if v >= 0 else -1


def start_features(start_dir):
    """(category, h, w, x_rel, y_rel, side, yaw, layout_id, style_id) for a
    captured start, parsed from ep_meta.json + fingerprint.json (no live env
    needed). `category` is canonicalized via categories.canonical_category before
    being returned or used for the h/w join (task 3 fix: fingerprint.json's raw
    category can be either alias name for the 17 dual-registered instances, see
    config.CATEGORY_ALIASES; canonicalizing here keeps this in sync with pool.py's
    pool-table category column, which is canonicalized the same way). h/w come
    from the category table (see _category_hw); x_rel/y_rel are obj_xyz - base_xy
    (matching check_train_eval_disjoint.py's descriptor); side is the
    dominant-axis rule; yaw is the robot base spawn yaw (init_robot_base_ori[2]),
    the one per-episode orientation ep_meta actually carries."""
    start_dir = Path(start_dir)
    ep_meta = json.loads((start_dir / "ep_meta.json").read_text())
    fp = json.loads((start_dir / "fingerprint.json").read_text())

    category = categories.canonical_category(fp.get("category", "unknown"))
    h, w = _category_hw(category)

    obj_xyz = fp["obj_xyz"]
    base_xy = fp["base_xy"]
    x_rel = round(float(obj_xyz[0]) - float(base_xy[0]), 4)
    y_rel = round(float(obj_xyz[1]) - float(base_xy[1]), 4)
    side = _side(x_rel, y_rel)

    base_ori = ep_meta.get("init_robot_base_ori", [0.0, 0.0, 0.0])
    yaw = float(base_ori[2])

    return {
        "category": category,
        "h": h,
        "w": w,
        "x_rel": x_rel,
        "y_rel": y_rel,
        "side": side,
        "yaw": yaw,
        "layout_id": ep_meta.get("layout_id", fp.get("layout_id")),
        "style_id": ep_meta.get("style_id", fp.get("style_id")),
    }
