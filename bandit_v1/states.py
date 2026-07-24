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


def restore(env, start_dir):
    """env.reset_to() the saved start in `start_dir`. Mirrors
    check_train_eval_disjoint.py's `env.reset(); env.reset_to(initial_state)`
    pattern (reset_to's own internal reset(unset_ep_meta=False) call is not a
    substitute for this on a freshly-constructed, never-reset env). Returns the
    post-restore observation dict."""
    start_dir = Path(start_dir)
    states_arr = np.load(start_dir / "state.npz")["states"]
    model_xml = (start_dir / "model.xml").read_text()
    ep_meta_json = (start_dir / "ep_meta.json").read_text()  # reset_to() json.loads's this itself
    env.reset()
    return env.reset_to({"states": states_arr, "model": model_xml, "ep_meta": ep_meta_json})


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
