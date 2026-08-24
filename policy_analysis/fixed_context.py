"""Fixed-context env config for PickPlaceCounterToSink: pin layout, style,
object start placement, and robot base so episodes vary ONLY by object.

Usage (must be imported in EVERY process that creates new episodes —
eval, demo collection, mimicgen; the monkeypatch is not stored in hdf5):

    import fixed_context
    # gym / create_env path (rollout_eval, record_pi0_demos, capture_rollouts):
    env = gym.make("robocasa/PickPlaceCounterToSink", seed=seed,
                   obj_groups="cup", **fixed_context.FIXED_ENV_KWARGS)
    # direct robosuite.make path (build_eval_set) — split kwarg must be OMITTED
    # (Kitchen.__init__ has no `split`; create_env-level only):
    env = robosuite.make(env_name="PickPlaceCounterToSink", robots="PandaOmron",
                         seed=seed, obj_groups="cup", **fixed_context.FIXED_ENV_KWARGS_RS, ...)

Verified (5-seed smoke test): layout/style/sink/robot-base identical across
seeds; object XY spread ~2 cm; object instance varies with seed. For one exact
instance pass obj_groups="/abs/path/to/model.xml". Details:
weakregion/FACTOR_ANALYSIS_REPORT.md section 6.
"""

from robocasa.environments.kitchen.atomic.kitchen_pick_place import PickPlaceCounterToSink

# Any pair with both ids in 11..60 (train split; 1-10 is the test split).
FIXED_LAYOUT_STYLE = [[11, 11]]

_orig_get_obj_cfgs = PickPlaceCounterToSink._get_obj_cfgs


def _pinned_get_obj_cfgs(self):
    cfgs = _orig_get_obj_cfgs(self)
    for cfg in cfgs:
        p = cfg["placement"]
        if cfg["name"] == "obj":
            p["sample_region_kwargs"]["loc"] = "left"  # kill left/right-of-sink coin flip
            p["size"] = (0.02, 0.02)  # ~point start; widen to (0.15, 0.20) to keep interior variation
            p["pos"] = ("ref", -1.0)
            p["ensure_object_boundary_in_range"] = False  # tiny region raises PlacementError otherwise
            p["rotation"] = 0.0  # pin yaw (default is +/- pi/4)
        elif cfg["name"] == "distr_counter":
            cfg["obj_groups"] = "bowl"  # distractors are otherwise re-sampled from ALL categories
        elif cfg["name"] == "distr_sink":
            cfg["obj_groups"] = "sponge"
    return cfgs


PickPlaceCounterToSink._get_obj_cfgs = _pinned_get_obj_cfgs

# For gym.make / create_env: split=None is REQUIRED — split='pretrain' silently
# overrides layout_and_style_ids (env_utils.py:86-104).
FIXED_ENV_KWARGS = dict(
    split=None,
    obj_instance_split="pretrain",  # keep instance-level train/test disjointness
    layout_and_style_ids=FIXED_LAYOUT_STYLE,
    robot_spawn_deviation_pos_x=0.0,
    robot_spawn_deviation_pos_y=0.0,
    robot_spawn_deviation_rot=0.0,
)

# For direct robosuite.make: identical but WITHOUT split (unknown kwarg -> TypeError).
FIXED_ENV_KWARGS_RS = {k: v for k, v in FIXED_ENV_KWARGS.items() if k != "split"}
