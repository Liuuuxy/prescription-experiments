"""Pure-part tests for bandit_v1/states.py (task 3, step 2; rollout-speedup
hash-cache decision added later).

Covers only the parts that don't touch a live robocasa env: fingerprint_diff
(dict comparison) and start_features (ep_meta.json + fingerprint.json parsing).
The env-touching parts (capture_start, restore, make_env, fingerprint) are
validated end-to-end by the validate_reset.py gate run (task 3, step 4), per the
brief -- TDD applies only to these pure parts. `_restore_plan` (restore()'s
hash-cache/pre-reset decision logic) is likewise pure -- no mujoco/robosuite
object is touched, just plain attributes on a stand-in "env" object -- so it is
unit-tested here too, against a bare `types.SimpleNamespace`.
"""
import json
import types

import numpy as np
import pytest

from bandit_v1 import states


def test_fingerprint_diff_empty_when_equal():
    a = {"category": "jar", "layout_id": 3, "obj_xyz": [1.0, 2.0, 0.5]}
    assert states.fingerprint_diff(a, dict(a)) == []


def test_fingerprint_diff_reports_one_mismatched_key():
    a = {"category": "jar", "layout_id": 3, "style_id": 7}
    b = {"category": "jar", "layout_id": 4, "style_id": 7}
    assert states.fingerprint_diff(a, b) == ["layout_id"]


def test_fingerprint_diff_reports_multiple_and_missing_keys():
    a = {"category": "jar", "layout_id": 3}
    b = {"category": "mug", "layout_id": 3, "style_id": 7}
    # sorted union of keys that disagree, including a key present in only one dict
    assert states.fingerprint_diff(a, b) == ["category", "style_id"]


def _write_synthetic_start(tmp_path, base_xy, obj_xyz, yaw=1.5707963267948966,
                            layout_id=12, style_id=5, cat="jar"):
    ep_meta = {
        "layout_id": layout_id,
        "style_id": style_id,
        "object_cfgs": [{"name": "obj", "info": {"cat": cat, "mjcf_path": "/x/model.xml"}}],
        "init_robot_base_pos": [base_xy[0], base_xy[1], 0.0],
        "init_robot_base_ori": [0.0, 0.0, yaw],
    }
    fp = {
        "category": cat,
        "instance": "/x/model.xml",
        "layout_id": layout_id,
        "style_id": style_id,
        "obj_xyz": [obj_xyz[0], obj_xyz[1], obj_xyz[2]],
        "base_xy": [base_xy[0], base_xy[1]],
    }
    (tmp_path / "ep_meta.json").write_text(json.dumps(ep_meta))
    (tmp_path / "fingerprint.json").write_text(json.dumps(fp))


def test_start_features_parses_synthetic_ep_meta(tmp_path):
    _write_synthetic_start(tmp_path, base_xy=(1.0, -2.0), obj_xyz=(1.3, -1.9, 0.9))
    feats = states.start_features(tmp_path)

    assert feats["category"] == "jar"
    assert feats["layout_id"] == 12
    assert feats["style_id"] == 5
    assert feats["x_rel"] == round(1.3 - 1.0, 4)
    assert feats["y_rel"] == round(-1.9 - (-2.0), 4)
    # dominant axis here is x (|0.3| > |0.1|) -> side follows sign(x_rel)
    assert feats["side"] == 1
    assert isinstance(feats["side"], int)
    assert feats["yaw"] == 1.5707963267948966
    # h/w come from FX_POOL_JSON's per-category table (jar is a real bandit_v1
    # pool category with a known (h, w), verified in task 2 / test_pool.py).
    assert feats["h"] == 0.194
    assert feats["w"] == 0.201


def test_start_features_side_dominant_axis_rule(tmp_path):
    # |y_rel| > |x_rel| and y negative -> side must follow sign(y_rel), NOT sign(x_rel)
    # (this is the exact distinction bandit_v1/pool.py's docstring calls out).
    _write_synthetic_start(tmp_path, base_xy=(0.0, 0.0), obj_xyz=(0.05, -0.30, 0.9))
    feats = states.start_features(tmp_path)
    assert feats["x_rel"] == 0.05
    assert feats["y_rel"] == -0.30
    assert feats["side"] == -1


def test_start_features_unknown_category_has_none_hw(tmp_path):
    _write_synthetic_start(tmp_path, base_xy=(0.0, 0.0), obj_xyz=(0.1, 0.1, 0.9),
                            cat="not_a_real_category")
    feats = states.start_features(tmp_path)
    assert feats["h"] is None and feats["w"] is None


def test_fingerprint_diff_alias_category_same_instance_compares_equal():
    """Task 3 fix: forward-sampled 'jug_wide_opening' (capture) vs. reverse-lookup
    'jug' (restore) for the SAME mjcf instance is not a real mismatch -- the mjcf
    path is the identity ground truth, category is just an alias of it here (see
    config.CATEGORY_ALIASES)."""
    captured = {
        "category": "jug_wide_opening",
        "instance": "/data/xinyua11/robocasa_pkg/robocasa/models/assets/objects/objaverse/jug/jug_4/model.xml",
        "layout_id": 11, "style_id": 8,
        "obj_xyz": [0.4843, -1.1632, 0.9958], "base_xy": [0.7681, -1.8298],
    }
    restored = dict(captured, category="jug")
    assert states.fingerprint_diff(captured, restored) == []


def test_fingerprint_diff_different_instance_compares_unequal_regardless_of_category():
    """A genuinely different physical instance must still register as a mismatch
    even if both sides happen to canonicalize to the same category label --
    canonicalization must not mask a real identity difference."""
    captured = {
        "category": "jug_wide_opening",
        "instance": "/data/xinyua11/robocasa_pkg/robocasa/models/assets/objects/objaverse/jug/jug_4/model.xml",
        "layout_id": 11, "style_id": 8,
        "obj_xyz": [0.4843, -1.1632, 0.9958], "base_xy": [0.7681, -1.8298],
    }
    restored = dict(
        captured,
        category="jug",  # same canonical category as captured's alias
        instance="/data/xinyua11/robocasa_pkg/robocasa/models/assets/objects/objaverse/jug/jug_1/model.xml",
    )
    diff = states.fingerprint_diff(captured, restored)
    assert "instance" in diff
    assert "category" not in diff


# --- restore()'s hash-cache/pre-reset decision (rollout speedups 2+3) ---------

def test_restore_plan_fresh_env_takes_full_path_with_prereset():
    """A never-restored env (no `_bandit_initialized` attribute at all) must
    always pay the explicit pre-reset, regardless of warm, since the fast path's
    "reuse the already-compiled model" premise cannot hold when nothing has been
    compiled/restored yet on this env."""
    env = types.SimpleNamespace()
    assert states._restore_plan(env, model_hash="abc123", warm=True) == "full_with_prereset"
    assert states._restore_plan(env, model_hash="abc123", warm=False) == "full_with_prereset"


def test_restore_plan_same_hash_after_init_takes_fast_path():
    """Second (or later) restore of the SAME start (matching model.xml hash) on
    an already-initialized env, with warm=True -> the fast path, no recompile."""
    env = types.SimpleNamespace(_bandit_initialized=True, _bandit_model_hash="abc123")
    assert states._restore_plan(env, model_hash="abc123", warm=True) == "fast"


def test_restore_plan_different_hash_takes_full_path_without_prereset():
    """A genuinely different start (model.xml hash differs from the last-restored
    one) on an already-initialized env still needs the full reset_to() path (the
    compiled model must actually change), but the throwaway pre-reset is skipped
    since some reset has already happened on this env."""
    env = types.SimpleNamespace(_bandit_initialized=True, _bandit_model_hash="abc123")
    assert states._restore_plan(env, model_hash="different_hash", warm=True) == "full_no_prereset"


def test_restore_plan_warm_false_forces_full_path_even_with_matching_hash():
    """warm=False (validate_reset.py --warm-check's cold-restore arm) must force
    the full path unconditionally, even when the hash would otherwise qualify for
    the fast path -- this is the gate's clean "old behavior" baseline."""
    env = types.SimpleNamespace(_bandit_initialized=True, _bandit_model_hash="abc123")
    assert states._restore_plan(env, model_hash="abc123", warm=False) == "full_with_prereset"


def test_restore_plan_initialized_but_no_cached_hash_takes_full_path_without_prereset():
    """An initialized env that has never gone through the full/hash-caching
    branch (e.g. only ever restored via the fast path so far -- not reachable in
    practice since the fast path requires a prior full restore to seed the hash,
    but exercised directly here) has no `_bandit_model_hash` attribute at all;
    getattr(..., None) must not accidentally equal a real hash string."""
    env = types.SimpleNamespace(_bandit_initialized=True)
    assert states._restore_plan(env, model_hash="abc123", warm=True) == "full_no_prereset"


# --- _restore_warm's deterministic_reset flag must never leak (review fix) ---

class _FakeRawEnv:
    """Minimal stand-in for the underlying robosuite env `_restore_warm`
    touches: only `deterministic_reset` (the flag under test) and
    `set_attrs_from_ep_meta` are needed before `env.reset()` is called."""
    def __init__(self):
        self.deterministic_reset = False
        self.ep_meta_applied = None

    def set_attrs_from_ep_meta(self, ep_meta):
        self.ep_meta_applied = ep_meta


class _FakeEnvResetRaises:
    def __init__(self, raw):
        self.env = raw

    def reset(self, unset_ep_meta=False):
        raise RuntimeError("boom -- controller error mid-reset")


class _FakeEnvResetOk:
    def __init__(self, raw):
        self.env = raw
        self.reset_called_with = None

    def reset(self, unset_ep_meta=False):
        self.reset_called_with = unset_ep_meta

    def get_observation(self):
        return {"ok": True}


def test_restore_warm_resets_deterministic_reset_flag_even_when_env_reset_raises():
    """If `env.reset()` itself raises (bad ep_meta, a controller error,
    anything), `deterministic_reset` must still end up False -- a stuck
    `True` would silently force EVERY later reset() on this same env (warm
    or cold) onto the soft/no-recompile branch, corrupting every restore
    after the exception until the process restarts. This is the exact bug a
    bare (no try/finally) flag flip has."""
    raw = _FakeRawEnv()
    env = _FakeEnvResetRaises(raw)

    with pytest.raises(RuntimeError, match="boom"):
        states._restore_warm(env, states_arr=None, ep_meta_json="{}")

    assert raw.deterministic_reset is False


def test_restore_warm_sets_flag_true_during_reset_then_false_after_success():
    """Happy path: the flag is True for the duration of `env.reset()` (so
    robosuite's soft-reset branch actually fires) and False again once
    `_restore_warm` returns -- the try/finally must not change this normal
    case's outcome at all."""
    raw = _FakeRawEnv()
    seen_during_reset = {}

    class _FakeEnvCapturing(_FakeEnvResetOk):
        def reset(self, unset_ep_meta=False):
            seen_during_reset["flag"] = raw.deterministic_reset
            super().reset(unset_ep_meta=unset_ep_meta)

    env = _FakeEnvCapturing(raw)

    class _FakeSim:
        def set_state_from_flattened(self, arr):
            pass

        def forward(self):
            pass

    raw.sim = _FakeSim()

    obs = states._restore_warm(env, states_arr=np.zeros(3), ep_meta_json="{}")

    assert seen_during_reset["flag"] is True
    assert raw.deterministic_reset is False
    assert obs == {"ok": True}
