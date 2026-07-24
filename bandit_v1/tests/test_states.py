"""Pure-part tests for bandit_v1/states.py (task 3, step 2).

Covers only the parts that don't touch a live robocasa env: fingerprint_diff
(dict comparison) and start_features (ep_meta.json + fingerprint.json parsing).
The env-touching parts (capture_start, restore, make_env, fingerprint) are
validated end-to-end by the validate_reset.py gate run (task 3, step 4), per the
brief -- TDD applies only to these pure parts.
"""
import json

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
