"""Unit tests for analysis.py using synthetic episode records.

These validate the quantitative logic (success/failure bucketing, per-category
and per-region breakdowns, weakest-bucket detection) WITHOUT running the
simulator -- so the success path is exercised even though a random policy on a
hard task essentially never succeeds.

Run:  python -m pytest test_analysis.py -q   (or)   python test_analysis.py
"""

import analysis


def _make_records():
    # 6 episodes. peaches all fail; apples all succeed. left region fails,
    # right region succeeds -> known ground truth for the breakdowns.
    return [
        {"success": False, "object_category": "peach", "obj_xy_rel": [-0.3, 0.1], "layout_id": 1, "style_id": 1},
        {"success": False, "object_category": "peach", "obj_xy_rel": [-0.25, 0.1], "layout_id": 1, "style_id": 1},
        {"success": False, "object_category": "peach", "obj_xy_rel": [-0.2, 0.1], "layout_id": 1, "style_id": 1},
        {"success": True,  "object_category": "apple", "obj_xy_rel": [0.2, 0.1], "layout_id": 2, "style_id": 3},
        {"success": True,  "object_category": "apple", "obj_xy_rel": [0.25, 0.1], "layout_id": 2, "style_id": 3},
        {"success": True,  "object_category": "apple", "obj_xy_rel": [0.3, 0.1], "layout_id": 2, "style_id": 3},
    ]


def test_overall_rate():
    s = analysis.summarize(_make_records())
    assert s["overall"]["n"] == 6
    assert s["overall"]["successes"] == 3
    assert abs(s["overall"]["success_rate"] - 0.5) < 1e-9


def test_by_object_category():
    s = analysis.summarize(_make_records())
    assert s["by_object_category"]["peach"]["success_rate"] == 0.0
    assert s["by_object_category"]["apple"]["success_rate"] == 1.0
    # weakest-first ordering: peach should come before apple
    keys = list(s["by_object_category"].keys())
    assert keys.index("peach") < keys.index("apple")


def test_region_separation():
    s = analysis.summarize(_make_records())
    # the failing peaches (left) and succeeding apples (right) must land in
    # different region buckets, and the failing region must be 0%.
    rates = {k: v["success_rate"] for k, v in s["by_region"].items()}
    assert 0.0 in rates.values()
    assert 1.0 in rates.values()


def test_weakest_buckets():
    s = analysis.summarize(_make_records())
    weakest = s["weakest"]
    assert weakest, "expected at least one weak bucket"
    # the single weakest bucket should be a 0% one
    assert weakest[0]["success_rate"] == 0.0


def test_wilson_ci_bounds():
    lo, hi = analysis.wilson_ci(0, 10)
    assert lo == 0.0 and 0.0 < hi < 0.4
    lo, hi = analysis.wilson_ci(10, 10)
    assert hi == 1.0 and 0.6 < lo < 1.0
    assert analysis.wilson_ci(0, 0) == (0.0, 0.0)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")
