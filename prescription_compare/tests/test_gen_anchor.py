import pytest

from prescription_compare.gen_anchor_arms import (
    distribute, build_category_plan, GROUP_TO_CATEGORIES,
)


def test_distribute_is_proportional_and_sums_to_count():
    out = distribute(100, ["a", "b"], {"a": 60, "b": 60})
    assert sum(out.values()) == 100
    assert out["a"] == out["b"] == 50


def test_distribute_caps_total_at_availability():
    out = distribute(300, ["a", "b"], {"a": 30, "b": 60})
    assert out == {"a": 30, "b": 60}   # target capped at 90 total, all taken


def test_distribute_respects_per_category_cap_and_still_sums():
    out = distribute(100, ["a", "b"], {"a": 10, "b": 200})
    assert out["a"] <= 10 and out["b"] <= 200
    assert sum(out.values()) == 100


def test_distribute_zero_or_empty():
    assert sum(distribute(0, ["a"], {"a": 5}).values()) == 0
    assert sum(distribute(50, ["a"], {"a": 0}).values()) == 0


def test_build_category_plan_totals_match_group_counts():
    # groups in region order; only unfixable(10) + hard(20) requested
    region_names = ["unfixable_hard", "hard_collectable", "retention_toxic", "easy_majority"]
    group_counts = [10, 20, 0, 0]
    avail = {c: 100 for cats in GROUP_TO_CATEGORIES.values() for c in cats}
    plan = build_category_plan(region_names, group_counts, avail)
    assert sum(plan.values()) == 30
    # only categories from the two requested groups appear
    used = set(plan)
    assert used <= set(GROUP_TO_CATEGORIES["unfixable_hard"]) | set(GROUP_TO_CATEGORIES["hard_collectable"])


def test_group_mapping_covers_the_four_groups():
    assert set(GROUP_TO_CATEGORIES) == {
        "unfixable_hard", "hard_collectable", "retention_toxic", "easy_majority"
    }
    # no category is shared across groups
    all_cats = [c for cats in GROUP_TO_CATEGORIES.values() for c in cats]
    assert len(all_cats) == len(set(all_cats))
