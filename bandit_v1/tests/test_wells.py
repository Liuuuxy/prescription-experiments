"""Tests for bandit_v1/wells.py (Task 10, second half): nearest-centroid
region assignment (D0-excluded), the well-count table, and the B rule.
"""
import numpy as np
import pandas as pd
import pytest

from bandit_v1 import clustering, wells

POOL_COLS = ["episode_index", "category", "h", "w", "layout",
             "x_rel", "y_rel", "side", "traj_len", "in_d0"]


def _make_pool(rows: list) -> pd.DataFrame:
    defaults = {"h": 0.1, "w": 0.1, "layout": 0, "x_rel": 0.0, "y_rel": 0.0,
                "side": 1, "traj_len": 50, "in_d0": False}
    out = []
    for r in rows:
        row = dict(defaults)
        row.update(r)
        out.append(row)
    return pd.DataFrame(out)[POOL_COLS]


class _ConstantModels:
    """Fake MapModels double: predict_p/predict_stage return the SAME
    constant for every row regardless of features. Paired with a z_spec
    whose p_hat/p_stage means equal those same constants, the p_hat/p_stage
    Z-blocks come out exactly 0 for every row -- isolating
    `assign_regions`'s nearest-centroid decision to the knob block alone, so
    expected assignments are simple arithmetic."""

    def predict_p(self, df):
        return np.zeros(len(df))

    def predict_stage(self, df):
        return np.tile(np.array([0.2] * 5), (len(df), 1))


def _identity_z_spec() -> clustering.ZSpec:
    return clustering.ZSpec(
        knob_cols=clustering.KNOB_COLS,
        knob_mean=np.zeros(5), knob_scale=np.ones(5),
        p_hat_mean=0.0, p_hat_scale=1.0,
        p_stage_mean=np.array([0.2] * 5), p_stage_scale=1.0,
        stages=clustering.STAGES,
    )


def _z11(knob5) -> list:
    """11-dim Z-space centroid for a 5-dim raw knob vector, under
    `_identity_z_spec` + `_ConstantModels` (p_hat/p_stage blocks are always
    exactly 0, so they never affect nearest-centroid distance here)."""
    return [float(x) for x in knob5] + [0.0] * 6


# =============================================================================
# assign_regions
# =============================================================================

def test_assign_regions_excludes_d0_and_picks_nearest_centroid():
    pool_df = _make_pool([
        {"episode_index": 1, "category": "jar", "h": 2, "w": 2, "x_rel": 2, "y_rel": 2, "side": 1, "in_d0": True},
        {"episode_index": 2, "category": "jar", "h": 2.1, "w": 2, "x_rel": 2, "y_rel": 2, "side": 1, "in_d0": False},
        {"episode_index": 3, "category": "jar", "h": -2, "w": -2, "x_rel": -2, "y_rel": -2, "side": -1, "in_d0": False},
        {"episode_index": 4, "category": "jar", "h": -2.1, "w": -2, "x_rel": -2, "y_rel": -2, "side": -1, "in_d0": True},
    ])
    arms_spec = {
        "z_spec": _identity_z_spec().to_dict(),
        "arms": [
            {"name": "armA", "index": 0, "centroid": {"standardized": _z11([2, 2, 2, 2, 1])}},
            {"name": "armB", "index": 1, "centroid": {"standardized": _z11([-2, -2, -2, -2, -1])}},
        ],
    }

    regions = wells.assign_regions(pool_df, _ConstantModels(), arms_spec)

    # D0 rows (episode_index 1 and 4) must never appear, at all.
    assert set(regions.index) == {2, 3}
    assert regions.loc[2] == "armA"
    assert regions.loc[3] == "armB"
    assert regions.index.name == "episode_index"


def test_assign_regions_raises_on_empty_arms():
    pool_df = _make_pool([{"episode_index": 1, "category": "jar", "in_d0": False}])
    arms_spec = {"z_spec": _identity_z_spec().to_dict(), "arms": []}
    with pytest.raises(ValueError):
        wells.assign_regions(pool_df, _ConstantModels(), arms_spec)


def _identity_behavior_z_spec() -> clustering.ZSpec:
    """Same identity-scaling idea as `_identity_z_spec`, but for
    descriptor="behavior" (no knob block at all -- p_hat(1) + p_stage(5),
    6-dim total)."""
    return clustering.ZSpec(
        descriptor="behavior",
        knob_cols=(), knob_mean=np.zeros(0), knob_scale=np.zeros(0),
        p_hat_mean=0.0, p_hat_scale=1.0,
        p_stage_mean=np.array([0.2] * 5), p_stage_scale=1.0,
        stages=clustering.STAGES,
    )


class _PHatFromH:
    """predict_p reads the pool row's `h` column directly as a stand-in
    probability -- descriptor="behavior" has no knob block at all, so this is
    the only lever available to make different pool rows land in different
    arms. predict_stage is a constant uniform [0.2]*5 vector, paired with
    `_identity_behavior_z_spec`'s p_stage_mean=[0.2]*5/p_stage_scale=1.0 so
    the p_stage block is always exactly 0 and never affects the
    nearest-centroid decision -- isolating it to the p_hat block alone."""

    def predict_p(self, df):
        return df["h"].to_numpy(dtype=float)

    def predict_stage(self, df):
        return np.tile(np.array([0.2] * 5), (len(df), 1))


def test_assign_regions_behavior_descriptor_uses_p_hat_only_no_knob_block():
    pool_df = _make_pool([
        {"episode_index": 1, "category": "jar", "h": 0.90, "in_d0": True},
        {"episode_index": 2, "category": "jar", "h": 0.85, "in_d0": False},
        {"episode_index": 3, "category": "jar", "h": 0.10, "in_d0": False},
        {"episode_index": 4, "category": "jar", "h": 0.05, "in_d0": True},
    ])
    arms_spec = {
        "z_spec": _identity_behavior_z_spec().to_dict(),
        "arms": [
            {"name": "armHigh", "index": 0,
             "centroid": {"standardized": [0.9, 0.0, 0.0, 0.0, 0.0, 0.0]}},
            {"name": "armLow", "index": 1,
             "centroid": {"standardized": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]}},
        ],
    }

    regions = wells.assign_regions(pool_df, _PHatFromH(), arms_spec)

    # D0 rows (episode_index 1 and 4) must never appear, at all.
    assert set(regions.index) == {2, 3}
    assert regions.loc[2] == "armHigh"
    assert regions.loc[3] == "armLow"
    assert regions.index.name == "episode_index"


def test_assign_regions_all_d0_yields_empty_series():
    pool_df = _make_pool([
        {"episode_index": 1, "category": "jar", "h": 1, "in_d0": True},
        {"episode_index": 2, "category": "jar", "h": 1, "in_d0": True},
    ])
    arms_spec = {
        "z_spec": _identity_z_spec().to_dict(),
        "arms": [{"name": "armA", "index": 0, "centroid": {"standardized": _z11([0, 0, 0, 0, 0])}}],
    }
    regions = wells.assign_regions(pool_df, _ConstantModels(), arms_spec)
    assert len(regions) == 0


# =============================================================================
# well_table
# =============================================================================

def test_well_table_includes_random_row_as_total():
    regions = pd.Series(["armA", "armA", "armB", "armA"], index=[10, 11, 12, 13])
    regions.index.name = "episode_index"

    table = wells.well_table(regions)

    by_arm = dict(zip(table["arm"], table["count"]))
    assert by_arm["armA"] == 3
    assert by_arm["armB"] == 1
    assert by_arm["random"] == 4
    assert list(table["arm"]) == sorted(table["arm"])


def test_well_table_empty_regions_still_has_random_row_zero():
    regions = pd.Series([], index=pd.Index([], name="episode_index"), dtype=object)
    table = wells.well_table(regions)
    assert list(table["arm"]) == ["random"]
    assert int(table.loc[table["arm"] == "random", "count"].iloc[0]) == 0


# =============================================================================
# choose_B (the B rule, brief's exact cases)
# =============================================================================

def _wt(counts: dict) -> pd.DataFrame:
    return pd.DataFrame([{"arm": arm, "count": c} for arm, c in counts.items()])


def test_choose_B_case_650_400_picks_100_not_200():
    table = _wt({"a": 650, "b": 400, "random": 1050})
    B, limiting = wells.choose_B(table)
    assert (B, limiting) == (100, "b")


def test_choose_B_case_900_700_picks_200():
    table = _wt({"a": 900, "b": 700, "random": 1600})
    B, limiting = wells.choose_B(table)
    assert (B, limiting) == (200, "b")


def test_choose_B_case_single_arm_70_picks_20():
    table = _wt({"a": 70, "random": 70})
    B, limiting = wells.choose_B(table)
    assert (B, limiting) == (20, "a")


def test_choose_B_random_arm_excluded_from_the_min():
    # Random's count (all of W) is huge, but must NEVER be the limiting arm.
    table = _wt({"a": 61, "random": 100000})
    B, limiting = wells.choose_B(table)
    assert (B, limiting) == (20, "a")


def test_choose_B_raises_when_even_smallest_candidate_fails():
    table = _wt({"a": 50, "random": 50})  # 50 < 3*20=60
    with pytest.raises(ValueError):
        wells.choose_B(table)


def test_choose_B_raises_on_no_cluster_arms():
    table = _wt({"random": 1000})
    with pytest.raises(ValueError):
        wells.choose_B(table)
