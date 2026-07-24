"""Pure-part tests for bandit_v1/diagnosis.py (task 6, step 1).

Covers only assign_cell -- the grid-assignment logic that needs no live env, no
captured start directories, and no filesystem I/O. select_conditions (the
env-touching scan/capture/keep-discard loop) is validated end-to-end by the
--out_check dry-run + full 300-condition run (task 6, step 3), per the brief --
TDD applies only to this pure part, mirroring test_states.py's split for
capture_start/restore vs. fingerprint_diff/start_features.
"""
from collections import Counter

import pytest

from bandit_v1 import diagnosis


TERCILE_MAP = {"jar": 0, "jug": 1, "apple": 2}


def test_assign_cell_h_below_lower_boundary_is_bin0():
    feats = {"category": "jar", "h": 0.05, "x_rel": 0.1}
    tercile, hbin, xbin = diagnosis.assign_cell(feats, TERCILE_MAP)
    assert (tercile, hbin, xbin) == (0, 0, 0)


def test_assign_cell_h_exactly_lower_boundary_is_middle_bin():
    # 0.08 is the lower h cutoff; closed on the middle side (per docstring).
    feats = {"category": "jar", "h": 0.08, "x_rel": 0.1}
    _, hbin, _ = diagnosis.assign_cell(feats, TERCILE_MAP)
    assert hbin == 1


def test_assign_cell_h_exactly_upper_boundary_is_middle_bin():
    # 0.212 is the upper h cutoff; closed on the middle side (per docstring),
    # NOT the upper bin -- the brief's explicit boundary case.
    feats = {"category": "jar", "h": 0.212, "x_rel": 0.1}
    _, hbin, _ = diagnosis.assign_cell(feats, TERCILE_MAP)
    assert hbin == 1


def test_assign_cell_h_above_upper_boundary_is_bin2():
    feats = {"category": "jar", "h": 0.30, "x_rel": 0.1}
    _, hbin, _ = diagnosis.assign_cell(feats, TERCILE_MAP)
    assert hbin == 2


def test_assign_cell_xrel_exactly_upper_boundary_is_middle_bin():
    # |x_rel| == 0.65 is the brief's other explicit boundary case: middle bin.
    feats = {"category": "jar", "h": 0.05, "x_rel": 0.65}
    _, _, xbin = diagnosis.assign_cell(feats, TERCILE_MAP)
    assert xbin == 1


def test_assign_cell_xrel_exactly_lower_boundary_is_middle_bin():
    feats = {"category": "jar", "h": 0.05, "x_rel": 0.325}
    _, _, xbin = diagnosis.assign_cell(feats, TERCILE_MAP)
    assert xbin == 1


def test_assign_cell_xrel_bin_uses_absolute_value_negative_side():
    # x_rel is signed; the bin is on |x_rel|, so a negative value past -0.65
    # must land in bin 2, same as +0.70 would.
    feats = {"category": "jar", "h": 0.05, "x_rel": -0.70}
    _, _, xbin = diagnosis.assign_cell(feats, TERCILE_MAP)
    assert xbin == 2

    feats_neg_mid = {"category": "jar", "h": 0.05, "x_rel": -0.40}
    _, _, xbin_mid = diagnosis.assign_cell(feats_neg_mid, TERCILE_MAP)
    assert xbin_mid == 1


def test_assign_cell_tercile_from_map():
    feats_jar = {"category": "jar", "h": 0.05, "x_rel": 0.1}
    feats_apple = {"category": "apple", "h": 0.05, "x_rel": 0.1}
    assert diagnosis.assign_cell(feats_jar, TERCILE_MAP)[0] == 0
    assert diagnosis.assign_cell(feats_apple, TERCILE_MAP)[0] == 2


def test_assign_cell_unseen_category_defaults_to_middle_tercile():
    feats = {"category": "not_in_prior_table", "h": 0.05, "x_rel": 0.1}
    tercile, _, _ = diagnosis.assign_cell(feats, TERCILE_MAP)
    assert tercile == 1


def test_assign_cell_canonicalizes_alias_category_before_tercile_lookup():
    # tercile_map is keyed by canonical category names only ("jug", never
    # "jug_wide_opening"); assign_cell must canonicalize features["category"]
    # itself before the lookup, same convention as states.start_features.
    feats = {"category": "jug_wide_opening", "h": 0.05, "x_rel": 0.1}
    tercile, _, _ = diagnosis.assign_cell(feats, TERCILE_MAP)
    assert tercile == TERCILE_MAP["jug"]


def test_assign_cell_full_grid_position_typical_case():
    feats = {"category": "apple", "h": 0.15, "x_rel": -0.5}
    assert diagnosis.assign_cell(feats, TERCILE_MAP) == (2, 1, 1)


# --- tercile-map construction against the real repo prior table ------------
# (not env-touching -- pure file reads -- so still fits step 1's "no env" scope,
# unlike select_conditions itself.)

def test_load_prior_category_rates_merges_jug_alias():
    rates = diagnosis._load_prior_category_rates()
    # jug_wide_opening (n=56) must be folded into jug (n=151), n-weighted:
    # (0.299*151 + 0.251*56) / 207 == 0.28601449...
    assert "jug_wide_opening" not in rates
    assert rates["jug"] == pytest.approx((0.299 * 151 + 0.251 * 56) / 207, abs=1e-6)


def test_build_tercile_map_partitions_all_categories_into_3_roughly_equal_groups():
    tmap = diagnosis.build_tercile_map(write=False)
    counts = Counter(tmap.values())
    assert set(counts) == {0, 1, 2}
    # 80 canonical categories (81 raw cats.json entries minus the jug alias
    # merge) split as evenly as np.array_split allows.
    assert sum(counts.values()) == 80
    assert max(counts.values()) - min(counts.values()) <= 1


def test_build_tercile_map_orders_hardest_to_easiest():
    rates = diagnosis._load_prior_category_rates()
    tmap = diagnosis.build_tercile_map(write=False)
    # every tercile-0 category's rate must be <= every tercile-2 category's rate
    hard = [rates[c] for c, t in tmap.items() if t == 0]
    easy = [rates[c] for c, t in tmap.items() if t == 2]
    assert max(hard) <= min(easy)


def test_build_tercile_map_pot_is_absent_saucepan_alias_is_merged():
    # "pot" is reachable by the live env but genuinely missing from the prior
    # table (falls back to the middle tercile via assign_cell, not this map).
    # "saucepan_with_lid" is an alias, never a tercile_map key -- only its
    # canonical form "saucepan" is.
    tmap = diagnosis.build_tercile_map(write=False)
    assert "pot" not in tmap
    assert "saucepan_with_lid" not in tmap
    assert "saucepan" in tmap


# --- nearest_undersubscribed overflow cell selection logic -------------------

def test_nearest_undersubscribed_same_tercile_undersubscribed_cell_returned():
    """Given a full target cell and one undersubscribed cell in the same tercile,
    it returns that undersubscribed cell."""
    cell = (1, 1, 1)  # target cell (tercile 1, middle of h and x bins)
    counts = {
        (1, 1, 1): 12,  # full
        (1, 0, 0): 5,   # undersubscribed, same tercile
        (0, 1, 1): 12,  # full, different tercile
    }
    target = 12
    all_cells = [(0, 0, 0), (0, 1, 1), (1, 0, 0), (1, 1, 1), (2, 1, 1)]

    result = diagnosis._nearest_undersubscribed(cell, counts, target, all_cells)
    assert result == (1, 0, 0)


def test_nearest_undersubscribed_tercile_distance_dominates_bin_distance():
    """Tercile distance dominates: an undersubscribed cell in the same tercile
    is preferred over a closer-by-bins cell in a different tercile."""
    cell = (1, 1, 1)  # target cell in tercile 1
    counts = {
        (1, 1, 1): 12,    # full
        (1, 0, 0): 8,     # undersubscribed, same tercile, far by bins
        (2, 1, 2): 6,     # undersubscribed, different tercile, close by bins
    }
    target = 12
    all_cells = [(0, 0, 0), (1, 0, 0), (1, 1, 1), (2, 1, 2)]

    # (1, 0, 0): tercile distance = 0, bin distance = |0-1| + |0-1| = 2
    # (2, 1, 2): tercile distance = 1, bin distance = |1-1| + |2-1| = 1
    # tercile distance comes first in the ranking, so (1, 0, 0) wins
    result = diagnosis._nearest_undersubscribed(cell, counts, target, all_cells)
    assert result == (1, 0, 0)


def test_nearest_undersubscribed_deterministic_tiebreak_by_cell_tuple():
    """Deterministic tiebreak between two equally-distant undersubscribed cells:
    the cell with smaller (tercile, hbin, xbin) tuple wins."""
    cell = (1, 1, 1)  # target cell
    counts = {
        (1, 1, 1): 12,    # full
        (1, 0, 0): 5,     # undersubscribed, same tercile distance, bin distance = 2
        (1, 2, 2): 7,     # undersubscribed, same tercile distance, bin distance = 2
    }
    target = 12
    all_cells = [(1, 0, 0), (1, 1, 1), (1, 2, 2)]

    # Both (1, 0, 0) and (1, 2, 2) have:
    # - tercile distance = 0
    # - bin distance = |0-1| + |0-1| = 2 and |2-1| + |2-1| = 2
    # The tiebreaker is the cell tuple itself: (1, 0, 0) < (1, 2, 2)
    result = diagnosis._nearest_undersubscribed(cell, counts, target, all_cells)
    assert result == (1, 0, 0)


def test_nearest_undersubscribed_returns_none_when_all_cells_full():
    """Returns None when all cells are at or above the target."""
    cell = (1, 1, 1)
    counts = {
        (0, 0, 0): 12,
        (1, 0, 0): 12,
        (1, 1, 1): 12,
        (2, 2, 2): 15,  # over target
    }
    target = 12
    all_cells = [(0, 0, 0), (1, 0, 0), (1, 1, 1), (2, 2, 2)]

    result = diagnosis._nearest_undersubscribed(cell, counts, target, all_cells)
    assert result is None


def test_nearest_undersubscribed_complex_ranking():
    """Test a more complex scenario with mixed tercile and bin distances."""
    cell = (1, 1, 1)  # target cell
    counts = {
        (1, 1, 1): 12,    # full, target
        (0, 1, 1): 5,     # tercile distance = 1, bin distance = 0
        (2, 1, 1): 6,     # tercile distance = 1, bin distance = 0
        (1, 0, 0): 7,     # tercile distance = 0, bin distance = 2
        (1, 1, 0): 8,     # tercile distance = 0, bin distance = 1
    }
    target = 12
    all_cells = [(0, 1, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 1)]

    # (0, 1, 1): tercile=1, bin=0
    # (2, 1, 1): tercile=1, bin=0
    # (1, 0, 0): tercile=0, bin=2
    # (1, 1, 0): tercile=0, bin=1
    # Rankings: (1, 1, 0) wins with tercile=0, bin=1
    result = diagnosis._nearest_undersubscribed(cell, counts, target, all_cells)
    assert result == (1, 1, 0)


def test_nearest_undersubscribed_all_cells_missing_from_counts_defaults_to_zero():
    """When a cell is not in counts dict, its count defaults to 0 (undersubscribed)."""
    cell = (1, 1, 1)
    counts = {}  # all cells have implicit count 0
    target = 12
    all_cells = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]

    # (1, 1, 1) is implicit 0, which is < 12 (undersubscribed)
    result = diagnosis._nearest_undersubscribed(cell, counts, target, all_cells)
    # With all undersubscribed, ranking by distance and tiebreak:
    # (1, 1, 1): tercile=0, bin=0, cell=(1,1,1)
    # (0, 0, 0): tercile=1, bin=2, cell=(0,0,0)
    # (2, 2, 2): tercile=1, bin=2, cell=(2,2,2)
    # (1, 1, 1) is the winner as it's closest
    assert result == (1, 1, 1)
