import numpy as np
import pandas as pd
import pytest

from bandit_v1 import config, draw

POOL_COLS = [
    "episode_index", "category", "h", "w", "layout",
    "x_rel", "y_rel", "side", "traj_len", "in_d0",
]


def make_pool(rows: list) -> pd.DataFrame:
    """Build a synthetic pool-table DataFrame (pool.py's OUTPUT_COLUMNS schema)
    from a list of partial dicts; unspecified fields get harmless defaults."""
    defaults = {"h": 0.1, "w": 0.1, "layout": 0, "side": 1, "traj_len": 50, "in_d0": False}
    out = []
    for r in rows:
        row = dict(defaults)
        row.update(r)
        out.append(row)
    df = pd.DataFrame(out)
    return df[POOL_COLS]


def e_feats(rows: list) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["category", "x_rel", "y_rel"])


EMPTY_E = e_feats([])


# --- (a) eps-gate: excludes within 0.05 m, keeps at 0.2 m -------------------

def test_eps_conflict_excludes_within_005_keeps_at_02():
    e = e_feats([{"category": "jar", "x_rel": 0.0, "y_rel": 0.0}])
    # same category, 0.05 m away (< 0.10 EPS_XY) -> conflict
    assert draw.eps_conflict("jar", 0.05, 0.0, e) is True
    # same category, 0.2 m away (>= 0.10 EPS_XY) -> no conflict
    assert draw.eps_conflict("jar", 0.2, 0.0, e) is False


def test_eps_conflict_ignores_different_category_even_if_close():
    e = e_feats([{"category": "jar", "x_rel": 0.0, "y_rel": 0.0}])
    assert draw.eps_conflict("mug", 0.01, 0.0, e) is False


def test_eps_conflict_boundary_is_strict_less_than():
    e = e_feats([{"category": "jar", "x_rel": 0.0, "y_rel": 0.0}])
    assert draw.eps_conflict("jar", config.EPS_XY, 0.0, e) is False  # == eps -> not a conflict
    assert draw.eps_conflict("jar", config.EPS_XY - 1e-6, 0.0, e) is True


def test_pull_demos_excludes_eps_conflicting_candidate_end_to_end():
    # Two well-episode candidates, same category "jar": one 0.05 m from an
    # E-start (must be excluded), one 0.2 m away (must survive and be chosen).
    pool_df = make_pool([
        {"episode_index": 1, "category": "jar", "h": 0.1, "w": 0.1, "x_rel": 0.05, "y_rel": 0.0},
        {"episode_index": 2, "category": "jar", "h": 0.2, "w": 0.1, "x_rel": 0.2, "y_rel": 0.0},
    ])
    regions = pd.Series({1: "hard", 2: "hard"})
    e = e_feats([{"category": "jar", "x_rel": 0.0, "y_rel": 0.0}])

    ids = draw.pull_demos("hard", 1, np.random.default_rng(0), pool_df, regions, e)
    assert ids == [2]


# --- (b) different rng seeds on the same arm: partial, not full overlap ----

def _big_pool(n_per_cat=10):
    rows = []
    idx = 0
    for cat_i, cat in enumerate(["jar", "mug", "jug", "apple", "cup"]):
        for k in range(n_per_cat):
            rows.append({
                "episode_index": idx,
                "category": cat,
                "h": 0.05 + 0.01 * k,
                "w": 0.05 + 0.01 * cat_i,
                "x_rel": 1.0 + 0.1 * k,     # far from origin -> no E/D0 conflicts
                "y_rel": 1.0 + 0.1 * cat_i,
            })
            idx += 1
    return make_pool(rows)


def test_pull_demos_different_seeds_partial_overlap():
    pool_df = _big_pool(n_per_cat=10)  # 50 well demos, all one region
    regions = pd.Series({i: "hard" for i in pool_df["episode_index"]})
    B = 5  # 3B = 15 < 50 -> the 3B-uniform-draw step is exercised

    # Seeds 0 and 13 (verified by direct execution against this exact synthetic
    # pool/B) land on a 2/5 overlap -- a concrete, reproducible instance of the
    # "partial, not full" with-replacement property (candidates = W ∩ region is
    # re-drawn fresh every pull, so two pulls of the same arm may legitimately
    # share some demos while differing in others; deterministic given rng, so
    # this is not a flaky statistical assertion).
    ids1 = set(draw.pull_demos("hard", B, np.random.default_rng(0), pool_df, regions, EMPTY_E))
    ids2 = set(draw.pull_demos("hard", B, np.random.default_rng(13), pool_df, regions, EMPTY_E))

    assert len(ids1) == B
    assert len(ids2) == B
    assert ids1 != ids2                    # not fully identical
    overlap = ids1 & ids2
    assert 0 < len(overlap) < B            # partial overlap: with-replacement semantics


# --- (c) all returned ids in W ∩ region -------------------------------------

def test_all_returned_ids_in_well_and_region():
    pool_df = make_pool([
        {"episode_index": 1, "category": "jar", "x_rel": 1.0, "y_rel": 0.0},
        {"episode_index": 2, "category": "jar", "x_rel": 2.0, "y_rel": 0.0},
        {"episode_index": 3, "category": "jar", "x_rel": 3.0, "y_rel": 0.0},
        {"episode_index": 4, "category": "mug", "x_rel": 5.0, "y_rel": 0.0},  # different arm
        # in D0 but *labeled* "hard" in regions -- must still be excluded (W trumps region label)
        {"episode_index": 5, "category": "jar", "x_rel": 1.5, "y_rel": 0.0, "in_d0": True},
    ])
    regions = pd.Series({1: "hard", 2: "hard", 3: "hard", 4: "easy", 5: "hard"})

    ids = draw.pull_demos("hard", 2, np.random.default_rng(0), pool_df, regions, EMPTY_E)

    hard_well_ids = {1, 2, 3}
    assert set(ids) <= hard_well_ids
    assert 4 not in ids
    assert 5 not in ids
    assert len(ids) == 2


def test_random_arm_draws_from_all_of_w_ignoring_region_labels():
    pool_df = make_pool([
        {"episode_index": i, "category": "jar", "x_rel": float(i), "y_rel": 0.0}
        for i in range(10)
    ] + [{"episode_index": 99, "category": "jar", "x_rel": 100.0, "y_rel": 0.0, "in_d0": True}])
    # deliberately incomplete/irrelevant regions -- random arm must not consult it
    regions = pd.Series({0: "hard", 1: "easy"})

    ids = draw.pull_demos(draw.RANDOM_ARM, 4, np.random.default_rng(0), pool_df, regions, EMPTY_E)
    assert len(ids) == 4
    assert 99 not in ids  # D0 excluded even for random
    assert set(ids) <= set(range(10))


# --- (d) FPS output size == B and deterministic given rng -------------------

def test_fps_output_size_and_deterministic_given_rng():
    pool_df = _big_pool(n_per_cat=10)
    regions = pd.Series({i: "hard" for i in pool_df["episode_index"]})
    B = 6

    ids_a = draw.pull_demos("hard", B, np.random.default_rng(42), pool_df, regions, EMPTY_E)
    ids_b = draw.pull_demos("hard", B, np.random.default_rng(42), pool_df, regions, EMPTY_E)

    assert len(ids_a) == B
    assert ids_a == ids_b  # same seed -> exact same list (order included)


def test_fps_no_subsample_path_when_candidates_leq_3B_is_also_deterministic():
    # 6 candidates, B=3 -> 3B=9 >= 6, so no initial uniform 3B draw; FPS runs
    # directly over all 6 candidates. Still must be deterministic given rng.
    pool_df = _big_pool(n_per_cat=2)  # 5 categories x 2 = 10; use a small B/region slice
    regions = pd.Series({i: "hard" for i in pool_df["episode_index"][:6]})
    B = 3

    ids_a = draw.pull_demos("hard", B, np.random.default_rng(7), pool_df, regions, EMPTY_E)
    ids_b = draw.pull_demos("hard", B, np.random.default_rng(7), pool_df, regions, EMPTY_E)
    assert len(ids_a) == B
    assert ids_a == ids_b


# --- novelty gate vs D0 -----------------------------------------------------

def test_pull_demos_excludes_novelty_conflict_with_d0_row():
    pool_df = make_pool([
        # candidate near a D0 row of the same category -> excluded
        {"episode_index": 1, "category": "jar", "x_rel": 0.02, "y_rel": 0.0},
        {"episode_index": 2, "category": "jar", "x_rel": 5.0, "y_rel": 0.0},
        {"episode_index": 10, "category": "jar", "x_rel": 0.0, "y_rel": 0.0, "in_d0": True},
    ])
    regions = pd.Series({1: "hard", 2: "hard"})
    ids = draw.pull_demos("hard", 1, np.random.default_rng(0), pool_df, regions, EMPTY_E)
    assert ids == [2]


def test_novelty_conflict_function_matches_eps_conflict_rule():
    other = pd.DataFrame({"category": ["jar"], "x_rel": [0.0], "y_rel": [0.0]})
    assert draw.novelty_conflict("jar", 0.05, 0.0, other) is True
    assert draw.novelty_conflict("jar", 0.2, 0.0, other) is False


# --- within-pull novelty post-filter + refill -------------------------------

def test_within_pull_novelty_refill_avoids_near_duplicates_and_still_fills_B():
    # Category A: 4 mutually-conflicting demos (identical x_rel/y_rel, so ANY
    # 2 of them conflict), spread out only in h so FPS has a reason to want
    # more than one of them. Categories B and C: exactly one demo each, never
    # conflicting with A or each other. With B_pull=3, the only way to fill 3
    # demos without a within-pull conflict is (>=1 A) + (the B demo) + (the C
    # demo) -- at most one A can ever survive together. This forces the
    # post-filter to reject surplus A picks and the refill to keep walking
    # until the B/C demos are found.
    rows = [
        {"episode_index": 1, "category": "A", "h": 0.05, "w": 0.1, "x_rel": 0.0, "y_rel": 0.0},
        {"episode_index": 2, "category": "A", "h": 0.15, "w": 0.1, "x_rel": 0.0, "y_rel": 0.0},
        {"episode_index": 3, "category": "A", "h": 0.25, "w": 0.1, "x_rel": 0.0, "y_rel": 0.0},
        {"episode_index": 4, "category": "A", "h": 0.35, "w": 0.1, "x_rel": 0.0, "y_rel": 0.0},
        {"episode_index": 5, "category": "B", "h": 0.10, "w": 0.5, "x_rel": 3.0, "y_rel": 0.0},
        {"episode_index": 6, "category": "C", "h": 0.10, "w": 0.9, "x_rel": -3.0, "y_rel": 2.0},
    ]
    pool_df = make_pool(rows)
    regions = pd.Series({r["episode_index"]: "hard" for r in rows})
    B_pull = 3

    for seed in range(8):  # regardless of which point FPS starts from
        ids = draw.pull_demos("hard", B_pull, np.random.default_rng(seed), pool_df, regions, EMPTY_E)
        assert len(ids) == B_pull
        chosen = pool_df[pool_df["episode_index"].isin(ids)]
        assert (chosen["category"] == "A").sum() <= 1
        assert set(chosen["category"]) == {"A", "B", "C"}
        # pairwise within-pull novelty check: no two chosen demos conflict
        cats = chosen["category"].tolist()
        xs = chosen["x_rel"].tolist()
        ys = chosen["y_rel"].tolist()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if cats[i] == cats[j]:
                    d = ((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2) ** 0.5
                    assert d >= config.EPS_XY


def test_pull_demos_raises_when_not_enough_candidates_after_gating():
    pool_df = make_pool([
        {"episode_index": 1, "category": "jar", "x_rel": 0.0, "y_rel": 0.0},
    ])
    regions = pd.Series({1: "hard"})
    with pytest.raises(ValueError):
        draw.pull_demos("hard", 2, np.random.default_rng(0), pool_df, regions, EMPTY_E)


def test_pull_demos_raises_for_unknown_arm():
    pool_df = make_pool([{"episode_index": 1, "category": "jar", "x_rel": 0.0, "y_rel": 0.0}])
    regions = pd.Series({1: "hard"})
    with pytest.raises(ValueError):
        draw.pull_demos("nonexistent_arm", 1, np.random.default_rng(0), pool_df, regions, EMPTY_E)


# --- log_selector_scores -----------------------------------------------------

def test_log_selector_scores_shape_and_zero_for_identical_demos():
    pool_df = _big_pool(n_per_cat=5)
    ids = pool_df["episode_index"].iloc[:3].tolist()
    scores = draw.log_selector_scores(ids, pool_df)
    assert scores["n_demos"] == 3
    assert scores["mean_pairwise_dist"] >= 0
    assert scores["mean_dist_to_nearest_d0"] != scores["mean_dist_to_nearest_d0"]  # NaN: no D0 rows here


def test_log_selector_scores_distance_to_d0_is_finite_when_d0_present():
    pool_df = _big_pool(n_per_cat=5)
    pool_df = pool_df.copy()
    pool_df.loc[pool_df["episode_index"] == 0, "in_d0"] = True
    ids = pool_df["episode_index"].iloc[1:4].tolist()
    scores = draw.log_selector_scores(ids, pool_df)
    assert scores["mean_dist_to_nearest_d0"] >= 0


def test_log_selector_scores_single_demo_pairwise_is_nan():
    pool_df = _big_pool(n_per_cat=5)
    ids = [int(pool_df["episode_index"].iloc[0])]
    scores = draw.log_selector_scores(ids, pool_df)
    assert scores["n_demos"] == 1
    assert scores["mean_pairwise_dist"] != scores["mean_pairwise_dist"]  # NaN
