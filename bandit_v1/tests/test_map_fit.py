"""Tests for bandit_v1/map_fit.py (Task 8, step 1): p_hat_0 + p_stage map
models, synthetic-data-validated (real fit on the diagnosis ledger is a later,
deferred step -- the 2,400-rollout diag batch hasn't run yet).

Step 1 (a)-(c), per the brief:
  (a) test_predict_p_orders_tall_vs_short_and_holdout_auc_above_0_75
  (b) test_category_target_encoding_shrinks_singleton_category_to_global_mean
  (c) test_predict_stage_rows_sum_to_one
Plus (explicitly requested additions): GroupKFold no-straddle, zero-variance
feature safety, save/load roundtrip equality, category-canonicalization
guard, and the pool/eval commensurability property (design section 2 item 4
-- pool demos and eval conditions must embed identically since they share no
columns beyond [h, w, x_rel, y_rel, side, category]).
"""
import numpy as np
import pandas as pd
import pytest

from bandit_v1 import map_fit


def _make_synthetic_diag_df(n_conditions=250, repeats=4, seed=0):
    """Synthetic diag-ledger-shaped rows: success = 1{h<0.2} + Bernoulli noise
    (0.88/0.12 true rates, not 1/0 -- "+ noise" per the brief), repeats of the
    SAME condition (start_id) share the SAME condition-level features, only
    the stochastic outcome varies across repeat_idx -- exactly how real diag
    rows are generated (rollout.run: one start_dir, `repeats` rollouts).
    category cycles independently of h (no confound), so cat_te doesn't
    accidentally carry the h-based signal the AUC test is meant to isolate."""
    rng = np.random.default_rng(seed)
    cats = ["jar", "jug", "apple", "plate", "saucepan"]
    stage_choices = np.array([
        "never_reached", "reached_no_grasp",
        "fail_grasped_no_transport", "fail_reached_sink_no_place",
    ])
    rows = []
    for i in range(n_conditions):
        h = float(rng.uniform(0.02, 0.40))
        w = float(rng.uniform(0.02, 0.20))
        x_rel = float(rng.uniform(-0.8, 0.8))
        y_rel = float(rng.uniform(-0.8, 0.8))
        side = 1 if (x_rel if abs(x_rel) >= abs(y_rel) else y_rel) >= 0 else -1
        cat = cats[i % len(cats)]
        base_p = 0.88 if h < 0.2 else 0.12
        for r in range(repeats):
            success = bool(rng.random() < base_p)
            stage = "success" if success else str(rng.choice(stage_choices))
            rows.append(dict(
                start_id=f"start_{i:05d}", repeat_idx=r, success=success,
                failure_stage=stage, category=cat, h=h, w=w,
                x_rel=x_rel, y_rel=y_rel, side=side,
                yaw=0.0, layout_id=1, style_id=1,
            ))
    return pd.DataFrame(rows)


# --- (a) AUC + tall/short ordering ------------------------------------------

def test_predict_p_orders_tall_vs_short_and_holdout_auc_above_0_75():
    df = _make_synthetic_diag_df(n_conditions=250, repeats=4, seed=0)
    models = map_fit.fit(df)

    short_probe = {"category": "jar", "h": 0.05, "w": 0.10, "x_rel": 0.1, "y_rel": 0.1, "side": 1}
    tall_probe = {"category": "jar", "h": 0.35, "w": 0.10, "x_rel": 0.1, "y_rel": 0.1, "side": 1}
    p_short = models.predict_p(short_probe)[0]
    p_tall = models.predict_p(tall_probe)[0]
    assert p_short > p_tall, f"expected short (h=0.05) > tall (h=0.35), got {p_short} <= {p_tall}"

    report = map_fit.validation_report(models, df, n_splits=5)
    assert report["auc"] > 0.75, f"held-out AUC {report['auc']} did not clear 0.75"
    assert report["n"] == len(df)
    assert report["n_groups"] == df["start_id"].nunique()


# --- (b) cat_te shrinkage -----------------------------------------------------

def test_category_target_encoding_shrinks_singleton_category_to_global_mean():
    rng = np.random.default_rng(1)
    rows = [dict(category="common", success=bool(rng.random() < 0.5)) for _ in range(60)]
    rows.append(dict(category="rare", success=True))  # n=1, extreme outcome
    df = pd.DataFrame(rows)

    cat_te, global_mean = map_fit.category_target_encoding(df, k=8)

    expected_rare = (8 * global_mean + 1 * 1.0) / (8 + 1)
    assert cat_te["rare"] == pytest.approx(expected_rare)
    # n=1 -> heavily shrunk: closer to global_mean than to its own raw (extreme) rate
    assert abs(cat_te["rare"] - global_mean) < abs(cat_te["rare"] - 1.0)
    assert abs(cat_te["rare"] - global_mean) < 0.15


def test_build_feature_matrix_unseen_category_falls_back_to_global_mean():
    cat_encoding = {"a": 0.9}
    global_mean = 0.4
    df = pd.DataFrame([{"category": "never_seen_at_fit_time", "h": 0.1, "w": 0.1,
                         "x_rel": 0.0, "y_rel": 0.0, "side": 1}])
    X = map_fit._build_feature_matrix(df, cat_encoding, global_mean)
    assert X["cat_te"].iloc[0] == pytest.approx(global_mean)


# --- (c) predict_stage sums to 1 ---------------------------------------------

def test_predict_stage_rows_sum_to_one():
    df = _make_synthetic_diag_df(n_conditions=120, repeats=3, seed=7)
    models = map_fit.fit(df)
    sample = df.sample(20, random_state=0)

    stage_proba = models.predict_stage(sample)
    assert stage_proba.shape == (20, len(map_fit.STAGES))
    assert np.all(stage_proba >= 0)
    assert np.allclose(stage_proba.sum(axis=1), 1.0)


# --- GroupKFold no-straddle ----------------------------------------------------

def test_group_kfold_splits_never_straddle_a_start_id():
    df = _make_synthetic_diag_df(n_conditions=40, repeats=5, seed=2)
    splits, n_splits_eff = map_fit._group_kfold_splits(df, n_splits=5)
    assert n_splits_eff == 5
    start_ids = df["start_id"].to_numpy()

    for tr, va in splits:
        assert set(start_ids[tr]).isdisjoint(set(start_ids[va]))

    # folds partition every row exactly once
    all_va = sorted(np.concatenate([va for _, va in splits]).tolist())
    assert all_va == list(range(len(df)))


def test_group_kfold_splits_caps_n_splits_to_n_groups():
    # only 3 distinct start_ids -> n_splits can't exceed 3, even if 5 was asked for
    df = _make_synthetic_diag_df(n_conditions=3, repeats=6, seed=3)
    _, n_splits_eff = map_fit._group_kfold_splits(df, n_splits=5)
    assert n_splits_eff == 3


# --- zero-variance feature safety --------------------------------------------

def test_fit_handles_zero_variance_features_without_crashing():
    rng = np.random.default_rng(3)
    rows = []
    for i in range(60):
        success = bool(rng.random() < 0.5)
        rows.append(dict(
            start_id=f"start_{i:05d}", repeat_idx=0, success=success,
            failure_stage="success" if success else "reached_no_grasp",
            category="jar",       # constant -> zero-variance cat_te too
            h=0.15,                # constant -> zero-variance h and h_sq
            w=0.10, x_rel=float(rng.uniform(-0.5, 0.5)), y_rel=0.0,
            side=1,                # constant -> zero-variance side_enc
        ))
    df = pd.DataFrame(rows)

    models = map_fit.fit(df)  # must not raise / divide-by-zero

    p = models.predict_p(df)
    assert np.all(np.isfinite(p))
    assert np.all((p >= 0) & (p <= 1))

    stage = models.predict_stage(df)
    assert np.all(np.isfinite(stage))
    assert np.allclose(stage.sum(axis=1), 1.0)


def test_fit_handles_single_class_target_via_dummy_fallback():
    # every row succeeds -> y has 1 class; LogisticRegression alone would raise.
    rows = [dict(start_id=f"s{i}", repeat_idx=0, success=True, failure_stage="success",
                 category="jar", h=0.1, w=0.1, x_rel=0.0, y_rel=0.0, side=1)
            for i in range(10)]
    df = pd.DataFrame(rows)
    models = map_fit.fit(df)
    p = models.predict_p(df)
    assert np.allclose(p, 1.0)


# --- save/load roundtrip -------------------------------------------------------

def test_save_load_roundtrip_predictions_match(tmp_path):
    df = _make_synthetic_diag_df(n_conditions=80, repeats=3, seed=4)
    models = map_fit.fit(df)
    path = tmp_path / "map_models.joblib"

    returned_path = map_fit.save(models, path)
    assert returned_path == path
    assert path.exists()

    loaded = map_fit.load(path)
    probe = df.iloc[:10]

    assert np.array_equal(models.predict_p(probe), loaded.predict_p(probe))
    assert np.array_equal(models.predict_stage(probe), loaded.predict_stage(probe))
    assert loaded.cat_encoding == models.cat_encoding
    assert loaded.metadata == models.metadata


# --- category canonicalization guard -------------------------------------------

def test_fit_rejects_non_canonical_category():
    df = _make_synthetic_diag_df(n_conditions=10, repeats=2, seed=9)
    df.loc[df.index[0], "category"] = "jug_wide_opening"  # real alias, config.CATEGORY_ALIASES
    with pytest.raises(AssertionError):
        map_fit.fit(df)


def test_predict_features_rejects_non_canonical_category():
    df = _make_synthetic_diag_df(n_conditions=10, repeats=2, seed=10)
    models = map_fit.fit(df)
    bad_row = {"category": "saucepan_with_lid", "h": 0.1, "w": 0.1,
               "x_rel": 0.0, "y_rel": 0.0, "side": 1}
    with pytest.raises(AssertionError):
        models.predict_features(bad_row)


# --- pool/eval commensurability (design section 2 item 4) ---------------------

def test_predict_features_commensurable_across_pool_and_eval_row_shapes():
    """A pool_demos.parquet-shaped row (episode_index/layout/traj_len/in_d0,
    no yaw/style_id/layout_id) and an eval/diag-condition-shaped row
    (yaw/layout_id/style_id/start_id/repeat_idx, no episode_index/traj_len/
    in_d0) sharing the same [h, w, x_rel, y_rel, side, category] must embed
    to the exact same feature vector and the exact same predictions -- the
    property Tasks 9/10's retrieval-time membership scoring depends on."""
    df = _make_synthetic_diag_df(n_conditions=100, repeats=3, seed=11)
    models = map_fit.fit(df)

    shared = dict(category="jar", h=0.12, w=0.08, x_rel=0.3, y_rel=-0.2, side=-1)
    pool_row = {**shared, "episode_index": 42, "layout": 3, "traj_len": 120, "in_d0": False}
    eval_row = {**shared, "yaw": 1.57, "layout_id": 3, "style_id": 7,
                "start_id": "start_00042", "repeat_idx": 0}

    feats_pool = models.predict_features(pool_row)
    feats_eval = models.predict_features(eval_row)
    pd.testing.assert_frame_equal(feats_pool.reset_index(drop=True), feats_eval.reset_index(drop=True))

    assert models.predict_p(pool_row)[0] == models.predict_p(eval_row)[0]
    assert np.array_equal(models.predict_stage(pool_row), models.predict_stage(eval_row))
