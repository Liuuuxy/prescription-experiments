"""Tests for bandit_v1/clustering.py (Task 10): z-block standardization,
choose_k/merge_small, summarize/build_arms_entries, and the draft/finalize
naming-hard-stop CLI. Synthetic data only -- the real clustering run (against
the real diag batch + fitted map_models.joblib) is deferred, see
task-10-report.md.
"""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from bandit_v1 import clustering, config, ledger, map_fit


# =============================================================================
# shared synthetic-data helpers
# =============================================================================

def _make_synthetic_diag_df(n_conditions=120, repeats=1, seed=0):
    """Synthetic diag-ledger-shaped rows (same recipe as test_map_fit.py's
    helper, independently defined here so this test file has no cross-file
    dependency): category cycles independently of h/x_rel so cat_te doesn't
    confound the knob-based structure these tests probe. One row per
    (start_id, repeat_idx); repeats of the same start_id share condition-level
    features (only the stochastic outcome varies)."""
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
        base_p = 0.85 if h < 0.2 else 0.15
        for r in range(repeats):
            success = bool(rng.random() < base_p)
            stage = "success" if success else str(rng.choice(stage_choices))
            rows.append(dict(
                start_id=f"start_{i:05d}", repeat_idx=r, success=success,
                failure_stage=stage, category=cat, h=h, w=w,
                x_rel=x_rel, y_rel=y_rel, side=side,
            ))
    return pd.DataFrame(rows)


def _fit_models(df):
    return map_fit.fit(df)


def _diag_conditions(df):
    return df.drop_duplicates(subset="start_id").reset_index(drop=True)


# =============================================================================
# build_z / ZSpec
# =============================================================================

def test_build_z_shape_and_block_standardization():
    df = _make_synthetic_diag_df(n_conditions=150, seed=1)
    models = _fit_models(df)
    diag = _diag_conditions(df)

    Z, z_spec = clustering.build_z(diag, models)

    assert Z.shape == (len(diag), 11)

    knob_block = Z[:, :5]
    assert np.allclose(knob_block.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(knob_block.std(axis=0), 1.0, atol=1e-6)

    p_hat_block = Z[:, 5:6]
    assert np.isclose(p_hat_block.mean(), 0.0, atol=1e-8)
    # block TOTAL variance (== its own variance, since it is 1-dim) is 1.
    assert np.isclose((p_hat_block ** 2).mean(), 1.0, atol=1e-6)

    p_stage_block = Z[:, 6:11]
    assert np.allclose(p_stage_block.mean(axis=0), 0.0, atol=1e-8)
    # block TOTAL variance (summed across the 5 columns) is 1 -- NOT each
    # column standardized to unit variance individually.
    total_var = (p_stage_block ** 2).mean(axis=0).sum()
    assert np.isclose(total_var, 1.0, atol=1e-6)


def test_build_z_applies_frozen_spec_to_new_rows_without_refitting():
    df = _make_synthetic_diag_df(n_conditions=150, seed=2)
    models = _fit_models(df)
    diag = _diag_conditions(df)

    _, z_spec = clustering.build_z(diag, models)

    # A "pool-shaped" frame with a wildly different h/w than anything in diag.
    pool_like = pd.DataFrame({
        "category": ["jar", "jug", "apple", "plate"] * 10,
        "h": np.full(40, 5.0),
        "w": np.full(40, 5.0),
        "x_rel": np.linspace(-1, 1, 40),
        "y_rel": np.linspace(-1, 1, 40),
        "side": [1, -1] * 20,
    })

    Z_pool, z_spec_2 = clustering.build_z(pool_like, models, z_spec=z_spec)

    # the frozen spec is returned unchanged (identity), never refit
    assert z_spec_2 is z_spec

    # manual re-derivation using the FROZEN diag mean/scale must match exactly
    expected_h_z = (5.0 - z_spec.knob_mean[0]) / z_spec.knob_scale[0]
    assert np.allclose(Z_pool[:, 0], expected_h_z)
    # sanity: not the degenerate all-zero column a FRESH fit on this
    # constant-h pool_like frame would have produced
    assert not np.allclose(Z_pool[:, 0], 0.0)


def test_zspec_roundtrip_through_dict():
    df = _make_synthetic_diag_df(n_conditions=80, seed=3)
    models = _fit_models(df)
    diag = _diag_conditions(df)
    _, z_spec = clustering.build_z(diag, models)

    restored = clustering.ZSpec.from_dict(z_spec.to_dict())
    Z1 = clustering.transform_z(diag, models, z_spec)
    Z2 = clustering.transform_z(diag, models, restored)
    assert np.allclose(Z1, Z2)


# =============================================================================
# descriptor="behavior" mode
# =============================================================================

class _ToyModels:
    """predict_p/predict_stage doubles that return FIXED arrays passed in at
    construction (row order = input df's row order), independent of any
    feature column -- lets a test hand-compute the exact expected z-block
    values with plain numpy instead of trusting a real logistic-regression
    fit. `metadata` mimics just enough of `map_fit.MapModels` for
    `fit_z_spec`'s `models.metadata.get("stages", STAGES)` call."""

    def __init__(self, p_hat, p_stage):
        self._p_hat = np.asarray(p_hat, dtype=float)
        self._p_stage = np.asarray(p_stage, dtype=float)
        self.metadata = {"stages": list(clustering.STAGES)}

    def predict_p(self, df):
        return self._p_hat[: len(df)]

    def predict_stage(self, df):
        return self._p_stage[: len(df)]


def test_behavior_descriptor_build_z_matches_hand_computed_standardization():
    p_hat = np.array([0.1, 0.5, 0.9, 0.3, 0.7, 0.2])
    p_stage = np.array([
        [0.60, 0.10, 0.10, 0.10, 0.10],
        [0.20, 0.30, 0.20, 0.20, 0.10],
        [0.05, 0.05, 0.05, 0.05, 0.80],
        [0.40, 0.20, 0.20, 0.10, 0.10],
        [0.10, 0.60, 0.10, 0.10, 0.10],
        [0.30, 0.30, 0.20, 0.10, 0.10],
    ])
    models = _ToyModels(p_hat, p_stage)
    # h/w/x_rel/y_rel/side/category are irrelevant to the behavior descriptor
    # (no knob block at all) -- present only because features_df conceivably
    # carries them; fit_z_spec/transform_z never read them in this mode.
    df = pd.DataFrame({
        "category": ["jar"] * 6, "h": [0.1] * 6, "w": [0.1] * 6,
        "x_rel": [0.0] * 6, "y_rel": [0.0] * 6, "side": [1] * 6,
    })

    Z, z_spec = clustering.build_z(df, models, descriptor="behavior")

    assert z_spec.descriptor == "behavior"
    assert z_spec.knob_cols == ()
    assert Z.shape == (6, 6)  # p_hat(1) + p_stage(5), no knob block

    # Hand-computed expected values: descriptor="behavior" standardizes ALL 6
    # columns (p_hat + each of the 5 p_stage columns) PER-COLUMN -- an
    # ordinary mean-0/std-1 StandardScaler on each column independently, NOT
    # cluster_study.py's Z3_behavior "shared_total" scheme (see clustering.py
    # module docstring's "descriptor modes" section for why this deliberate
    # deviation was verified empirically, via ARI against the owner-approved
    # cross-check fixture, rather than assumed).
    p_hat_mean = p_hat.mean()
    p_hat_scale = p_hat.std()  # per-column == "shared_total" for a 1-dim block
    expected_p_hat_z = (p_hat - p_hat_mean) / p_hat_scale

    p_stage_mean = p_stage.mean(axis=0)
    p_stage_scale = p_stage.std(axis=0)  # PER-COLUMN, independent per stage
    expected_p_stage_z = (p_stage - p_stage_mean) / p_stage_scale

    assert np.allclose(Z[:, 0], expected_p_hat_z)
    assert np.allclose(Z[:, 1:], expected_p_stage_z)

    # This is NOT the same p_stage scaling "hybrid" mode uses (shared_total:
    # all 5 columns divided by ONE scalar): "behavior"'s per-column scales
    # are genuinely unequal to each other, while "hybrid"'s (necessarily
    # single-valued) shared scalar is the same value repeated 5x -- and the
    # two descriptors' scale values differ from each other.
    behavior_scale = np.asarray(z_spec.p_stage_scale, dtype=float)
    assert not np.allclose(behavior_scale, behavior_scale[0])  # genuinely per-column

    _, z_spec_hybrid = clustering.build_z(df, models, descriptor="hybrid")
    assert z_spec_hybrid.descriptor == "hybrid"
    hybrid_scale = np.broadcast_to(np.asarray(z_spec_hybrid.p_stage_scale, dtype=float), (5,))
    assert np.allclose(hybrid_scale, hybrid_scale[0])  # one shared scalar, repeated
    assert not np.allclose(behavior_scale, hybrid_scale)


def test_behavior_descriptor_transform_z_applies_frozen_spec_without_refitting():
    p_hat = np.array([0.2, 0.4, 0.6, 0.8])
    p_stage = np.tile(np.array([0.2, 0.2, 0.2, 0.2, 0.2]), (4, 1))
    models = _ToyModels(p_hat, p_stage)
    df = pd.DataFrame({"category": ["jar"] * 4})

    _, z_spec = clustering.build_z(df, models, descriptor="behavior")

    # A "pool-shaped" frame with different predict_p outputs -- transform_z
    # must apply the FROZEN (diag-fit) mean/scale, not refit on this frame.
    pool_models = _ToyModels(np.array([0.9, 0.1]),
                              np.tile(np.array([0.2] * 5), (2, 1)))
    pool_df = pd.DataFrame({"category": ["jar", "jug"]})

    Z_pool, z_spec_2 = clustering.build_z(pool_df, pool_models, z_spec=z_spec)
    assert z_spec_2 is z_spec  # frozen spec returned unchanged, never refit

    expected = (np.array([0.9, 0.1]) - z_spec.p_hat_mean) / z_spec.p_hat_scale
    assert np.allclose(Z_pool[:, 0], expected)
    assert Z_pool.shape == (2, 6)


def test_zspec_roundtrip_through_dict_preserves_behavior_descriptor():
    p_hat = np.array([0.1, 0.4, 0.6, 0.9, 0.3, 0.7, 0.2, 0.5])
    p_stage = np.tile(np.array([0.3, 0.2, 0.2, 0.15, 0.15]), (8, 1))
    models = _ToyModels(p_hat, p_stage)
    df = pd.DataFrame({"category": ["jar"] * 8})

    _, z_spec = clustering.build_z(df, models, descriptor="behavior")
    restored = clustering.ZSpec.from_dict(z_spec.to_dict())

    assert restored.descriptor == "behavior"
    assert restored.knob_cols == ()
    Z1 = clustering.transform_z(df, models, z_spec)
    Z2 = clustering.transform_z(df, models, restored)
    assert np.allclose(Z1, Z2)
    assert Z1.shape == (8, 6)


def test_fit_z_spec_rejects_unknown_descriptor():
    df = _make_synthetic_diag_df(n_conditions=20, seed=42)
    models = _fit_models(df)
    diag = _diag_conditions(df)
    with pytest.raises(ValueError):
        clustering.fit_z_spec(diag, models, descriptor="nonsense")


# =============================================================================
# choose_k / merge_small
# =============================================================================

def test_choose_k_recovers_two_well_separated_blobs():
    rng = np.random.default_rng(0)
    blob_a = rng.normal(loc=-5.0, scale=0.3, size=(60, 4))
    blob_b = rng.normal(loc=5.0, scale=0.3, size=(60, 4))
    Z = np.vstack([blob_a, blob_b])

    # Real runs use config.K_RANGE (starts at 3); this test widens the range
    # to include k=2 so the recovered structure (2 true blobs) is reachable.
    k, labels, table = clustering.choose_k(Z, k_range=range(2, 6))

    assert k == 2
    a_labels = set(labels[:60].tolist())
    b_labels = set(labels[60:].tolist())
    assert len(a_labels) == 1
    assert len(b_labels) == 1
    assert a_labels != b_labels
    assert set(table["k"]) == {2, 3, 4, 5}
    assert int(table["chosen"].sum()) == 1


def test_choose_k_caps_at_max_arms_minus_one():
    rng = np.random.default_rng(1)
    # 8 tight, well-separated blobs: silhouette favors a high k, but
    # max_arms=5 -> cap=4 (non-Random) clusters must win instead.
    centers = np.arange(8) * 10.0
    blobs = [rng.normal(loc=c, scale=0.2, size=(20, 2)) for c in centers]
    Z = np.vstack(blobs)

    k, labels, table = clustering.choose_k(Z, k_range=range(3, 9), max_arms=5)

    assert k <= 4
    best_uncapped_k = int(table.loc[table["silhouette"].idxmax(), "k"])
    assert best_uncapped_k > 4  # confirms the cap mechanism actually fired


def test_merge_small_merges_a_3pct_cluster_into_nearest_big_centroid():
    # 100 points: cluster 0 = 60 near (0,0); cluster 1 = 37 near (10,10);
    # cluster 2 = 3 (3%) near (0.1,0.1) -- much closer to cluster 0's
    # centroid than cluster 1's -- must be reassigned into cluster 0.
    rng = np.random.default_rng(3)
    c0 = rng.normal(loc=0.0, scale=0.05, size=(60, 2))
    c1 = rng.normal(loc=10.0, scale=0.05, size=(37, 2))
    c2 = rng.normal(loc=0.1, scale=0.01, size=(3, 2))
    Z = np.vstack([c0, c1, c2])
    labels = np.array([0] * 60 + [1] * 37 + [2] * 3)

    merged = clustering.merge_small(labels, Z, frac=0.05)

    unique, counts = np.unique(merged, return_counts=True)
    fracs = counts / len(merged)
    assert np.all(fracs >= 0.05)
    assert len(unique) == 2
    assert set(unique.tolist()) == {0, 1}          # densely renumbered
    assert set(merged[97:100].tolist()) == {0}      # absorbed into the NEAR big cluster
    assert len(merged) == 100


def test_merge_small_noop_when_nothing_is_small():
    rng = np.random.default_rng(4)
    Z = np.vstack([rng.normal(0, 1, (50, 2)), rng.normal(10, 1, (50, 2))])
    labels = np.array([0] * 50 + [1] * 50)

    merged = clustering.merge_small(labels, Z, frac=0.05)
    assert np.array_equal(merged, labels)


def test_merge_small_degenerate_all_small_is_a_noop_besides_renumbering():
    # 3 clusters, none reaching 5% would be absurd at this size, so use a
    # tiny frac threshold that DOES make all 3 "small" relative to it... use
    # frac > every actual share to force the "no big cluster" branch.
    Z = np.array([[0.0], [0.0], [1.0], [1.0], [2.0], [2.0]])
    labels = np.array([0, 0, 1, 1, 2, 2])  # each holds 1/3 = 33%

    merged = clustering.merge_small(labels, Z, frac=0.5)  # nothing clears 50%

    assert set(np.unique(merged).tolist()) == {0, 1, 2}
    assert len(merged) == 6


# =============================================================================
# summarize / build_arms_entries
# =============================================================================

def test_summarize_cards_shape_and_content():
    df = _make_synthetic_diag_df(n_conditions=100, seed=5)
    models = _fit_models(df)
    diag = _diag_conditions(df)

    Z, z_spec = clustering.build_z(diag, models)
    k, labels, _ = clustering.choose_k(Z, k_range=range(3, 6))

    cards = clustering.summarize(labels, diag, models)

    assert len(cards) == k
    assert sum(c["size"] for c in cards) == len(diag)
    assert np.isclose(sum(c["share"] for c in cards), 1.0)

    for card in cards:
        assert 0.0 <= card["mean_p_hat"] <= 1.0
        assert card["dominant_stage"] in clustering.STAGES
        assert set(card["stage_distribution"]) == set(clustering.STAGES)
        assert np.isclose(sum(card["stage_distribution"].values()), 1.0)
        assert len(card["top_categories"]) <= 5
        assert set(card["centroid_knobs"]) == set(clustering.KNOB_COLS)

        stage_part, sep, dir_part = card["suggested_slug"].partition("-")
        assert sep == "-"
        assert stage_part == card["dominant_stage"]
        assert dir_part  # non-empty


def test_build_arms_entries_shapes_and_placeholder_names():
    df = _make_synthetic_diag_df(n_conditions=100, seed=6)
    models = _fit_models(df)
    diag = _diag_conditions(df)

    Z, z_spec = clustering.build_z(diag, models)
    k, labels, _ = clustering.choose_k(Z, k_range=range(3, 6))

    entries = clustering.build_arms_entries(labels, diag, models, z_spec, Z=Z)

    assert len(entries) == k
    assert sorted(e["index"] for e in entries) == list(range(k))
    assert np.isclose(sum(e["share"] for e in entries), 1.0)

    for e in entries:
        assert e["name"] == f"UNNAMED_{e['index']}"
        assert len(e["centroid"]["standardized"]) == 11
        assert set(e["centroid"]["raw"]) == set(clustering.KNOB_COLS)
        assert set(e["cov_diag"]) == set(clustering.KNOB_COLS)
        assert all(v >= 0 for v in e["cov_diag"].values())
        assert e["dominant_stage"] in clustering.STAGES


# =============================================================================
# CLI: draft -> hard stop -> finalize (subprocess, tmp dirs, testable overrides)
# =============================================================================

def _write_cli_fixture(tmp_path, n_conditions=60, replicas=50, n_d0=40):
    """Ledger episodes.parquet (phase=diag) + a fitted map_models.joblib +
    a pool parquet built by replicating each diag condition's own knob values
    (with tiny jitter) `replicas` times. Replication (rather than an
    independently-drawn pool) guarantees every surviving cluster (>=1
    merge_small-safe 5% share of the 60 diag conditions, i.e. >=3 conditions)
    gets a comfortably large W membership regardless of exactly how KMeans
    splits this run's Z -- so `wells.choose_B` reliably succeeds instead of
    depending on luck."""
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir()

    diag_df = _make_synthetic_diag_df(n_conditions=n_conditions, repeats=1, seed=10)
    diag_df = diag_df.assign(phase="diag")
    diag_df.to_parquet(ledger_dir / "episodes.parquet", index=False)

    models = _fit_models(diag_df)
    map_path = ledger_dir / "map_models.joblib"
    map_fit.save(models, map_path)

    rng = np.random.default_rng(11)
    rows = []
    idx = 0
    for _, r in diag_df.iterrows():
        for _ in range(replicas):
            rows.append(dict(
                episode_index=idx,
                category=r["category"],
                h=float(r["h"] + rng.normal(0, 0.002)),
                w=float(r["w"] + rng.normal(0, 0.002)),
                layout=0,
                x_rel=float(r["x_rel"] + rng.normal(0, 0.01)),
                y_rel=float(r["y_rel"] + rng.normal(0, 0.01)),
                side=int(r["side"]),
                traj_len=50,
                in_d0=(idx < n_d0),
            ))
            idx += 1
    pool_df = pd.DataFrame(rows)
    pool_path = tmp_path / "pool_demos.parquet"
    pool_df.to_parquet(pool_path, index=False)

    return ledger_dir, map_path, pool_path


def _run_cli(args):
    return subprocess.run(
        [sys.executable, "-m", "bandit_v1.clustering", *args],
        cwd=str(config.REPO), capture_output=True, text=True, timeout=120,
    )


def _produce_draft(tmp_path, name="cli1", descriptor="hybrid", k=None):
    base = tmp_path / name
    base.mkdir()
    ledger_dir, map_path, pool_path = _write_cli_fixture(base)
    draft_path = base / "arms_draft.yaml"
    args = [
        "--ledger-dir", str(ledger_dir),
        "--pool-parquet", str(pool_path),
        "--map-models-path", str(map_path),
        "--draft-path", str(draft_path),
        "--descriptor", descriptor,
    ]
    if k is not None:
        args += ["--k", str(k)]
    result = _run_cli(args)
    return result, draft_path, base


def test_cli_draft_then_hard_stop_nonzero_exit(tmp_path):
    result, draft_path, _ = _produce_draft(tmp_path)

    assert result.returncode != 0, result.stdout + result.stderr
    assert "HARD STOP" in result.stdout
    assert "--names" in result.stdout
    assert "SILHOUETTE_TABLE" in result.stdout
    assert "WELL_COUNT_TABLE" in result.stdout
    assert "PROPOSED_B" in result.stdout
    assert draft_path.exists()

    draft = yaml.safe_load(draft_path.read_text())
    assert draft["random_arm"] is True
    assert len(draft["arms"]) >= 1
    assert all(a["name"].startswith("UNNAMED_") for a in draft["arms"])
    for key in ("z_spec", "map_hash", "proposed_B", "frozen_at", "limiting_arm"):
        assert key in draft


def test_cli_k_override_pins_cluster_count(tmp_path):
    """--k pins the cluster count regardless of what the automatic
    silhouette sweep over config.K_RANGE would otherwise pick -- the owner
    override compute_draft's docstring describes (needed because the
    auto-picked k can be within refit noise of an adjacent candidate)."""
    auto_result, auto_draft_path, _ = _produce_draft(tmp_path, name="k_auto")
    auto_draft = yaml.safe_load(auto_draft_path.read_text())
    auto_k = len(auto_draft["arms"])

    pinned_k = 2  # config.K_RANGE starts at 3 -- 2 is NOT reachable by the
                  # automatic sweep at all, so this only appears if --k pins it
    pinned_result, pinned_draft_path, _ = _produce_draft(
        tmp_path, name="k_pinned", k=pinned_k)

    assert "HARD STOP" in pinned_result.stdout, pinned_result.stdout + pinned_result.stderr
    pinned_draft = yaml.safe_load(pinned_draft_path.read_text())
    assert len(pinned_draft["arms"]) == pinned_k
    assert f"chosen k={pinned_k}" in pinned_result.stdout
    # sanity: the pinned k really is different from what auto-sweep picked
    # (otherwise this test wouldn't distinguish "pinned" from "auto" at all)
    assert pinned_k != auto_k


def test_cli_behavior_descriptor_yaml_roundtrip_draft_and_finalize(tmp_path):
    """--descriptor behavior end-to-end: draft carries descriptor="behavior"
    + a 6-dim (not 11-dim) centroid, and finalize copies it through verbatim
    (clustering.finalize never recomputes -- see module docstring)."""
    result, draft_path, base = _produce_draft(tmp_path, name="behavior_cli",
                                               descriptor="behavior")
    assert "HARD STOP" in result.stdout, result.stdout + result.stderr

    draft = yaml.safe_load(draft_path.read_text())
    assert draft["z_spec"]["descriptor"] == "behavior"
    assert draft["z_spec"]["knob_cols"] == []
    n = len(draft["arms"])
    for arm in draft["arms"]:
        assert len(arm["centroid"]["standardized"]) == 6

    names = [f"bslug{i}" for i in range(n)]
    out_path = base / "arms.yaml"
    hashes_path = base / "hashes.json"
    result2 = _run_cli([
        "--draft-path", str(draft_path), "--out-path", str(out_path),
        "--hashes-path", str(hashes_path), "--names", ",".join(names),
    ])

    assert result2.returncode == 0, result2.stdout + result2.stderr
    final = yaml.safe_load(out_path.read_text())
    assert final["z_spec"]["descriptor"] == "behavior"
    assert final["z_spec"] == draft["z_spec"]
    for arm in final["arms"]:
        assert len(arm["centroid"]["standardized"]) == 6


def test_cli_names_wrong_count_errors_and_writes_nothing(tmp_path):
    _, draft_path, base = _produce_draft(tmp_path)
    draft = yaml.safe_load(draft_path.read_text())
    n = len(draft["arms"])

    out_path = base / "arms.yaml"
    hashes_path = base / "hashes.json"
    wrong_names = ",".join(f"slug{i}" for i in range(max(n - 1, 0)))  # one too few

    result = _run_cli([
        "--draft-path", str(draft_path),
        "--out-path", str(out_path),
        "--hashes-path", str(hashes_path),
        "--names", wrong_names,
    ])

    assert result.returncode != 0, result.stdout + result.stderr
    assert not out_path.exists()
    assert not hashes_path.exists()


def test_cli_valid_names_finalizes_and_hashes(tmp_path):
    _, draft_path, base = _produce_draft(tmp_path)
    draft = yaml.safe_load(draft_path.read_text())
    n = len(draft["arms"])
    names = [f"myslug{i}" for i in range(n)]

    out_path = base / "arms.yaml"
    hashes_path = base / "hashes.json"

    result = _run_cli([
        "--draft-path", str(draft_path),
        "--out-path", str(out_path),
        "--hashes-path", str(hashes_path),
        "--names", ",".join(names),
    ])

    assert result.returncode == 0, result.stdout + result.stderr
    assert out_path.exists()

    final = yaml.safe_load(out_path.read_text())
    ordered = sorted(final["arms"], key=lambda a: a["index"])
    assert [a["name"] for a in ordered] == names
    assert final["frozen_at"] == draft["frozen_at"]
    assert final["map_hash"] == draft["map_hash"]
    assert final["random_arm"] is True
    assert final["z_spec"] == draft["z_spec"]

    hashes = json.load(open(hashes_path))
    assert hashes["arms.yaml"] == ledger.file_hash(out_path)


# --- review fix: --names finalize wires append_arms_freeze_to_config_yaml ----

def test_cli_names_finalize_wires_arms_freeze_into_config_yaml(tmp_path):
    """A `--names` finalize run must now ALSO append the `arms_freeze:` note
    to config.yaml itself (previously this only ever happened via a separate,
    manual invocation -- see task-armsfreeze-report.md's "Real freeze run"
    section). well_table/B are re-evaluated against the newly-FROZEN
    centroids + the synthetic pool fixture (never a re-clustering)."""
    _, draft_path, base = _produce_draft(tmp_path, name="wire1")
    draft = yaml.safe_load(draft_path.read_text())
    n = len(draft["arms"])
    names = [f"wireslug{i}" for i in range(n)]

    pool_path = base / "pool_demos.parquet"
    map_path = base / "ledger" / "map_models.joblib"
    out_path = base / "arms.yaml"
    hashes_path = base / "hashes.json"
    cfg_path = base / "config.yaml"
    cfg_path.write_text("# pre-existing comment\nsomething_else: 1\n")

    result = _run_cli([
        "--draft-path", str(draft_path), "--out-path", str(out_path),
        "--hashes-path", str(hashes_path), "--config-yaml-path", str(cfg_path),
        "--pool-parquet", str(pool_path), "--map-models-path", str(map_path),
        "--names", ",".join(names),
    ])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "appended arms_freeze block" in result.stdout

    doc = yaml.safe_load(cfg_path.read_text())
    assert doc["something_else"] == 1  # pre-existing content preserved verbatim

    final = yaml.safe_load(out_path.read_text())
    freeze = doc["arms_freeze"]
    expected_names = [a["name"] for a in sorted(final["arms"], key=lambda a: a["index"])]
    assert freeze["names"] == expected_names
    assert freeze["descriptor"] == final["z_spec"]["descriptor"]
    assert freeze["k"] == len(final["arms"])
    assert freeze["map_hash"] == final["map_hash"]
    assert freeze["arms_frozen_at"] == final["frozen_at"]
    assert "random" in freeze["well_counts"]
    assert freeze["B"] > 0


def test_cli_names_finalize_does_not_double_append_arms_freeze_when_already_present(tmp_path):
    """Guard: a config.yaml that ALREADY has an arms_freeze block (the real
    ledger/config.yaml's normal state after the one real freeze -- see
    task-armsfreeze-report.md) must be left byte-for-byte untouched by a
    later --names finalize run, not grown a duplicate block."""
    _, draft_path, base = _produce_draft(tmp_path, name="wire2")
    draft = yaml.safe_load(draft_path.read_text())
    n = len(draft["arms"])
    names = [f"guardslug{i}" for i in range(n)]

    pool_path = base / "pool_demos.parquet"
    map_path = base / "ledger" / "map_models.joblib"
    out_path = base / "arms.yaml"
    cfg_path = base / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({"arms_freeze": {"B": 999, "names": ["preexisting"]}},
                                       sort_keys=False))
    before = cfg_path.read_text()

    result = _run_cli([
        "--draft-path", str(draft_path), "--out-path", str(out_path),
        "--config-yaml-path", str(cfg_path),
        "--pool-parquet", str(pool_path), "--map-models-path", str(map_path),
        "--names", ",".join(names),
    ])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "already has an arms_freeze block" in result.stdout
    assert cfg_path.read_text() == before  # untouched -- guard fired, no re-append
    assert out_path.exists()  # finalize itself still ran normally


def test_config_yaml_has_key_helper(tmp_path):
    missing = tmp_path / "no_such.yaml"
    assert clustering._config_yaml_has_key(missing, "arms_freeze") is False

    present = tmp_path / "present.yaml"
    present.write_text(yaml.safe_dump({"arms_freeze": {}, "other": 1}))
    assert clustering._config_yaml_has_key(present, "arms_freeze") is True
    assert clustering._config_yaml_has_key(present, "baseline") is False


def test_cli_names_duplicate_or_bad_slug_format_rejected(tmp_path):
    _, draft_path, base = _produce_draft(tmp_path)
    draft = yaml.safe_load(draft_path.read_text())
    n = len(draft["arms"])

    out_path = base / "arms.yaml"

    # duplicate names
    dup_names = ",".join(["same_slug"] * n)
    result = _run_cli([
        "--draft-path", str(draft_path), "--out-path", str(out_path),
        "--names", dup_names,
    ])
    assert result.returncode != 0
    assert not out_path.exists()

    # bad slug format (uppercase, not matching SLUG_RE)
    bad_names = ",".join([f"Bad-Slug{i}" for i in range(n)])
    result = _run_cli([
        "--draft-path", str(draft_path), "--out-path", str(out_path),
        "--names", bad_names,
    ])
    assert result.returncode != 0
    assert not out_path.exists()


def test_cli_finalize_without_draft_errors(tmp_path):
    missing_draft = tmp_path / "no_such_draft.yaml"
    out_path = tmp_path / "arms.yaml"
    result = _run_cli([
        "--draft-path", str(missing_draft), "--out-path", str(out_path),
        "--names", "slug0",
    ])
    assert result.returncode != 0
    assert not out_path.exists()
