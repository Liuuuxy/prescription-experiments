"""Tests for bandit_v1/cluster_study.py: the split-by-repeat helper, the
bootstrap ARI-stability helper (synthetic blobs), and the Jensen-Shannon
divergence helpers. Synthetic data only -- the real study run reads a ledger
snapshot via the CLI and is scratch-only."""
import numpy as np
import pandas as pd
import pytest

from bandit_v1 import cluster_study


# =============================================================================
# split_by_repeat
# =============================================================================

def test_split_by_repeat_partitions_rows_by_repeat_idx():
    df = pd.DataFrame({
        "start_id": ["a"] * 8 + ["b"] * 3,
        "repeat_idx": list(range(8)) + [0, 1, 2],
        "success": [True] * 11,
    })
    build, val = cluster_study.split_by_repeat(df)

    assert set(build["repeat_idx"]) == {0, 1, 2, 3}
    assert set(val["repeat_idx"]) == {4, 5, 6, 7}
    assert len(build) + len(val) == len(df)
    # condition "b" has only repeats 0-2: all on the build side, absent from val
    assert (build["start_id"] == "b").sum() == 3
    assert "b" not in set(val["start_id"])
    # condition "a" contributes exactly 4 rows to each side
    assert (build["start_id"] == "a").sum() == 4
    assert (val["start_id"] == "a").sum() == 4


def test_split_by_repeat_custom_build_repeats():
    df = pd.DataFrame({"start_id": ["a"] * 4, "repeat_idx": [0, 1, 2, 3]})
    build, val = cluster_study.split_by_repeat(df, build_repeats=(0, 1))
    assert set(build["repeat_idx"]) == {0, 1}
    assert set(val["repeat_idx"]) == {2, 3}


# =============================================================================
# bootstrap_stability_ari (synthetic blobs)
# =============================================================================

def test_bootstrap_stability_ari_on_separated_blobs():
    from sklearn.datasets import make_blobs
    Z, _ = make_blobs(n_samples=240, centers=3, cluster_std=0.3,
                      random_state=0)
    ref_labels, _ = cluster_study.fit_config("kmeans", Z, k=3)
    mean, sd, aris = cluster_study.bootstrap_stability_ari(
        Z, ref_labels, cluster_study.make_fit_labeler("kmeans", k=3),
        n_boot=10, seed=0)
    # well-separated blobs must be essentially perfectly stable
    assert mean > 0.95
    assert sd < 0.05
    assert len(aris) == 10


def test_bootstrap_stability_ari_is_deterministic():
    from sklearn.datasets import make_blobs
    Z, _ = make_blobs(n_samples=120, centers=3, cluster_std=1.5,
                      random_state=1)
    ref_labels, _ = cluster_study.fit_config("kmeans", Z, k=3)
    labeler = cluster_study.make_fit_labeler("kmeans", k=3)
    a = cluster_study.bootstrap_stability_ari(Z, ref_labels, labeler,
                                              n_boot=5, seed=0)
    b = cluster_study.bootstrap_stability_ari(Z, ref_labels, labeler,
                                              n_boot=5, seed=0)
    assert a[2] == b[2]


# =============================================================================
# js_divergence / within_cluster_js
# =============================================================================

def test_js_divergence_identical_and_disjoint():
    p = np.array([0.2, 0.3, 0.5])
    assert cluster_study.js_divergence(p, p) == pytest.approx(0.0, abs=1e-12)
    # disjoint supports: JSD (base 2) is exactly 1
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert cluster_study.js_divergence(a, b) == pytest.approx(1.0, abs=1e-12)


def test_js_divergence_symmetric_and_bounded():
    rng = np.random.default_rng(0)
    for _ in range(5):
        p = rng.dirichlet(np.ones(5))
        q = rng.dirichlet(np.ones(5))
        d_pq = cluster_study.js_divergence(p, q)
        d_qp = cluster_study.js_divergence(q, p)
        assert d_pq == pytest.approx(d_qp, abs=1e-12)
        assert 0.0 <= d_pq <= 1.0


def test_within_cluster_js_zero_for_pure_clusters():
    profiles = np.array([
        [1.0, 0.0], [1.0, 0.0],   # cluster 0: identical profiles
        [0.0, 1.0], [0.0, 1.0],   # cluster 1: identical profiles
    ])
    labels = np.array([0, 0, 1, 1])
    assert cluster_study.within_cluster_js(profiles, labels) == pytest.approx(
        0.0, abs=1e-12)
    # mixing the two profiles inside one cluster must strictly increase it
    mixed = cluster_study.within_cluster_js(profiles, np.array([0, 1, 0, 1]))
    assert mixed > 0.1
