"""Cluster-validation study for the bandit_v1 arms step (Task 10 audit).

Motivation: the owner doubts (a) the z-descriptor choice (the Task-10 hybrid
z = [standardized knobs | p_hat | p_stage]) and (b) k-means specifically.
This module runs a descriptor x method grid with standard metrics and STOPS
for human review -- it never writes arms.yaml and never touches ledger/
(all outputs go to an explicit --out-dir; the episodes/diag_conditions/pool
inputs are plain read-only paths, so a live run can be audited from a
snapshot copy).

--- Study design (implemented exactly; see the task brief) ---

1. TRAIN/VAL split BY REPEAT: repeats 0-3 are the "build" side (map fit,
   empirical descriptors, clustering); repeats 4-7 are held-out outcomes used
   ONLY for external-validity metrics. Conditions missing the val side are
   excluded from external metrics only (they still cluster). ONE map is fit
   on build-side rows only: `map_fit.compare_families` picks the family,
   `map_fit.fit(build, family=winner)` is saved to the out dir. All model
   outputs everywhere in this study come from that build-side map.

2. DESCRIPTORS over the diag conditions (block standardization follows
   clustering.fit_z_spec's scheme exactly: knob-like blocks are standardized
   PER-COLUMN; probability blocks (p_hat / p_stage / empirical profiles) are
   centered per-column then scaled by ONE shared scalar so the block's TOTAL
   variance is 1 -- see clustering.py's module docstring for the rationale;
   a 1-dim shared-total block is numerically identical to per-column):
     Z1_hybrid          [knobs(5) | p_hat(1) | p_stage(5)]   (current design;
                        verified at fit time to match clustering.build_z
                        bit-for-bit, so this study audits the REAL descriptor)
     Z2_knobs           [knobs(5)]
     Z3_behavior        [p_hat(1) | p_stage(5)]
     Z4_knobs_empirical [knobs(5) | build-side empirical succ_frac(1) |
                        build-side empirical stage_dist(5)]  (NOT
                        pool-deployable: well demos have no empirical
                        rollout profiles)
     Z5_knobs_catte     [knobs(5) | cat_te(1)]  (cat_te from the build map's
                        shrinkage encoding, via models.predict_features)
     Z6_hybrid_absx     Z1 + |x_rel| as an extra per-column knob column

3. METHODS: KMeans(n_init=50, rs=0), GaussianMixture(full, n_init=5, rs=0),
   AgglomerativeClustering(ward), sklearn HDBSCAN(min_cluster_size=15; if
   unavailable, DBSCAN fallback -- flagged in the results). k-parameterized
   methods sweep k=3..8 and pick per-method best k by silhouette; HDBSCAN
   chooses its own k (noise fraction reported; noise rows keep label -1 and
   count as their OWN group in every internal/triviality/cross-view metric).

4. METRICS per (descriptor, method) best config:
   - internal: silhouette, Davies-Bouldin, Calinski-Harabasz;
   - STABILITY ARI: 20 bootstraps (resample conditions with replacement,
     refit the same config, label ALL conditions by nearest-centroid /
     predict, ARI vs the reference labels); KMeans additionally gets a
     seed-stability ARI over 10 alternative seeds vs the rs=0 reference;
   - EXTERNAL VALIDITY on held-out repeats 4-7 (never used in any
     descriptor or the map): eta^2 + F + p of per-condition val success
     across clusters, and mean within-cluster Jensen-Shannon divergence
     (base 2) of val stage profiles to the cluster-mean profile, compared
     against a same-sizes random-partition baseline (100 permutations;
     js_perm_p = fraction of permutations doing at least as well, lower
     observed JS than baseline mean = purer than chance);
   - TRIVIALITY: ARI vs one-axis partitions (side sign, height terciles,
     |x_rel| terciles, prior tercile parsed from diag_conditions' cell
     prefix) -- a high value means the clustering is just re-deriving a
     single knob;
   - CROSS-VIEW: ARI matrix between the best config per descriptor (best =
     highest held-out eta^2, ties/NaN broken by bootstrap ARI);
   - ACTIONABILITY per best-config candidate: cluster shares (min share),
     pool deployability (Z4 flagged NOT deployable), well counts per
     cluster by nearest-centroid over the pool's W rows (embedded with the
     SAME frozen descriptor spec + build-side map), the B rule
     (wells.choose_B), and a nameability card per cluster (top categories,
     raw centroid knobs, dominant model-predicted stage, mean p_hat, mean
     build-side empirical success).

5. OUTPUTS: results.json + a compact report.md in --out-dir; the headline
   comparison table is printed. Deterministic: every stochastic step has a
   fixed seed and a config-local RNG (order of configs cannot change any
   number).

HDBSCAN bootstrap note: the reference labeling may contain a -1 noise group,
but the bootstrap full-relabeling path (nearest non-noise centroid) never
emits -1, so noise instability is charged AGAINST the stability score
(conservative, and consistent across bootstraps).
"""
import argparse
import json
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.mixture import GaussianMixture

from . import clustering, map_fit, pool, wells
from .draw import RANDOM_ARM

try:  # sklearn >= 1.3 ships HDBSCAN; fall back to DBSCAN if absent (flagged).
    from sklearn.cluster import HDBSCAN as _HDBSCAN
    HDBSCAN_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    from sklearn.cluster import DBSCAN as _DBSCAN
    HDBSCAN_AVAILABLE = False

# --- frozen study constants ---------------------------------------------------

BUILD_REPEATS = (0, 1, 2, 3)          # build side; everything else is held out
K_SWEEP = tuple(range(3, 9))          # k=3..8 for k-parameterized methods
N_BOOTSTRAP = 20
BOOTSTRAP_SEED = 0
N_SEED_STABILITY = 10                 # KMeans alternative seeds 1..10 vs rs=0
N_PERMUTATIONS = 100
PERMUTATION_SEED = 0
HDBSCAN_MIN_CLUSTER_SIZE = 15
NOISE_LABEL = -1

KMEANS_N_INIT = 50
GMM_N_INIT = 5
METHOD_RANDOM_STATE = 0

STAGES = map_fit.STAGES
KNOB_COLS = clustering.KNOB_COLS
STAGE_COLS = [f"stage_{s}" for s in STAGES]

DESCRIPTORS = ("Z1_hybrid", "Z2_knobs", "Z3_behavior", "Z4_knobs_empirical",
               "Z5_knobs_catte", "Z6_hybrid_absx")
METHODS = ("kmeans", "gmm", "agglo", "hdbscan")

_CELL_TERCILE_RE = re.compile(r"^t(\d+)")


# =============================================================================
# split + empirical profiles
# =============================================================================

def split_by_repeat(df_diag: pd.DataFrame, build_repeats=BUILD_REPEATS):
    """(build_df, val_df): rows whose `repeat_idx` is in `build_repeats` vs
    ALL remaining rows. A condition (start_id) present only on one side is
    simply absent from the other -- callers decide what that means (here:
    build-side-only conditions still cluster but are excluded from external
    metrics; val-side-only conditions are dropped from the study entirely,
    since no descriptor/map input exists for them)."""
    build_mask = df_diag["repeat_idx"].isin(build_repeats)
    build = df_diag[build_mask].reset_index(drop=True)
    val = df_diag[~build_mask].reset_index(drop=True)
    return build, val


def empirical_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Per-condition empirical outcome profile: one row per start_id present
    in `df`, columns [n, succ_frac, stage_<s> for STAGES] -- stage columns
    are the empirical distribution of `failure_stage` over the 5 canonical
    STAGES (rows sum to 1; `success` rollouts land in stage_success)."""
    stage = df["failure_stage"].astype(str)
    rows = {}
    for sid, g in df.groupby("start_id"):
        counts = stage.loc[g.index].value_counts()
        n = len(g)
        dist = np.array([counts.get(s, 0) for s in STAGES], dtype=float) / n
        rows[sid] = [n, float(g["success"].mean()), *dist]
    out = pd.DataFrame.from_dict(rows, orient="index",
                                 columns=["n", "succ_frac", *STAGE_COLS])
    out.index.name = "start_id"
    return out


# =============================================================================
# descriptors (block standardization following clustering.fit_z_spec's scheme)
# =============================================================================

def _fit_block(X: np.ndarray, mode: str):
    """(mean, scale) for one block. "per_column": ordinary per-column
    standardization (zero-variance columns divided by 1, same guard as
    clustering._safe_scale). "shared_total": per-column centering + ONE
    shared scalar making the block's total variance 1 (clustering.py's
    p_hat/p_stage rule; identical to per_column when the block is 1-dim)."""
    mean = X.mean(axis=0)
    if mode == "per_column":
        scale = clustering._safe_scale(X.std(axis=0))
    elif mode == "shared_total":
        total_var = float(np.mean((X - mean) ** 2, axis=0).sum())
        scale = float(np.sqrt(total_var)) if total_var > 0 else 1.0
    else:
        raise ValueError(f"unknown block mode {mode!r}")
    return mean, scale


@dataclass
class DescriptorSpec:
    """Frozen standardization for one descriptor: fitted on the diag
    conditions, applied unchanged to any other feature table (the pool's W
    rows) -- same fit-once-transform-anywhere contract as clustering.ZSpec."""
    name: str
    pool_deployable: bool
    blocks: list  # [(block_name, mode, mean ndarray, scale ndarray|float)]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "pool_deployable": self.pool_deployable,
            "blocks": [
                {"block": b, "mode": m,
                 "mean": np.asarray(mu).ravel().tolist(),
                 "scale": (np.asarray(sc).ravel().tolist()
                           if isinstance(sc, np.ndarray) else float(sc))}
                for b, m, mu, sc in self.blocks
            ],
        }


def _raw_blocks(name: str, features_df: pd.DataFrame, models,
                emp_df: pd.DataFrame = None) -> list:
    """[(block_name, mode, raw ndarray)] for descriptor `name` over
    `features_df` (one row per condition, columns [start_id?, category,
    KNOB_COLS]). `emp_df` (empirical_profiles output, indexed by start_id) is
    required for Z4 only -- passing None there raises, which is exactly the
    pool-deployability boundary (well demos have no empirical profiles)."""
    knobs = features_df[list(KNOB_COLS)].to_numpy(dtype=float)

    def model_blocks():
        p_hat = models.predict_p(features_df).astype(float).reshape(-1, 1)
        p_stage = models.predict_stage(features_df).astype(float)
        return [("p_hat", "shared_total", p_hat),
                ("p_stage", "shared_total", p_stage)]

    if name == "Z1_hybrid":
        return [("knobs", "per_column", knobs)] + model_blocks()
    if name == "Z2_knobs":
        return [("knobs", "per_column", knobs)]
    if name == "Z3_behavior":
        return model_blocks()
    if name == "Z4_knobs_empirical":
        if emp_df is None:
            raise ValueError(
                "Z4_knobs_empirical needs build-side empirical profiles "
                "(emp_df) -- it is NOT computable for pool/well rows")
        emp = emp_df.reindex(features_df["start_id"].to_numpy())
        assert not emp["succ_frac"].isna().any(), (
            "Z4: some conditions have no build-side empirical profile")
        succ = emp["succ_frac"].to_numpy(dtype=float).reshape(-1, 1)
        stage = emp[STAGE_COLS].to_numpy(dtype=float)
        return [("knobs", "per_column", knobs),
                ("emp_succ", "shared_total", succ),
                ("emp_stage", "shared_total", stage)]
    if name == "Z5_knobs_catte":
        cat_te = (models.predict_features(features_df)["cat_te"]
                  .to_numpy(dtype=float).reshape(-1, 1))
        return [("knobs", "per_column", knobs), ("cat_te", "per_column", cat_te)]
    if name == "Z6_hybrid_absx":
        absx = np.abs(features_df["x_rel"].to_numpy(dtype=float)).reshape(-1, 1)
        return ([("knobs", "per_column", knobs)] + model_blocks()
                + [("abs_x_rel", "per_column", absx)])
    raise ValueError(f"unknown descriptor {name!r}")


def fit_descriptor(name: str, features_df: pd.DataFrame, models,
                   emp_df: pd.DataFrame = None):
    """(Z, DescriptorSpec) fitted on `features_df`. For Z1_hybrid the result
    is verified to match clustering.build_z EXACTLY (same knob/p_hat/p_stage
    blocks, same scaling scheme) so this study audits the real production
    descriptor, not a reimplementation drift of it."""
    raw = _raw_blocks(name, features_df, models, emp_df=emp_df)
    fitted, cols = [], []
    for block_name, mode, X in raw:
        mean, scale = _fit_block(X, mode)
        fitted.append((block_name, mode, mean, scale))
        cols.append((X - mean) / scale)
    Z = np.hstack(cols)
    spec = DescriptorSpec(name=name, pool_deployable=(name != "Z4_knobs_empirical"),
                          blocks=fitted)
    if name == "Z1_hybrid":
        Z_ref, _ = clustering.build_z(features_df, models)
        assert np.allclose(Z, Z_ref, atol=1e-10), (
            "Z1_hybrid deviates from clustering.build_z -- study descriptor "
            "no longer audits the production one")
    return Z, spec


def transform_descriptor(spec: DescriptorSpec, features_df: pd.DataFrame,
                         models, emp_df: pd.DataFrame = None) -> np.ndarray:
    """Apply an ALREADY-FITTED spec to another feature table (the pool-row
    path). Raises for Z4 without emp_df -- see _raw_blocks."""
    raw = _raw_blocks(spec.name, features_df, models, emp_df=emp_df)
    cols = []
    for (block_name, mode, X), (fb, fm, mean, scale) in zip(raw, spec.blocks):
        assert block_name == fb and mode == fm
        cols.append((X - mean) / scale)
    return np.hstack(cols)


# =============================================================================
# methods (fit + full-relabel closures)
# =============================================================================

def _centroids_from_labels(Z: np.ndarray, labels: np.ndarray):
    """(centroids, uniq_labels) -- member-mean centroid per non-noise label
    (ascending). Noise (-1) never gets a centroid: it is a leftover set, not
    an arm/predictable region."""
    uniq = [int(u) for u in np.unique(labels) if int(u) != NOISE_LABEL]
    if not uniq:
        return np.zeros((0, Z.shape[1])), []
    centroids = np.stack([Z[labels == u].mean(axis=0) for u in uniq])
    return centroids, uniq


def _nearest_centroid_labels(Z: np.ndarray, centroids: np.ndarray,
                             uniq: list) -> np.ndarray:
    if len(uniq) == 0:
        return np.zeros(len(Z), dtype=int)
    d = np.linalg.norm(Z[:, None, :] - centroids[None, :, :], axis=2)
    return np.asarray(uniq, dtype=int)[np.argmin(d, axis=1)]


def fit_config(method: str, Z: np.ndarray, k: int = None,
               seed: int = METHOD_RANDOM_STATE):
    """Fit one clustering config on `Z`. Returns (labels, label_full_fn)
    where `label_full_fn(Z_any)` labels arbitrary rows in the same space:
    KMeans/GMM use their own .predict; ward-agglomerative and HDBSCAN (which
    have no predict) use nearest member-mean centroid -- the exact rule
    wells.assign_regions deploys, so stability is measured under the same
    labeling rule the pool would experience."""
    if method == "kmeans":
        km = KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=seed).fit(Z)
        return km.labels_.astype(int), lambda Q: km.predict(Q).astype(int)
    if method == "gmm":
        gm = GaussianMixture(n_components=k, covariance_type="full",
                             n_init=GMM_N_INIT, random_state=seed).fit(Z)
        return gm.predict(Z).astype(int), lambda Q: gm.predict(Q).astype(int)
    if method == "agglo":
        ac = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(Z)
        labels = ac.labels_.astype(int)
        centroids, uniq = _centroids_from_labels(Z, labels)
        return labels, lambda Q: _nearest_centroid_labels(Q, centroids, uniq)
    if method == "hdbscan":
        if HDBSCAN_AVAILABLE:
            hd = _HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE).fit(Z)
        else:  # pragma: no cover - environment-dependent fallback
            hd = _DBSCAN(min_samples=HDBSCAN_MIN_CLUSTER_SIZE).fit(Z)
        labels = hd.labels_.astype(int)
        centroids, uniq = _centroids_from_labels(Z, labels)
        return labels, lambda Q: _nearest_centroid_labels(Q, centroids, uniq)
    raise ValueError(f"unknown method {method!r}")


def make_fit_labeler(method: str, k: int = None, seed: int = METHOD_RANDOM_STATE):
    """Closure for bootstrap_stability_ari: Z_fit -> label_full_fn."""
    def _fit(Z_fit):
        _, label_fn = fit_config(method, Z_fit, k=k, seed=seed)
        return label_fn
    return _fit


def _safe_silhouette(Z, labels):
    if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(Z):
        return float("nan")
    return float(silhouette_score(Z, labels))


def sweep_k(method: str, Z: np.ndarray, k_range=K_SWEEP):
    """Per-method k sweep: fit each k, silhouette on the FULL Z (n is small
    here -- no subsample needed), best k = argmax silhouette. Returns
    (best_k, sweep_rows)."""
    rows = []
    for k in k_range:
        labels, _ = fit_config(method, Z, k=k)
        rows.append({"k": int(k), "silhouette": _safe_silhouette(Z, labels)})
    valid = [r for r in rows if np.isfinite(r["silhouette"])]
    if not valid:
        raise ValueError(f"sweep_k({method}): no k in {list(k_range)} produced "
                         f"a valid silhouette")
    best_k = max(valid, key=lambda r: r["silhouette"])["k"]
    return best_k, rows


# =============================================================================
# stability
# =============================================================================

def bootstrap_stability_ari(Z: np.ndarray, ref_labels: np.ndarray, fit_labeler,
                            n_boot: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED):
    """Bootstrap stability: `n_boot` times, resample rows of `Z` with
    replacement, refit the same config on the resample (`fit_labeler(Z_boot)
    -> label_full_fn`), relabel ALL original rows, and score ARI vs
    `ref_labels`. Returns (mean, sd, per-bootstrap list). ARI is invariant to
    label permutation, so no matching step is needed."""
    rng = np.random.default_rng(seed)
    n = len(Z)
    aris = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        label_fn = fit_labeler(Z[idx])
        aris.append(float(adjusted_rand_score(ref_labels, label_fn(Z))))
    return float(np.mean(aris)), float(np.std(aris)), aris


def kmeans_seed_stability(Z: np.ndarray, k: int, ref_labels: np.ndarray,
                          n_seeds: int = N_SEED_STABILITY):
    """ARI of KMeans(random_state=s) for s=1..n_seeds vs the rs=0 reference
    labels (mean, sd). With n_init=50 this should be ~1.0 unless the
    objective landscape genuinely has competing optima."""
    aris = []
    for s in range(1, n_seeds + 1):
        labels, _ = fit_config("kmeans", Z, k=k, seed=s)
        aris.append(float(adjusted_rand_score(ref_labels, labels)))
    return float(np.mean(aris)), float(np.std(aris))


# =============================================================================
# external validity (held-out repeats)
# =============================================================================

def eta_squared_anova(values: np.ndarray, labels: np.ndarray):
    """One-way ANOVA of `values` across the groups in `labels`:
    (eta_squared, F, p). NaN triple when degenerate (single group, zero total
    variance, or no residual dof)."""
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    n, k = len(values), len(uniq)
    grand = values.mean()
    ss_total = float(((values - grand) ** 2).sum())
    if k < 2 or n <= k or ss_total == 0:
        return float("nan"), float("nan"), float("nan")
    ss_between = float(sum(
        (labels == u).sum() * (values[labels == u].mean() - grand) ** 2
        for u in uniq))
    ss_within = ss_total - ss_between
    df1, df2 = k - 1, n - k
    eta2 = ss_between / ss_total
    if ss_within <= 0:
        return float(eta2), float("inf"), 0.0
    F = (ss_between / df1) / (ss_within / df2)
    return float(eta2), float(F), float(scipy_stats.f.sf(F, df1, df2))


def js_divergence(p, q, base: float = 2.0) -> float:
    """Jensen-Shannon divergence between two discrete distributions (base 2
    by default, so the value lives in [0, 1]). Inputs are normalized
    defensively; an all-zero input is treated as uniform."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum() if p.sum() > 0 else np.full_like(p, 1.0 / len(p))
    q = q / q.sum() if q.sum() > 0 else np.full_like(q, 1.0 / len(q))
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask]))) / np.log(base)

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def within_cluster_js(profiles: np.ndarray, labels: np.ndarray) -> float:
    """Mean over all rows of JSD(row profile, member-mean profile of the
    row's cluster). Lower = stage profiles purer within clusters."""
    vals = []
    for u in np.unique(labels):
        P = profiles[labels == u]
        mean_prof = P.mean(axis=0)
        vals.extend(js_divergence(P[i], mean_prof) for i in range(len(P)))
    return float(np.mean(vals))


def permutation_js_baseline(profiles: np.ndarray, labels: np.ndarray,
                            n_perm: int = N_PERMUTATIONS,
                            seed: int = PERMUTATION_SEED):
    """(observed, baseline_mean, p): within_cluster_js observed vs `n_perm`
    random same-sizes partitions (label vector permutations). p = fraction of
    permutations scoring <= observed (add-one smoothed) -- small p means the
    clustering beats chance at stage-profile purity."""
    obs = within_cluster_js(profiles, labels)
    rng = np.random.default_rng(seed)
    perm_stats = [within_cluster_js(profiles, rng.permutation(labels))
                  for _ in range(n_perm)]
    p = (1 + sum(s <= obs for s in perm_stats)) / (n_perm + 1)
    return obs, float(np.mean(perm_stats)), float(p)


# =============================================================================
# triviality (one-axis partitions)
# =============================================================================

def one_axis_partitions(features_df: pd.DataFrame,
                        cell: pd.Series = None) -> dict:
    """{name: label array (NaN = row excluded from that partition's ARI)}:
    side sign, h terciles, |x_rel| terciles, and (when `cell` -- the
    diag_conditions cell string, e.g. "t2h0x0", aligned to features_df -- is
    given) the prior tercile parsed from the cell's leading t<digit>."""
    parts = {
        "side_sign": (features_df["side"].to_numpy(dtype=float) > 0).astype(float),
        "h_tercile": pd.qcut(features_df["h"], 3, labels=False,
                             duplicates="drop").to_numpy(dtype=float),
        "absx_tercile": pd.qcut(features_df["x_rel"].abs(), 3, labels=False,
                                duplicates="drop").to_numpy(dtype=float),
    }
    if cell is not None:
        def _parse(c):
            m = _CELL_TERCILE_RE.match(str(c))
            return float(m.group(1)) if m else float("nan")
        parts["prior_tercile"] = cell.map(_parse).to_numpy(dtype=float)
    return parts


def triviality_aris(labels: np.ndarray, partitions: dict) -> dict:
    out = {}
    for name, part in partitions.items():
        mask = np.isfinite(part)
        out[name] = (float(adjusted_rand_score(part[mask].astype(int),
                                               np.asarray(labels)[mask]))
                     if mask.sum() >= 2 else float("nan"))
    return out


# =============================================================================
# actionability
# =============================================================================

def nameability_cards(labels: np.ndarray, features_df: pd.DataFrame, models,
                      emp_build: pd.DataFrame) -> list:
    """Per-cluster naming card: size/share, top-3 categories, raw centroid
    knobs, dominant model-predicted stage (argmax vote, clustering.summarize's
    convention -- kept model-based for every descriptor so cards are
    comparable), mean model p_hat, mean build-side empirical success."""
    labels = np.asarray(labels)
    n = len(features_df)
    p_hat = models.predict_p(features_df).astype(float)
    p_stage = models.predict_stage(features_df).astype(float)
    dominant_row_stage = np.asarray(STAGES)[np.argmax(p_stage, axis=1)]
    emp_succ = emp_build.reindex(features_df["start_id"].to_numpy())["succ_frac"] \
        .to_numpy(dtype=float)

    cards = []
    for u in sorted(int(v) for v in np.unique(labels)):
        mask = labels == u
        size = int(mask.sum())
        top = features_df.loc[mask, "category"].value_counts().head(3)
        stage_votes = pd.Series(dominant_row_stage[mask]).value_counts()
        cards.append({
            "cluster": ("noise" if u == NOISE_LABEL else int(u)),
            "size": size,
            "share": size / n,
            "top_categories": [{"category": str(c), "count": int(cnt)}
                               for c, cnt in top.items()],
            "centroid_knobs": {c: float(features_df.loc[mask, c].mean())
                               for c in KNOB_COLS},
            "dominant_stage": str(stage_votes.idxmax()),
            "mean_p_hat": float(p_hat[mask].mean()),
            "mean_build_succ": float(np.nanmean(emp_succ[mask])),
        })
    return cards


def pool_actionability(spec: DescriptorSpec, Z_diag: np.ndarray,
                       labels: np.ndarray, pool_df: pd.DataFrame, models) -> dict:
    """Deploy the candidate onto the pool: embed W rows (pool.well_mask) with
    the frozen spec + build-side map, nearest member-mean centroid (noise
    clusters excluded -- they are not arms), well counts per cluster, and the
    B rule via wells.choose_B (Random row = |W| by definition). Z4 returns a
    not-deployable marker instead."""
    if not spec.pool_deployable:
        return {"pool_deployable": False,
                "note": "empirical build-side profiles do not exist for well "
                        "demos -- descriptor cannot score the pool"}
    well_df = pool_df[pool.well_mask(pool_df)].reset_index(drop=True)
    Zw = transform_descriptor(spec, well_df, models)
    centroids, uniq = _centroids_from_labels(Z_diag, np.asarray(labels))
    assign = _nearest_centroid_labels(Zw, centroids, uniq)
    rows = [{"arm": f"c{u}", "count": int((assign == u).sum())} for u in uniq]
    rows.append({"arm": RANDOM_ARM, "count": int(len(assign))})
    tbl = pd.DataFrame(rows)
    try:
        B, limiting = wells.choose_B(tbl)
        b_result = {"B": int(B), "limiting_arm": str(limiting)}
    except ValueError as e:
        b_result = {"B": None, "limiting_arm": None, "error": str(e)}
    return {"pool_deployable": True, "n_well": int(len(assign)),
            "well_counts": {f"c{u}": int((assign == u).sum()) for u in uniq},
            **b_result}


# =============================================================================
# study driver
# =============================================================================

def _load_inputs(episodes_parquet, diag_conditions_parquet, pool_parquet):
    ep = pd.read_parquet(episodes_parquet)
    diag = ep[ep["phase"] == "diag"].reset_index(drop=True)
    if len(diag) == 0:
        raise ValueError(f"no phase=='diag' rows in {episodes_parquet}")
    cond = pd.read_parquet(diag_conditions_parquet)
    pool_df = (pd.read_parquet(pool_parquet) if pool_parquet is not None
               else pool.build_pool_table(write=False))
    return diag, cond, pool_df


def _select_best(rows: list) -> dict:
    """Best config among one descriptor's rows: highest held-out eta^2
    (NaN -> -inf), ties broken by bootstrap ARI mean."""
    def key(r):
        e = r["external"]["eta_squared"]
        b = r["stability"]["bootstrap_ari_mean"]
        return (e if np.isfinite(e) else -np.inf,
                b if np.isfinite(b) else -np.inf)
    return max(rows, key=key)


def run_study(episodes_parquet, diag_conditions_parquet, out_dir,
              pool_parquet=None, preview_label: str = None) -> dict:
    """Full study (module docstring). Writes map_models_build.joblib,
    results.json, report.md into `out_dir`; returns the results dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    diag, cond, pool_df = _load_inputs(episodes_parquet, diag_conditions_parquet,
                                       pool_parquet)

    # 1. split + build-side map + empirical profiles ---------------------------
    build, val = split_by_repeat(diag)
    features_df = (build.drop_duplicates(subset="start_id", keep="first")
                   .reset_index(drop=True)[["start_id", "category", *KNOB_COLS]])
    n_cond = len(features_df)

    comparison = map_fit.compare_families(build)
    winner = comparison["winner"]
    models = map_fit.fit(build, family=winner)
    map_path = map_fit.save(models, path=out_dir / "map_models_build.joblib")

    emp_build = empirical_profiles(build)
    emp_val = empirical_profiles(val)
    ext_mask = features_df["start_id"].isin(emp_val.index).to_numpy()
    emp_val_aligned = emp_val.reindex(
        features_df.loc[ext_mask, "start_id"].to_numpy())
    val_succ = emp_val_aligned["succ_frac"].to_numpy(dtype=float)
    val_profiles = emp_val_aligned[STAGE_COLS].to_numpy(dtype=float)

    cell = (cond.set_index("start_id")["cell"]
            .reindex(features_df["start_id"].to_numpy())
            .reset_index(drop=True)) if "cell" in cond.columns else None
    partitions = one_axis_partitions(features_df, cell=cell)

    meta = {
        "preview_label": preview_label,
        "episodes_parquet": str(episodes_parquet),
        "diag_conditions_parquet": str(diag_conditions_parquet),
        "n_diag_rows": int(len(diag)),
        "n_build_rows": int(len(build)),
        "n_val_rows": int(len(val)),
        "n_conditions": int(n_cond),
        "n_conditions_planned": int(len(cond)),
        "n_conditions_external": int(ext_mask.sum()),
        "build_repeats": list(BUILD_REPEATS),
        "map_family_winner": winner,
        "map_comparison": comparison,
        "map_models_path": str(map_path),
        "hdbscan_available": HDBSCAN_AVAILABLE,
        "hdbscan_fallback": None if HDBSCAN_AVAILABLE else
            "sklearn HDBSCAN unavailable -- DBSCAN(min_samples=%d) used"
            % HDBSCAN_MIN_CLUSTER_SIZE,
    }

    # 2-4. descriptor x method grid -------------------------------------------
    per_config, best_per_descriptor, spec_store, Z_store = [], {}, {}, {}
    for dname in DESCRIPTORS:
        Z, spec = fit_descriptor(dname, features_df, models, emp_df=emp_build)
        spec_store[dname], Z_store[dname] = spec, Z
        rows = []
        for method in METHODS:
            if method == "hdbscan":
                labels, _ = fit_config(method, Z)
                k = int(len([u for u in np.unique(labels) if u != NOISE_LABEL]))
                sweep_rows, noise_frac = None, float((labels == NOISE_LABEL).mean())
            else:
                k, sweep_rows = sweep_k(method, Z)
                labels, _ = fit_config(method, Z, k=k)
                noise_frac = 0.0

            sil = _safe_silhouette(Z, labels)
            multi = len(np.unique(labels)) >= 2
            db = float(davies_bouldin_score(Z, labels)) if multi else float("nan")
            ch = float(calinski_harabasz_score(Z, labels)) if multi else float("nan")

            boot_mean, boot_sd, _ = bootstrap_stability_ari(
                Z, labels, make_fit_labeler(method, k=k))
            seed_mean, seed_sd = (kmeans_seed_stability(Z, k, labels)
                                  if method == "kmeans" else (None, None))

            eta2, F, p = eta_squared_anova(val_succ, np.asarray(labels)[ext_mask])
            js_obs, js_base, js_p = permutation_js_baseline(
                val_profiles, np.asarray(labels)[ext_mask])

            sizes = {str(int(u)): int((np.asarray(labels) == u).sum())
                     for u in np.unique(labels)}
            row = {
                "descriptor": dname, "method": method, "k": int(k),
                "noise_frac": noise_frac, "sizes": sizes,
                "k_sweep": sweep_rows,
                "internal": {"silhouette": sil, "davies_bouldin": db,
                             "calinski_harabasz": ch},
                "stability": {"bootstrap_ari_mean": boot_mean,
                              "bootstrap_ari_sd": boot_sd,
                              "seed_ari_mean": seed_mean,
                              "seed_ari_sd": seed_sd},
                "external": {"eta_squared": eta2, "F": F, "p": p,
                             "js_within": js_obs, "js_baseline_mean": js_base,
                             "js_perm_p": js_p,
                             "n_external": int(ext_mask.sum())},
                "triviality_ari": triviality_aris(labels, partitions),
            }
            rows.append(row)
            per_config.append(row)
            row["_labels"] = np.asarray(labels)  # stripped before JSON

        best = _select_best(rows)
        best_per_descriptor[dname] = {"method": best["method"], "k": best["k"]}

    # cross-view ARI between best config per descriptor ------------------------
    best_labels = {}
    for dname in DESCRIPTORS:
        b = best_per_descriptor[dname]
        row = next(r for r in per_config
                   if r["descriptor"] == dname and r["method"] == b["method"])
        best_labels[dname] = row["_labels"]
    cross_view = {
        a: {b: float(adjusted_rand_score(best_labels[a], best_labels[b]))
            for b in DESCRIPTORS}
        for a in DESCRIPTORS}

    # actionability for each descriptor's best config --------------------------
    actionability = {}
    for dname in DESCRIPTORS:
        labels = best_labels[dname]
        act = {"best_method": best_per_descriptor[dname]["method"],
               "k": best_per_descriptor[dname]["k"],
               "shares": {str(k_): v / n_cond for k_, v in
                          sorted(((int(u), int((labels == u).sum()))
                                  for u in np.unique(labels)))},
               "min_share": float(min((labels == u).mean()
                                      for u in np.unique(labels))),
               "cards": nameability_cards(labels, features_df, models, emp_build)}
        act.update(pool_actionability(spec_store[dname], Z_store[dname], labels,
                                      pool_df, models))
        actionability[dname] = act

    for row in per_config:
        row.pop("_labels", None)

    results = {
        "meta": meta,
        "per_config": per_config,
        "best_per_descriptor": best_per_descriptor,
        "cross_view_ari": cross_view,
        "actionability": actionability,
        "descriptor_specs": {d: spec_store[d].to_dict() for d in DESCRIPTORS},
    }

    (out_dir / "results.json").write_text(
        json.dumps(_jsonify(results), indent=2, sort_keys=False))
    (out_dir / "report.md").write_text(render_report(results))
    return results


def _jsonify(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _jsonify(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# =============================================================================
# reporting
# =============================================================================

def _md_table(df: pd.DataFrame, index: bool = False) -> str:
    """Plain-python markdown table (pandas.to_markdown needs tabulate, which
    is not in the robocasa env -- keep this module dependency-free)."""
    if index:
        df = df.reset_index()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.tolist()) + " |")
    return "\n".join(lines)


def _fmt(x, nd=3):
    if x is None:
        return "-"
    try:
        if not np.isfinite(x):
            return "-"
    except TypeError:
        return str(x)
    return f"{x:.{nd}f}"


def headline_table(results: dict) -> pd.DataFrame:
    rows = []
    for r in results["per_config"]:
        best = results["best_per_descriptor"][r["descriptor"]]
        triv = r["triviality_ari"]
        finite_triv = {k: v for k, v in triv.items()
                       if v is not None and np.isfinite(v)}
        max_triv = (max(finite_triv, key=finite_triv.get)
                    if finite_triv else "-")
        rows.append({
            "descriptor": r["descriptor"], "method": r["method"], "k": r["k"],
            "noise%": f"{100 * r['noise_frac']:.0f}",
            "sil": _fmt(r["internal"]["silhouette"]),
            "DB": _fmt(r["internal"]["davies_bouldin"], 2),
            "CH": _fmt(r["internal"]["calinski_harabasz"], 0),
            "bootARI": f"{_fmt(r['stability']['bootstrap_ari_mean'])}"
                       f"±{_fmt(r['stability']['bootstrap_ari_sd'], 2)}",
            "seedARI": (_fmt(r["stability"]["seed_ari_mean"])
                        if r["stability"]["seed_ari_mean"] is not None else "-"),
            "eta2": _fmt(r["external"]["eta_squared"]),
            "F": _fmt(r["external"]["F"], 1),
            "p": _fmt(r["external"]["p"], 4),
            "JS": _fmt(r["external"]["js_within"]),
            "JSbase": _fmt(r["external"]["js_baseline_mean"]),
            "JSp": _fmt(r["external"]["js_perm_p"], 3),
            "maxTrivARI": (f"{_fmt(finite_triv.get(max_triv))} ({max_triv})"
                           if finite_triv else "-"),
            "best": "*" if (r["method"] == best["method"]) else "",
        })
    return pd.DataFrame(rows)


def render_report(results: dict) -> str:
    meta = results["meta"]
    lines = []
    label = meta.get("preview_label")
    lines.append("# Cluster-validation study -- bandit_v1 arms step")
    if label:
        lines.append(f"\n**{label}**")
    lines.append(
        f"\nConditions: {meta['n_conditions']} (of {meta['n_conditions_planned']}"
        f" planned) | build rows (repeats 0-3): {meta['n_build_rows']} | held-out"
        f" rows (repeats 4-7): {meta['n_val_rows']} | conditions in external"
        f" metrics: {meta['n_conditions_external']}")
    two = meta["map_comparison"]["two_model"]
    seq = meta["map_comparison"]["sequential"]
    lines.append(
        f"\nBuild-side map: family winner **{meta['map_family_winner']}** "
        f"(two_model log-loss {two['log_loss']:.4f} / AUC {two['auc']:.3f}; "
        f"sequential log-loss {seq['log_loss']:.4f} / AUC {seq['auc']:.3f}).")
    if not meta["hdbscan_available"]:
        lines.append(f"\n**NOTE**: {meta['hdbscan_fallback']}")

    df = headline_table(results)
    lines.append("\n## Headline comparison (best-k per method; * = per-descriptor"
                 " best config by held-out eta^2, ties by bootstrap ARI)\n")
    lines.append(_md_table(df))

    lines.append("\n## Cross-view ARI (best config per descriptor)\n")
    cv = pd.DataFrame(results["cross_view_ari"]).round(3)
    lines.append(_md_table(cv, index=True))

    lines.append("\n## Actionability (per-descriptor best config)\n")
    for dname, act in results["actionability"].items():
        lines.append(f"### {dname} ({act['best_method']}, k={act['k']})\n")
        lines.append(f"- min cluster share: {act['min_share']:.3f}")
        if act.get("pool_deployable"):
            lines.append(f"- well counts: {act['well_counts']} (|W|={act['n_well']})")
            if act.get("B") is not None:
                lines.append(f"- B rule: B={act['B']} (limiting arm "
                             f"{act['limiting_arm']})")
            else:
                lines.append(f"- B rule FAILED: {act.get('error')}")
        else:
            lines.append(f"- NOT pool-deployable: {act.get('note')}")
        for c in act["cards"]:
            cats = ", ".join(f"{t['category']}({t['count']})"
                             for t in c["top_categories"])
            knobs = ", ".join(f"{k_}={v:.2f}" for k_, v in
                              c["centroid_knobs"].items())
            lines.append(
                f"  - c{c['cluster']}: n={c['size']} share={c['share']:.2f} "
                f"stage={c['dominant_stage']} p_hat={c['mean_p_hat']:.2f} "
                f"emp_succ={c['mean_build_succ']:.2f} | {cats} | {knobs}")
        lines.append("")
    return "\n".join(lines) + "\n"


# =============================================================================
# CLI
# =============================================================================

def _main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    warnings.filterwarnings("ignore", category=UserWarning,
                            module="sklearn.mixture")

    from . import config  # local import: only the CLI needs default paths
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes-parquet",
                    default=str(config.LEDGER_DIR / "episodes.parquet"),
                    help="episodes table (pass a SNAPSHOT copy while a run is "
                         "live -- this module only reads, never writes, ledger "
                         "paths)")
    ap.add_argument("--diag-conditions-parquet",
                    default=str(config.LEDGER_DIR / "diag_conditions.parquet"))
    ap.add_argument("--pool-parquet", default=None,
                    help="pool table parquet (default: pool.build_pool_table("
                         "write=False), which does not write anything)")
    ap.add_argument("--out-dir", required=True,
                    help="output directory (results.json, report.md, "
                         "map_models_build.joblib) -- must NOT be ledger/")
    ap.add_argument("--preview-label", default=None,
                    help='e.g. "PREVIEW -- 74%% of diag data" for a mid-run '
                         "snapshot study")
    args = ap.parse_args()

    results = run_study(args.episodes_parquet, args.diag_conditions_parquet,
                        args.out_dir, pool_parquet=args.pool_parquet,
                        preview_label=args.preview_label)

    print(headline_table(results).to_string(index=False))
    print(f"\ncluster_study: wrote {Path(args.out_dir) / 'results.json'} and "
          f"report.md")
    print("!!! STOP FOR REVIEW: this study never freezes arms -- inspect the "
          "report, then run bandit_v1.clustering as usual once the descriptor/"
          "method choice is signed off.")


if __name__ == "__main__":
    _main()
