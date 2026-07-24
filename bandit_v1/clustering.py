"""Clustering diag conditions into arms + arms.yaml (bandit_v1 Task 10).

Design: weakregion/BANDIT_V1_DESIGN.md section 2 items 4-6 ("Cluster into arms" /
"Per-arm samplers" / "Freeze") and section 0 item 2 (the B rule -- consumed here
via `wells.choose_B`, see that module). Brief: .superpowers/sdd/task-10-brief.md.

This module ONLY builds the MODULE + its synthetic tests + the run-CLI. The
REAL clustering run (against the real diag batch + fitted map_models.joblib)
is DEFERRED until the diagnosis batch actually finishes -- see
task-10-report.md. Every function here is exercised by synthetic data in
tests/test_clustering.py; nothing here has been run against real data yet.

--- z-block standardization scheme (design item 4: "z = [standardized
(h, w, x_rel, y_rel, side), p_hat_0(x), p_stage(x) (5-dim)]") ---

`ZSpec` freezes three independently-scaled blocks, fit ONCE on the diag
conditions (`fit_z_spec`) and applied UNCHANGED to any other feature table
(`transform_z`) -- notably the pool's W rows, so the diag-condition-space and
pool-demo-space embeddings stay in the SAME coordinate system (the whole point
of map_fit's commensurability contract, see map_fit.py's module docstring):

  1. **Knob block** (h, w, x_rel, y_rel, side): standardized PER-COLUMN, i.e.
     an ordinary StandardScaler over these 5 raw columns (mean 0, std 1 each,
     fit on the diag conditions). Each of the 5 columns therefore contributes
     exactly 1 unit of variance to the block -- 5 units total.
  2. **p_hat block** (`models.predict_p(df)`, 1-dim): centered (subtract its
     diag-fit mean), then divided by ONE scalar chosen so the block's TOTAL
     variance (== its only column's variance, since it is 1-dim) is 1. This is
     numerically identical to per-column standardizing a single column, but is
     described separately because item 3 below generalizes the same "total
     variance of the block, not of each column" rule to >1 dimension.
  3. **p_stage block** (`models.predict_stage(df)`, 5-dim, one column per
     `map_fit.STAGES`): each column centered by its OWN diag-fit mean, then ALL
     5 columns divided by the SAME scalar, chosen so the SUM of the 5 columns'
     variances (after centering, before this division) becomes 1 once divided.
     This is deliberately NOT per-column standardization: `p_stage` is a
     probability distribution over 5 stages, and some stages may be rare (near
     -constant, hence near-zero-variance) for a given diag batch -- per-column
     standardizing would blow such a column up to unit variance too, injecting
     noise instead of signal into the clustering descriptor. Scaling the whole
     block by one number instead preserves each column's variance RELATIVE to
     the others, while still giving the p_stage block, as a whole, the same
     total-variance budget (1 unit) as the p_hat block -- both "model output"
     blocks sit on equal footing against each other, distinct from the knob
     block's per-column (5-unit) budget.

A zero-variance column/block (e.g. a synthetic single-category test diag set
with constant height) is left un-scaled (divided by 1) rather than producing
NaN/inf, same zero-variance-safety convention as draw.py's `_standardize`.

--- Naming hard stop (design item 4's "naming test") ---

`summarize`/`build_arms_entries` produce per-cluster cards and a MECHANICAL
`suggested_slug` (dominant stage + strongest distinguishing knob direction) --
suggestions only, never used as the frozen name. The CLI (`python -m
bandit_v1.clustering`) always computes the full pipeline, prints the cards +
silhouette table + well-count table + proposed B, and writes a DRAFT arms yaml
with placeholder names (`UNNAMED_<i>`) -- then exits nonzero, unconditionally,
telling the operator to re-run with `--names slug0,slug1,...` to finalize.
There is no code path that invents and freezes a name on its own.

`finalize` (the `--names` path) never recomputes the clustering from the
ledger again -- it reads the EXISTING draft file's frozen numbers (index,
centroid, cov_diag, share, dominant_stage, z_spec, map_hash) and only swaps in
the supplied names, so a finalize can never silently diverge from the draft
the operator reviewed (e.g. if the ledger changed in between). `frozen_at` is
carried the same way: read from the draft file's own content by default (never
a hidden `datetime.now()` at finalize time), unless `--frozen-at` is passed
explicitly.
"""
import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from . import config, ledger, map_fit, pool

# The 5 raw knobs the design's descriptor standardizes per-column (design item
# 4). `side` is the canonical int {-1, +1} encoding (pool.py/states.py) --
# treated as an ordinary numeric column here, same as draw.py's `_standardize`.
KNOB_COLS = ("h", "w", "x_rel", "y_rel", "side")

STAGES = map_fit.STAGES

# Silhouette subsampling (brief: "silhouette on a 2,000-row subsample if
# needed" -- seeded for determinism, one fixed subsample reused across every
# k in the sweep so scores are comparable to each other).
SILHOUETTE_MAX_N = 2000
SILHOUETTE_SEED = 0

KMEANS_N_INIT = 50
KMEANS_RANDOM_STATE = 0

# Draft arms yaml -- an intermediate, unnamed artifact distinct from
# config.ARMS_YAML (the final, frozen, named file). Not itself a frozen
# constant (tests override the path), but this is the real-run default.
DRAFT_ARMS_YAML = config.LEDGER_DIR / "arms_draft.yaml"

# Slug format the naming hard-stop enforces: lowercase, starts with a letter,
# only [a-z0-9_] after that -- e.g. "reached_no_grasp_tall", "easy_short".
SLUG_RE = re.compile(r"^[a-z][a-z0-9_]*$")

# Mechanical direction words for `summarize`'s suggested_slug (design item 4:
# "dominant failure mode + region" -- "region" here read as "strongest
# distinguishing knob direction"). Suggestions only; never auto-frozen.
_KNOB_DIRECTION_WORDS = {
    "h": ("tall", "short"),
    "w": ("wide", "narrow"),
    "x_rel": ("offset_x_pos", "offset_x_neg"),
    "y_rel": ("offset_y_pos", "offset_y_neg"),
    "side": ("side_pos", "side_neg"),
}


# =============================================================================
# z-spec + build_z
# =============================================================================

@dataclass
class ZSpec:
    """Frozen standardization parameters for the 3-block descriptor (see
    module docstring). `to_dict`/`from_dict` are the arms.yaml round-trip
    (plain JSON/YAML-safe types only)."""
    knob_cols: tuple
    knob_mean: np.ndarray
    knob_scale: np.ndarray
    p_hat_mean: float
    p_hat_scale: float
    p_stage_mean: np.ndarray
    p_stage_scale: float
    stages: tuple

    def to_dict(self) -> dict:
        return {
            "knob_cols": list(self.knob_cols),
            "knob_mean": [float(x) for x in self.knob_mean],
            "knob_scale": [float(x) for x in self.knob_scale],
            "p_hat_mean": float(self.p_hat_mean),
            "p_hat_scale": float(self.p_hat_scale),
            "p_stage_mean": [float(x) for x in self.p_stage_mean],
            "p_stage_scale": float(self.p_stage_scale),
            "stages": list(self.stages),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ZSpec":
        return cls(
            knob_cols=tuple(d["knob_cols"]),
            knob_mean=np.asarray(d["knob_mean"], dtype=float),
            knob_scale=np.asarray(d["knob_scale"], dtype=float),
            p_hat_mean=float(d["p_hat_mean"]),
            p_hat_scale=float(d["p_hat_scale"]),
            p_stage_mean=np.asarray(d["p_stage_mean"], dtype=float),
            p_stage_scale=float(d["p_stage_scale"]),
            stages=tuple(d["stages"]),
        )


def _safe_scale(x: np.ndarray) -> np.ndarray:
    """Zero-variance guard (see module docstring): a 0 divisor is replaced by
    1, matching draw.py's `_standardize` convention."""
    return np.where(x == 0, 1.0, x)


def fit_z_spec(features_df: pd.DataFrame, models: map_fit.MapModels) -> ZSpec:
    """Fit a fresh `ZSpec` on `features_df` (the diag conditions) -- see
    module docstring for the exact per-block scheme."""
    knob = features_df[list(KNOB_COLS)].to_numpy(dtype=float)
    knob_mean = knob.mean(axis=0)
    knob_scale = _safe_scale(knob.std(axis=0))

    p_hat = models.predict_p(features_df).astype(float)
    p_hat_mean = float(p_hat.mean())
    p_hat_var = float(np.mean((p_hat - p_hat_mean) ** 2))
    p_hat_scale = float(np.sqrt(p_hat_var)) if p_hat_var > 0 else 1.0

    p_stage = models.predict_stage(features_df).astype(float)
    p_stage_mean = p_stage.mean(axis=0)
    p_stage_total_var = float(np.mean((p_stage - p_stage_mean) ** 2, axis=0).sum())
    p_stage_scale = float(np.sqrt(p_stage_total_var)) if p_stage_total_var > 0 else 1.0

    return ZSpec(
        knob_cols=KNOB_COLS, knob_mean=knob_mean, knob_scale=knob_scale,
        p_hat_mean=p_hat_mean, p_hat_scale=p_hat_scale,
        p_stage_mean=p_stage_mean, p_stage_scale=p_stage_scale,
        stages=tuple(models.metadata.get("stages", STAGES)),
    )


def transform_z(features_df: pd.DataFrame, models: map_fit.MapModels, z_spec: ZSpec) -> np.ndarray:
    """Apply an ALREADY-FITTED `z_spec` to `features_df` -- pure transform, no
    refitting. This is the pool-row path (design item 5: "demo -> z(demo) ...
    evaluated from the fitted models ... -> nearest centroid"): calling this
    with the diag-fit `z_spec` on pool rows is what keeps diag-condition-space
    and pool-demo-space commensurable."""
    knob = features_df[list(z_spec.knob_cols)].to_numpy(dtype=float)
    knob_z = (knob - z_spec.knob_mean) / z_spec.knob_scale

    p_hat = models.predict_p(features_df).astype(float)
    p_hat_z = ((p_hat - z_spec.p_hat_mean) / z_spec.p_hat_scale).reshape(-1, 1)

    p_stage = models.predict_stage(features_df).astype(float)
    p_stage_z = (p_stage - z_spec.p_stage_mean) / z_spec.p_stage_scale

    return np.hstack([knob_z, p_hat_z, p_stage_z])


def build_z(features_df: pd.DataFrame, models: map_fit.MapModels, z_spec: ZSpec = None):
    """(Z, z_spec). `z_spec=None` FITS a fresh one on `features_df` (the
    diag-condition path); a supplied `z_spec` is applied UNCHANGED (the
    pool-row path) -- see module docstring. Always returns the pair (rather
    than a bare array) so a caller on the fit path can persist `z_spec`
    (arms.yaml's `z_spec` block) regardless of which path was taken."""
    if z_spec is None:
        z_spec = fit_z_spec(features_df, models)
    Z = transform_z(features_df, models, z_spec)
    return Z, z_spec


# =============================================================================
# choose_k / merge_small
# =============================================================================

def _renumber(labels: np.ndarray) -> np.ndarray:
    """Relabel to a dense 0..k-1 range, preserving the ascending order of the
    original label values (so merge_small's output has no gaps left by
    merged-away cluster ids)."""
    uniq = sorted(int(u) for u in np.unique(labels))
    remap = {u: i for i, u in enumerate(uniq)}
    return np.array([remap[int(v)] for v in labels], dtype=int)


def merge_small(labels: np.ndarray, Z: np.ndarray, frac: float = None) -> np.ndarray:
    """Reassign every member of a cluster holding < `frac` (default
    config.MIN_CLUSTER_FRAC) of all rows to its NEAREST (in `Z`-space
    Euclidean distance) surviving ("big") cluster's centroid -- design item 4:
    "merge clusters holding <5% of conditions". Centroids used for the
    nearest-cluster decision are computed ONCE, from the ORIGINAL labels,
    before any reassignment (so merging one small cluster never shifts where
    another small cluster's members land). Returns densely renumbered labels
    (`_renumber`) whether or not anything was merged, so callers get a stable
    0..k'-1 range either way.

    If EVERY cluster is below `frac` (a degenerate all-small labeling -- no
    "big" cluster exists to merge into), this is a no-op besides renumbering:
    there is nothing sane to reassign into.
    """
    frac = config.MIN_CLUSTER_FRAC if frac is None else frac
    labels = np.asarray(labels)
    n = len(labels)
    unique, counts = np.unique(labels, return_counts=True)
    fracs = counts / n
    small = set(unique[fracs < frac].tolist())

    if not small:
        return _renumber(labels)

    big = [int(u) for u in unique if int(u) not in small]
    if not big:
        return _renumber(labels)

    centroids = {int(u): Z[labels == u].mean(axis=0) for u in unique}
    big_centroids = np.stack([centroids[u] for u in big])

    new_labels = labels.copy()
    for u in small:
        mask = labels == u
        d = np.linalg.norm(Z[mask][:, None, :] - big_centroids[None, :, :], axis=2)
        nearest = np.asarray(big)[np.argmin(d, axis=1)]
        new_labels[mask] = nearest

    return _renumber(new_labels)


def choose_k(Z: np.ndarray, k_range=None, max_arms: int = None, min_cluster_frac: float = None,
             n_init: int = KMEANS_N_INIT, random_state: int = KMEANS_RANDOM_STATE,
             silhouette_max_n: int = SILHOUETTE_MAX_N, silhouette_seed: int = SILHOUETTE_SEED):
    """k-means over `k_range` (default config.K_RANGE), `n_init`/`random_state`
    fixed for determinism; pick the k (capped at `max_arms`-1, default
    config.MAX_ARMS-1 -- Random takes the remaining arm slot) with the best
    silhouette score, computed on ONE fixed seeded subsample of up to
    `silhouette_max_n` rows (reused across every candidate k so scores are
    comparable); then `merge_small` the winning k's labels.

    "Capped ... if best-silhouette k exceeds it, take the best k within the
    cap" (design item 4) is implemented by restricting the argmax to
    `k <= cap` directly -- equivalent to, and simpler than, computing the
    global argmax first and only falling back if it happens to exceed the cap.

    Returns (k, labels, silhouette_table): `k` is the FINAL cluster count
    after `merge_small` (may be less than the chosen pre-merge k); `labels`
    is the corresponding merged/renumbered array, row-aligned to `Z`;
    `silhouette_table` is a DataFrame (columns: k, silhouette, within_cap,
    chosen) over every candidate in `k_range`, for the CLI's printed report.
    """
    k_range = config.K_RANGE if k_range is None else k_range
    max_arms = config.MAX_ARMS if max_arms is None else max_arms
    min_cluster_frac = config.MIN_CLUSTER_FRAC if min_cluster_frac is None else min_cluster_frac
    cap = max_arms - 1

    n = Z.shape[0]
    rng = np.random.default_rng(silhouette_seed)
    sub_idx = (rng.choice(n, size=silhouette_max_n, replace=False)
               if n > silhouette_max_n else np.arange(n))

    fits = {}
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels_k = km.fit_predict(Z)
        fits[k] = labels_k

        sub_labels = labels_k[sub_idx]
        if len(np.unique(sub_labels)) < 2:
            score = float("nan")
        else:
            score = float(silhouette_score(Z[sub_idx], sub_labels))
        rows.append({"k": int(k), "silhouette": score, "within_cap": bool(k <= cap)})

    table = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)

    within = table[table["within_cap"] & table["silhouette"].notna()]
    if within.empty:
        raise ValueError(
            f"choose_k: no candidate k in {list(k_range)} with k <= cap({cap}) "
            f"produced a valid (>=2-label) silhouette score on the subsample")
    best_k = int(within.loc[within["silhouette"].idxmax(), "k"])
    table["chosen"] = table["k"] == best_k

    labels = fits[best_k]
    merged_labels = merge_small(labels, Z, frac=min_cluster_frac)
    final_k = int(len(np.unique(merged_labels)))
    return final_k, merged_labels, table


# =============================================================================
# summarize (per-cluster cards -- the human-naming input)
# =============================================================================

def _dominant_direction_word(centroid_knob_z: dict) -> str:
    """Strongest distinguishing knob direction: the KNOB_COLS entry with the
    largest |standardized centroid offset| from the population mean (ties
    broken by KNOB_COLS's fixed order, via argmax-on-first-occurrence), mapped
    to a human word via `_KNOB_DIRECTION_WORDS`."""
    best_col = max(KNOB_COLS, key=lambda c: abs(centroid_knob_z[c]))
    pos_word, neg_word = _KNOB_DIRECTION_WORDS[best_col]
    return pos_word if centroid_knob_z[best_col] >= 0 else neg_word


def summarize(labels, features_df: pd.DataFrame, models: map_fit.MapModels) -> list:
    """Per-cluster cards (design item 4's naming-test input): for every
    cluster label present in `labels` (ascending order), a dict with:
      - index, size, share
      - mean_p_hat: mean `models.predict_p` over the cluster's members
      - dominant_stage / stage_distribution: each member's OWN dominant
        predicted stage (argmax of `models.predict_stage`) is voted, then
        aggregated into a per-cluster histogram over `map_fit.STAGES` --
        "dominant predicted stage distribution" read as the distribution,
        within the cluster, of each member's own dominant predicted stage.
      - top_categories: the cluster's top-5 `category` value counts
      - centroid_knobs: cluster-mean of the RAW (unstandardized) KNOB_COLS
        values -- the human-readable numbers a namer actually looks at
      - suggested_slug: MECHANICAL "{dominant_stage}-{direction}" (see
        `_dominant_direction_word`) -- a suggestion only, never auto-frozen
        (see module docstring's naming hard-stop).
    """
    labels = np.asarray(labels)
    features_df = features_df.reset_index(drop=True)
    n = len(features_df)

    p_hat = models.predict_p(features_df).astype(float)
    p_stage = models.predict_stage(features_df).astype(float)
    stages = tuple(models.metadata.get("stages", STAGES))
    dominant_row_stage = np.asarray(stages)[np.argmax(p_stage, axis=1)]

    knob = features_df[list(KNOB_COLS)].to_numpy(dtype=float)
    pop_mean = knob.mean(axis=0)
    pop_scale = _safe_scale(knob.std(axis=0))

    cards = []
    for c in sorted(np.unique(labels).tolist()):
        mask = labels == c
        size = int(mask.sum())
        share = size / n
        mean_p_hat = float(p_hat[mask].mean())

        stage_counts = pd.Series(dominant_row_stage[mask]).value_counts()
        stage_dist = {s: float(stage_counts.get(s, 0)) / size for s in stages}
        dominant_stage = max(stage_dist, key=stage_dist.get)

        top_cats = features_df.loc[mask, "category"].value_counts().head(5)
        top_categories = [{"category": str(cat), "count": int(cnt)} for cat, cnt in top_cats.items()]

        knob_cluster_mean = knob[mask].mean(axis=0)
        centroid_knobs = {k: float(v) for k, v in zip(KNOB_COLS, knob_cluster_mean)}
        centroid_knob_z = {k: float(v) for k, v in
                           zip(KNOB_COLS, (knob_cluster_mean - pop_mean) / pop_scale)}
        suggested_slug = f"{dominant_stage}-{_dominant_direction_word(centroid_knob_z)}"

        cards.append({
            "index": int(c),
            "size": size,
            "share": share,
            "mean_p_hat": mean_p_hat,
            "dominant_stage": dominant_stage,
            "stage_distribution": stage_dist,
            "top_categories": top_categories,
            "centroid_knobs": centroid_knobs,
            "suggested_slug": suggested_slug,
        })
    return cards


# =============================================================================
# arms.yaml entries (draft + final share this shape; only `name` differs)
# =============================================================================

def build_arms_entries(labels, features_df: pd.DataFrame, models: map_fit.MapModels,
                        z_spec: ZSpec, Z: np.ndarray = None) -> list:
    """Per-cluster arms.yaml entries: `{name: "UNNAMED_<i>", index, centroid:
    {standardized, raw}, cov_diag, share, dominant_stage}`.

    `centroid.standardized` is the cluster's mean Z-space (11-dim) vector --
    what `wells.assign_regions`'s nearest-centroid rule compares pool rows
    against. `centroid.raw` / `cov_diag` are the cluster's mean/variance of
    the RAW KNOB_COLS values (a dict keyed by KNOB_COLS) -- the human-readable
    pair the design's "per-arm samplers: truncated Gaussian (mean/cov of
    member conditions, clipped to knob ranges)" (item 5) needs for its dormant
    backup channel: you can only clip a physical knob (h, w, x_rel, y_rel,
    side) to a "knob range", not a Z-space coordinate.
    """
    labels = np.asarray(labels)
    features_df = features_df.reset_index(drop=True)
    if Z is None:
        Z = transform_z(features_df, models, z_spec)

    knob = features_df[list(KNOB_COLS)].to_numpy(dtype=float)
    cards = {c["index"]: c for c in summarize(labels, features_df, models)}

    entries = []
    for c in sorted(np.unique(labels).tolist()):
        mask = labels == c
        z_rows = Z[mask]
        knob_rows = knob[mask]
        card = cards[int(c)]

        entries.append({
            "name": f"UNNAMED_{int(c)}",
            "index": int(c),
            "centroid": {
                "standardized": [float(x) for x in z_rows.mean(axis=0)],
                "raw": {k: float(v) for k, v in card["centroid_knobs"].items()},
            },
            "cov_diag": {k: float(v) for k, v in
                         zip(KNOB_COLS, knob_rows.var(axis=0, ddof=0))},
            "share": float(card["share"]),
            "dominant_stage": card["dominant_stage"],
        })
    return entries


# =============================================================================
# real-data pipeline (draft compute + write + finalize) -- Task 10's run-CLI
# =============================================================================

def _load_diag_features(ledger_dir=None) -> pd.DataFrame:
    """One row per diag start_id (deduplicated across repeat_idx -- the
    design's PER-CONDITION descriptor, not one row per rollout repeat), with
    columns [start_id, category, h, w, x_rel, y_rel, side], read from
    `ledger_dir/episodes.parquet`'s phase=="diag" rows."""
    ledger_dir = Path(config.LEDGER_DIR if ledger_dir is None else ledger_dir)
    episodes_path = ledger_dir / "episodes.parquet"
    df = pd.read_parquet(episodes_path)
    df = df[df["phase"] == "diag"].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(
            f"clustering: zero phase='diag' rows in {episodes_path} -- the diagnosis "
            f"batch must complete before clustering can run (this run is deferred "
            f"until then, see task-10-report.md).")

    needed = ["start_id", "category", *KNOB_COLS]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"clustering: diag episodes.parquet missing columns {missing}")

    df = df.drop_duplicates(subset="start_id", keep="first").reset_index(drop=True)
    return df[needed]


def compute_draft(ledger_dir=None, pool_parquet=None, map_models_path=None) -> dict:
    """Full Task-10 compute pipeline: load diag conditions + MapModels + the
    pool table, fit z_spec on the diag conditions, cluster (`choose_k`, which
    internally applies `merge_small`), build cluster cards + arms entries
    (placeholder names), apply the SAME frozen z_spec to the pool's W rows to
    get the well-count table and the proposed B. Returns everything the CLI
    needs to print and persist, so the DRAFT yaml and the printed report are
    always built from one single, internally-consistent computation.

    `ledger_dir`/`pool_parquet`/`map_models_path` are overrides purely for
    testability (subprocess-driven CLI tests in a tmp dir) -- default to the
    real config paths (`config.LEDGER_DIR`, `pool.build_pool_table(write=
    False)`, `config.MAP_MODELS_JOBLIB`) for the eventual real run.
    """
    ledger_dir = Path(config.LEDGER_DIR if ledger_dir is None else ledger_dir)
    map_models_path = Path(config.MAP_MODELS_JOBLIB if map_models_path is None else map_models_path)

    models = map_fit.load(map_models_path)
    diag_df = _load_diag_features(ledger_dir)

    Z, z_spec = build_z(diag_df, models)
    k, labels, silhouette_table = choose_k(Z)
    cards = summarize(labels, diag_df, models)
    arms_entries = build_arms_entries(labels, diag_df, models, z_spec, Z=Z)

    if pool_parquet is not None:
        pool_df = pd.read_parquet(pool_parquet)
    else:
        pool_df = pool.build_pool_table(write=False)

    from . import wells as _wells  # deferred: wells.py imports this module at
                                    # top level (for ZSpec/transform_z), so a
                                    # top-level import here would cycle back.
    arms_spec_for_wells = {"arms": arms_entries, "z_spec": z_spec.to_dict()}
    regions = _wells.assign_regions(pool_df, models, arms_spec_for_wells)
    well_tbl = _wells.well_table(regions)
    B, limiting_arm = _wells.choose_B(well_tbl)

    map_hash = ledger.file_hash(map_models_path)

    return {
        "k": k, "labels": labels, "silhouette_table": silhouette_table,
        "cards": cards, "arms_entries": arms_entries, "z_spec": z_spec,
        "well_table": well_tbl, "B": B, "limiting_arm": limiting_arm,
        "map_hash": map_hash, "diag_df": diag_df,
    }


def write_draft(result: dict, path=None, frozen_at: str = None) -> Path:
    """Write the DRAFT arms yaml (default `DRAFT_ARMS_YAML`): `result`'s
    `arms_entries` verbatim (placeholder `UNNAMED_<i>` names), plus
    `random_arm`, `frozen_at` (default: today's UTC date -- the ONE place this
    module ever calls `datetime.now()`, since a draft is genuinely being
    created fresh right now), `z_spec`, `map_hash`, and the proposed B /
    limiting arm for the record. `finalize` later reads this file back
    verbatim rather than recomputing, so everything a `--names` re-run needs
    must already be here."""
    path = Path(DRAFT_ARMS_YAML if path is None else path)
    frozen_at = frozen_at or datetime.now(timezone.utc).date().isoformat()
    draft = {
        "arms": result["arms_entries"],
        "random_arm": True,
        "frozen_at": frozen_at,
        "z_spec": result["z_spec"].to_dict(),
        "map_hash": result["map_hash"],
        "proposed_B": result["B"],
        "limiting_arm": result["limiting_arm"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.yaml")
    with open(tmp, "w") as f:
        yaml.safe_dump(draft, f, sort_keys=False, default_flow_style=False)
    tmp.replace(path)
    return path


def finalize(names: list, draft_path=None, out_path=None, frozen_at: str = None,
             hashes_path=None) -> Path:
    """Validate `names` against the existing DRAFT (`draft_path`, default
    `DRAFT_ARMS_YAML`) and write the FINAL `arms.yaml` (`out_path`, default
    `config.ARMS_YAML`), hashing it into `hashes_path` (default
    `out_path.parent/hashes.json`).

    Never recomputes clustering: every field besides `name` (and `frozen_at`,
    if `--frozen-at` overrides it) is copied verbatim from the draft, so
    finalize can never silently diverge from what the operator reviewed when
    the draft was printed. Raises (never silently proceeds) on: missing
    draft, wrong `names` count, duplicate names, or a name failing
    `SLUG_RE` -- "NEVER silently name" (module docstring).
    """
    draft_path = Path(DRAFT_ARMS_YAML if draft_path is None else draft_path)
    if not draft_path.exists():
        raise FileNotFoundError(
            f"clustering.finalize: draft {draft_path} not found -- run "
            f"`python -m bandit_v1.clustering` (no --names) first to compute "
            f"and write it.")
    draft = yaml.safe_load(draft_path.read_text())
    arms = sorted(draft["arms"], key=lambda a: a["index"])

    if len(names) != len(arms):
        raise ValueError(
            f"clustering.finalize: got {len(names)} names, need exactly "
            f"{len(arms)} (one per cluster; draft indices "
            f"{[a['index'] for a in arms]}) -- pass --names slug0,slug1,...")
    if len(set(names)) != len(names):
        raise ValueError(f"clustering.finalize: names must be unique, got {names}")
    bad = [nm for nm in names if not SLUG_RE.match(nm)]
    if bad:
        raise ValueError(
            f"clustering.finalize: invalid slug format {bad} -- names must match "
            f"{SLUG_RE.pattern!r} (lowercase, start with a letter, only [a-z0-9_])")

    named_arms = []
    for arm, name in zip(arms, names):
        new_arm = dict(arm)
        new_arm["name"] = name
        named_arms.append(new_arm)

    final = {
        "arms": named_arms,
        "random_arm": True,
        "frozen_at": frozen_at or draft["frozen_at"],
        "z_spec": draft["z_spec"],
        "map_hash": draft["map_hash"],
    }

    out_path = Path(config.ARMS_YAML if out_path is None else out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.yaml")
    with open(tmp, "w") as f:
        yaml.safe_dump(final, f, sort_keys=False, default_flow_style=False)
    tmp.replace(out_path)

    hashes_path = Path(out_path.parent / "hashes.json") if hashes_path is None else Path(hashes_path)
    hashes = json.load(open(hashes_path)) if hashes_path.exists() else {}
    hashes[out_path.name] = ledger.file_hash(out_path)
    tmp_h = hashes_path.with_suffix(".tmp.json")
    with open(tmp_h, "w") as f:
        json.dump(hashes, f, indent=2, sort_keys=True)
    tmp_h.replace(hashes_path)

    return out_path


def load_arms_yaml(path=None) -> dict:
    """Load a frozen arms yaml (default `config.ARMS_YAML`, read fresh at call
    time). Convenience for later tasks / manual inspection."""
    path = Path(config.ARMS_YAML if path is None else path)
    return yaml.safe_load(path.read_text())


# =============================================================================
# CLI
# =============================================================================

def _main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger-dir", default=None,
                     help="override config.LEDGER_DIR (testability)")
    ap.add_argument("--pool-parquet", default=None,
                     help="read the pool table from this parquet instead of "
                          "pool.build_pool_table() (testability)")
    ap.add_argument("--map-models-path", default=None,
                     help="override config.MAP_MODELS_JOBLIB (testability)")
    ap.add_argument("--draft-path", default=None,
                     help="override DRAFT_ARMS_YAML (testability)")
    ap.add_argument("--out-path", default=None,
                     help="override config.ARMS_YAML, the finalize output (testability)")
    ap.add_argument("--hashes-path", default=None,
                     help="override <out-path's dir>/hashes.json (testability)")
    ap.add_argument("--frozen-at", default=None,
                     help="ISO date to freeze into arms.yaml. Default: at "
                          "finalize, the draft file's own frozen_at (never a "
                          "hidden datetime.now()); on a fresh draft, today's "
                          "UTC date (a draft IS being created right now).")
    ap.add_argument("--names", default=None,
                     help="comma-separated cluster names, one per cluster in "
                          "index order (e.g. --names easy_short,tall_grasp). "
                          "Supplying this FINALIZES arms.yaml from the "
                          "existing draft; omitting it (re-)computes and "
                          "prints the draft, then exits nonzero (hard stop -- "
                          "clusters must be named by a human).")
    args = ap.parse_args()

    if args.names is not None:
        names = [nm.strip() for nm in args.names.split(",")]
        try:
            out_path = finalize(names, draft_path=args.draft_path, out_path=args.out_path,
                                 frozen_at=args.frozen_at, hashes_path=args.hashes_path)
        except (FileNotFoundError, ValueError) as e:
            print(f"!!! clustering finalize FAILED: {e}")
            sys.exit(1)
        print(f"clustering: wrote FINAL {out_path}")
        sys.exit(0)

    result = compute_draft(ledger_dir=args.ledger_dir, pool_parquet=args.pool_parquet,
                            map_models_path=args.map_models_path)

    print(f"clustering: chosen k={result['k']} clusters (+ Random)")
    print("SILHOUETTE_TABLE")
    print(result["silhouette_table"].to_string(index=False))
    print("CLUSTER_CARDS")
    for card in result["cards"]:
        print(json.dumps(card, indent=2, sort_keys=False))
    print("WELL_COUNT_TABLE")
    print(result["well_table"].to_string(index=False))
    print(f"PROPOSED_B {result['B']} (limiting arm: {result['limiting_arm']!r})")

    draft_path = write_draft(result, path=args.draft_path, frozen_at=args.frozen_at)
    names_preview = [a["name"] for a in result["arms_entries"]]
    print(f"clustering: wrote DRAFT {draft_path} with placeholder names {names_preview}")
    print(
        f"!!! HARD STOP: clusters are unnamed. Re-run with --names "
        f"slug0,slug1,...,slug{result['k'] - 1} (one per cluster, in index "
        f"order, matching {SLUG_RE.pattern!r}) to validate + finalize "
        f"arms.yaml. Names are NEVER invented automatically."
    )
    sys.exit(1)


if __name__ == "__main__":
    _main()
