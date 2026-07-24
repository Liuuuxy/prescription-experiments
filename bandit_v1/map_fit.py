"""Difficulty map p_hat_0 + stage model p_stage (bandit_v1 Task 8).

Design: weakregion/BANDIT_V1_DESIGN.md section 2 items 3-4. Brief:
.superpowers/sdd/task-8-brief.md.

Fits two models on a diagnosis-ledger slice (`ledger/episodes.parquet` rows
with `phase == "diag"`, but this module takes a plain DataFrame and doesn't
care where it came from):
  - `p_success` (p_hat_0(x)): L2 logistic regression, P(success | x).
  - `p_stage` (p_stage(x)): multinomial logistic regression over the 5
    canonical failure stages (`STAGES`), same features.
Both live inside sklearn `Pipeline`s (StandardScaler -> LogisticRegression),
bundled with the fitted per-category shrinkage encoding into a `MapModels`
dataclass, joblib-persisted at `config.MAP_MODELS_JOBLIB`.

Commensurability requirement (design section 2 item 4 -- "condition-space
and pool-demo-space embeddings are commensurable"): pool_demos.parquet rows
(bandit_v1/pool.py) carry only `episode_index, category, h, w, layout, x_rel,
y_rel, side, traj_len, in_d0` -- no `yaw`, no `style_id`, and its `layout`
column is a different quantity than the diagnosis ledger's `layout_id` (never
joined here). So the feature set this module fits on and predicts from uses
ONLY `[h, w, x_rel, y_rel, side, category]` (RAW_FEATURE_COLUMNS below).
`yaw`, `layout_id`, `style_id` ARE logged in the diagnosis ledger (per the
Task 8 interface) and remain available for other analyses, but they are
NEVER read by `_build_feature_matrix`/`predict_features` -- so a pool-demo
row and a diagnosis/eval condition row embed into the exact same feature
space, which is the whole point: Task 9/10's clustering and retrieval-time
membership scoring evaluate `p_hat_0`/`p_stage` on pool demos using this same
code path.

Category canonicalization: every input DataFrame's `category` column must
already be canonical (`bandit_v1.categories.canonical_category(c) == c` for
every value) -- `_assert_canonical` enforces this on every entry point
(`fit`, `MapModels.predict_features`, `validation_report`) rather than
silently canonicalizing, since callers (pool.py, states.py, diagnosis.py) are
all already expected to canonicalize upstream per the task-3 convention; a
non-canonical value reaching here means some upstream join is wrong.

sklearn version note (verified against the installed scikit-learn==1.9.0):
`LogisticRegression`'s `penalty=`/`multi_class=` kwargs are gone/deprecated
in this version (`penalty` deprecated since 1.8, removed in 1.10; the class
now expresses the l1<->l2 mix via `l1_ratio`, defaulting to `l1_ratio=0.0` ==
pure L2 -- exactly the brief's "L2, C=1.0"). `multi_class=` no longer exists
either: since 1.5ish, every solver but 'liblinear' (we use the default
'lbfgs') fits the full multinomial loss automatically once y has >=3
classes. So `_make_estimator` below passes only `C=1.0` and relies on those
defaults -- functionally identical to the brief's literal
"LogisticRegression(C=1.0, l2)" / "multinomial LogisticRegression", without
tripping a deprecation warning that would (per the same docs) become a hard
error in sklearn 1.10.

Standardization scope: ALL seven engineered columns (h, w, x_rel, y_rel,
side_enc, h_sq, cat_te) are standardized together by ONE StandardScaler
inside each Pipeline, including h_sq and cat_te -- not just the five raw
knobs the brief lists before "+ h**2 quadratic + cat_te". L2-penalized
coefficients are scale-sensitive; leaving h_sq or cat_te unstandardized
would arbitrarily over/under-penalize those two terms relative to the rest
for no principled reason.

Held-out validation and target-encoding leakage: `validation_report` does
NOT evaluate `models` (which is fit on ALL of df_diag) against the same
df_diag -- that would report in-sample fit quality, and `cat_te` computed
from the full data would leak each held-out row's own label into its own
category's shrinkage encoding (categories repeat across many conditions/
folds). Instead it refits fresh, per GroupKFold fold (`_oof_predict` calls
`_fit_core` on the training rows only, including a fold-local cat_te), and
aggregates out-of-fold predictions -- an honest held-out estimate. `models`
is still a required argument: it supplies `metadata["cat_te_k"]` (the k the
production model was fit with) and its own `global_mean` is echoed into the
report, so the two calls (`fit` and `validation_report`) stay obviously
paired without re-deriving k from a second constant.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import categories, config, ledger

# --- canonical constants ----------------------------------------------------

CAT_TE_K = 8  # shrinkage constant k in (k*global + n*rate)/(k+n), per the brief

# bandit_v1 Task 9's "investigate-before-proceeding" gate (task-8-brief.md Step
# 3's own sanity note: "If AUC < 0.55 investigate before proceeding (wrong
# join, leaked constants)"). run_baseline.sh's map-fit step refuses to build E
# / run the baseline eval at all if the REAL diag-batch fit doesn't clear this
# -- see `_main` below and run_baseline.sh.
AUC_GATE_MIN = 0.55

# 5-stage failure signature, in rollout.py's `_failure_stage` return order.
# `predict_stage` always returns columns in exactly this order, zero-filling
# any stage never observed in the fitted model's training data.
STAGES = (
    "success",
    "never_reached",
    "reached_no_grasp",
    "fail_grasped_no_transport",
    "fail_reached_sink_no_place",
)
_STAGE_INDEX = {s: i for i, s in enumerate(STAGES)}

# sklearn's log_loss (verified against scikit-learn==1.9.0's docstring/runtime
# warning) ALWAYS assumes the y_proba columns it's given are in ALPHABETICAL
# label order -- regardless of what order the `labels=` kwarg itself lists
# them in; passing `labels` out of alphabetical order only silences the
# warning, it does NOT reorder the columns for you, so mismatched-order
# columns would silently compute the WRONG loss (columns bound to the wrong
# class). STAGES's own order (success-first, roughly earliest->latest
# failure) is deliberately readable and is what predict_stage returns, so it
# is kept as-is everywhere else; only the stage log_loss call below reorders
# into _STAGES_SORTED first.
_STAGES_SORTED = tuple(sorted(STAGES))
_STAGE_SORT_PERM = [STAGES.index(s) for s in _STAGES_SORTED]

# The ONLY raw columns _build_feature_matrix ever reads -- see module
# docstring's commensurability note. Any caller (fit, predict_features,
# validation_report) needs nothing else on its input df.
RAW_FEATURE_COLUMNS = ["category", "h", "w", "x_rel", "y_rel", "side"]

# Engineered numeric columns fed into each Pipeline's StandardScaler, in this
# fixed order (also recorded in MapModels.metadata["feature_columns"]).
FEATURE_COLUMNS = ["h", "w", "x_rel", "y_rel", "side_enc", "h_sq", "cat_te"]

DEFAULT_N_SPLITS = 5      # GroupKFold folds for validation_report
CALIBRATION_N_BINS = 10   # equal-count calibration bins


# --- category canonicalization guard ----------------------------------------

def _assert_canonical(df: pd.DataFrame) -> None:
    """Raise AssertionError if any `df["category"]` value is not already its
    own canonical form (bandit_v1.categories.canonical_category). Categories
    must be canonicalized upstream (pool.py / states.py already do); this is
    a defensive check, not a silent fix, so a wrong join surfaces immediately
    instead of quietly mixing alias/canonical rows in one category's stats."""
    canon = df["category"].map(categories.canonical_category)
    bad = sorted(df.loc[canon != df["category"], "category"].unique())
    assert not bad, (
        f"non-canonical category values found (canonicalize upstream via "
        f"bandit_v1.categories.canonical_category before calling map_fit): {bad}"
    )


def _to_dataframe(df_or_records) -> pd.DataFrame:
    """Accept a DataFrame, a single Series/dict-like record, or an iterable
    of records (list[dict] / list[Series]) -- the "pool-table rows" the
    module docstring's `predict_features` convenience promises to accept."""
    if isinstance(df_or_records, pd.DataFrame):
        return df_or_records
    if isinstance(df_or_records, (pd.Series, Mapping)):
        return pd.DataFrame([df_or_records])
    return pd.DataFrame(list(df_or_records))


# --- per-category shrinkage target encoding ---------------------------------

def category_target_encoding(df: pd.DataFrame, k: int = CAT_TE_K):
    """cat_te = per-category diag success rate shrunk toward the global mean:
    (k*global_mean + n*rate) / (k+n). Returns ({category: cat_te}, global_mean).

    A category with n=1 gets weight 1/(k+1) on its own (possibly extreme,
    single-condition) rate and k/(k+1) on global_mean -- with k=8 that's
    8/9 ~= 0.889 on global_mean, i.e. it sits close to it (Step 1b's test).
    An unseen category at inference time (_build_feature_matrix) falls back
    to global_mean directly -- the n=0 limit of this same formula."""
    y = df["success"].astype(float)
    grp = df.assign(_cat_te_y=y).groupby("category")["_cat_te_y"].agg(n="count", rate="mean")
    global_mean = float(y.mean())
    te = (k * global_mean + grp["n"] * grp["rate"]) / (k + grp["n"])
    return te.to_dict(), global_mean


def _build_feature_matrix(df: pd.DataFrame, cat_encoding: dict, global_mean: float) -> pd.DataFrame:
    """Raw RAW_FEATURE_COLUMNS -> the FEATURE_COLUMNS numeric matrix consumed
    by both Pipelines' StandardScaler. `side_enc` is just a float cast of
    `side` (already the canonical signed +-1 dominant-axis encoding, see
    bandit_v1/states.py's `_side` / pool.py's `side` column -- no further
    transform needed). `h_sq` is the brief's quadratic height term. `cat_te`
    looks each row's category up in `cat_encoding`, falling back to
    `global_mean` for any category absent from it (unseen at fit time)."""
    h = df["h"].astype(float)
    out = pd.DataFrame({
        "h": h.to_numpy(),
        "w": df["w"].astype(float).to_numpy(),
        "x_rel": df["x_rel"].astype(float).to_numpy(),
        "y_rel": df["y_rel"].astype(float).to_numpy(),
        "side_enc": df["side"].astype(float).to_numpy(),
        "h_sq": (h ** 2).to_numpy(),
        "cat_te": df["category"].map(lambda c: cat_encoding.get(c, global_mean)).astype(float).to_numpy(),
    })
    return out[FEATURE_COLUMNS]


# --- estimator construction (degenerate-fold safety) ------------------------

def _make_estimator(y):
    """LogisticRegression(C=1.0) relying on sklearn's current default L2
    penalty (l1_ratio=0.0) and lbfgs's automatic multinomial fit for >=3
    classes (see module docstring for why no penalty=/multi_class= kwarg is
    passed). Falls back to DummyClassifier(strategy="prior") when `y` has
    fewer than 2 distinct classes -- LogisticRegression cannot fit a
    single-class target (raises), and this keeps every GroupKFold training
    fold (and any single-outcome synthetic df) producing a valid, correctly
    shaped predict_proba instead of an uncaught exception -- the
    zero-variance-target safety net the tests require."""
    if pd.unique(np.asarray(y)).size < 2:
        return DummyClassifier(strategy="prior")
    return LogisticRegression(C=1.0, max_iter=1000)


def _positive_class_proba(pipeline: Pipeline, X) -> np.ndarray:
    """P(success=1) column from `pipeline.predict_proba(X)`, robust to a
    degenerate binary fit whose `classes_` doesn't contain both 0 and 1 (the
    DummyClassifier fallback from `_make_estimator` on an all-success or
    all-fail training slice yields classes_ == [1] or [0] only, shape
    (n, 1) -- indexing column 1 directly, as a normal two-class
    predict_proba[:, 1] would, is an IndexError in that case). Returns a
    uniform 1.0/0.0 column when only one class was ever observed."""
    proba = pipeline.predict_proba(X)
    classes = list(pipeline.classes_)
    if 1 in classes:
        return proba[:, classes.index(1)]
    return np.zeros(proba.shape[0], dtype=float)


def _reindex_stage_proba(proba: np.ndarray, classes_seen) -> np.ndarray:
    """`proba` (n, len(classes_seen)) -> (n, len(STAGES)), scattering each
    seen class's column into its fixed STAGES position and zero-filling any
    stage never observed during that fit. Row sums are preserved (still 1,
    since predict_proba's own columns already sum to 1) -- Step 1c's test."""
    out = np.zeros((proba.shape[0], len(STAGES)), dtype=float)
    for j, cls in enumerate(classes_seen):
        out[:, _STAGE_INDEX[cls]] = proba[:, j]
    return out


# --- MapModels ---------------------------------------------------------------

@dataclass
class MapModels:
    """Fitted difficulty map + stage model. `cat_encoding`/`metadata` are
    plain dict/JSON-ish (joblib-picklable alongside the two Pipelines)."""
    p_success: Pipeline
    p_stage: Pipeline
    cat_encoding: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def predict_features(self, df_or_records) -> pd.DataFrame:
        """Convenience used by Tasks 9/10 (design section 2 items 4-5):
        accepts pool-table rows OR full diagnosis/eval condition rows (a
        DataFrame, a single record, or an iterable of records) and returns
        the FEATURE_COLUMNS numeric matrix, using ONLY
        [h, w, x_rel, y_rel, side, category] -- see module docstring's
        commensurability note. yaw/layout_id/style_id, if present on the
        input, are ignored, not read."""
        df = _to_dataframe(df_or_records)
        _assert_canonical(df)
        return _build_feature_matrix(df, self.cat_encoding, self.metadata["global_mean"])

    def predict_p(self, df_or_records) -> np.ndarray:
        """p_hat_0(x): P(success | x), shape (n,)."""
        X = self.predict_features(df_or_records)
        return _positive_class_proba(self.p_success, X)

    def predict_stage(self, df_or_records) -> np.ndarray:
        """p_stage(x): P(stage | x) over STAGES, shape (n, 5); every row sums
        to 1."""
        X = self.predict_features(df_or_records)
        proba = self.p_stage.predict_proba(X)
        return _reindex_stage_proba(proba, self.p_stage.classes_)


def _fit_core(df: pd.DataFrame, k: int = CAT_TE_K) -> MapModels:
    """Shared fit path for both `fit()` (all of df_diag, the production
    model) and `_oof_predict()` (one GroupKFold training fold at a time) --
    a single source of truth so a fold-local fit and the final production
    fit can never silently diverge in how cat_te/features/estimators are
    built."""
    _assert_canonical(df)
    cat_encoding, global_mean = category_target_encoding(df, k=k)
    X = _build_feature_matrix(df, cat_encoding, global_mean)

    y_success = df["success"].astype(int).to_numpy()
    p_success = Pipeline([("scale", StandardScaler()), ("clf", _make_estimator(y_success))])
    p_success.fit(X, y_success)

    y_stage = df["failure_stage"].astype(str).to_numpy()
    p_stage = Pipeline([("scale", StandardScaler()), ("clf", _make_estimator(y_stage))])
    p_stage.fit(X, y_stage)

    metadata = {
        "global_mean": global_mean,
        "n_rows": int(len(df)),
        "n_categories": int(df["category"].nunique()),
        "feature_columns": list(FEATURE_COLUMNS),
        "raw_feature_columns": list(RAW_FEATURE_COLUMNS),
        "stages": list(STAGES),
        "cat_te_k": k,
        "success_classes_seen": list(p_success.classes_),
        "stage_classes_seen": list(p_stage.classes_),
    }
    return MapModels(p_success=p_success, p_stage=p_stage, cat_encoding=cat_encoding, metadata=metadata)


def fit(df_diag: pd.DataFrame) -> MapModels:
    """Fit p_hat_0 (p_success) + p_stage on ALL of `df_diag` -- the
    production model used for inference (Tasks 9/10). For an honest
    held-out accuracy estimate, call `validation_report(fit(df_diag),
    df_diag)` instead of evaluating this fit against df_diag directly (see
    module docstring)."""
    return _fit_core(df_diag)


# --- persistence -------------------------------------------------------------

def save(models: MapModels, path=None) -> Path:
    """joblib-dump `models` to `path` (default config.MAP_MODELS_JOBLIB),
    atomically (write-tmp-then-replace, matching ledger.py/pool.py's
    convention)."""
    path = Path(config.MAP_MODELS_JOBLIB if path is None else path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.joblib")
    joblib.dump(models, tmp)
    tmp.replace(path)
    return path


def load(path=None) -> MapModels:
    """Inverse of `save`."""
    path = Path(config.MAP_MODELS_JOBLIB if path is None else path)
    return joblib.load(path)


# --- GroupKFold validation ---------------------------------------------------

def _group_kfold_splits(df: pd.DataFrame, n_splits: int):
    """GroupKFold(by df["start_id"]) split indices -- repeats of one
    condition (same start_id, different repeat_idx) always land together in
    either train or val, never straddling a fold. `n_splits` is capped to
    [2, n_groups] (GroupKFold requires n_splits <= n_groups; 2 is the
    smallest a "held-out" split can mean). Returns (splits, n_splits_eff)."""
    groups = df["start_id"].to_numpy()
    n_groups = pd.unique(groups).size
    n_splits_eff = max(2, min(n_splits, n_groups))
    gkf = GroupKFold(n_splits=n_splits_eff)
    return list(gkf.split(df, groups=groups)), n_splits_eff


def _oof_predict(df: pd.DataFrame, n_splits: int = DEFAULT_N_SPLITS, k: int = CAT_TE_K):
    """Fold-local out-of-fold predictions: for each GroupKFold(by start_id)
    fold, refit (`_fit_core`, including a fold-LOCAL cat_te) on the training
    rows only and predict on the held-out rows -- never letting a held-out
    row's own label reach its category's shrinkage encoding. Returns
    (oof_p, oof_stage_proba, fold_id, n_splits_eff)."""
    splits, n_splits_eff = _group_kfold_splits(df, n_splits)
    oof_p = np.full(len(df), np.nan)
    oof_stage = np.full((len(df), len(STAGES)), np.nan)
    fold_id = np.full(len(df), -1, dtype=int)
    for fi, (tr, va) in enumerate(splits):
        fold_models = _fit_core(df.iloc[tr], k=k)
        oof_p[va] = fold_models.predict_p(df.iloc[va])
        oof_stage[va] = fold_models.predict_stage(df.iloc[va])
        fold_id[va] = fi
    return oof_p, oof_stage, fold_id, n_splits_eff


def _calibration_bins(p: np.ndarray, y: np.ndarray, n_bins: int = CALIBRATION_N_BINS):
    """10 (default) equal-count bins by predicted probability (np.array_split
    over the sort order, same convention as diagnosis.build_tercile_map):
    [{"n", "mean_predicted", "empirical"}, ...] ascending by predicted prob."""
    order = np.argsort(p, kind="mergesort")  # stable -> deterministic
    bins = []
    for g in np.array_split(order, n_bins):
        if len(g) == 0:
            continue
        bins.append({
            "n": int(len(g)),
            "mean_predicted": float(p[g].mean()),
            "empirical": float(y[g].mean()),
        })
    return bins


def validation_report(models: MapModels, df: pd.DataFrame, n_splits: int = DEFAULT_N_SPLITS) -> dict:
    """Held-out validation report: GroupKFold(by start_id) out-of-fold
    log-loss/AUC/calibration for p_success, plus out-of-fold multiclass
    log-loss for p_stage. Refits fresh per fold (see _oof_predict) rather
    than evaluating `models` (fit on all of `df`) against `df` itself --
    `models` supplies the k it was fit with (metadata["cat_te_k"]) and its
    own global_mean, echoed into the report for pairing/sanity."""
    _assert_canonical(df)
    k = models.metadata.get("cat_te_k", CAT_TE_K)
    oof_p, oof_stage, fold_id, n_splits_eff = _oof_predict(df, n_splits=n_splits, k=k)
    valid = fold_id >= 0

    y = df["success"].astype(int).to_numpy()[valid]
    p = oof_p[valid]
    ll = float(log_loss(y, p, labels=[0, 1]))
    auc = float(roc_auc_score(y, p)) if pd.unique(y).size >= 2 else float("nan")
    calib = _calibration_bins(p, y.astype(float), CALIBRATION_N_BINS)

    y_stage = df["failure_stage"].astype(str).to_numpy()[valid]
    stage_ll = float(log_loss(y_stage, oof_stage[valid][:, _STAGE_SORT_PERM],
                               labels=list(_STAGES_SORTED)))

    return {
        "n": int(valid.sum()),
        "n_groups": int(pd.unique(df["start_id"]).size),
        "n_splits": n_splits_eff,
        "log_loss": ll,
        "auc": auc,
        "calibration_bins": calib,
        "stage_log_loss": stage_ll,
        "global_mean": models.metadata.get("global_mean"),
    }


# --- CLI: fit on the real diag ledger slice (bandit_v1 Task 9's baseline ----
# orchestrator invokes this; see run_baseline.sh) -----------------------------

def _main():
    import argparse
    import json
    import sys

    # Line-buffer stdout regardless of invocation (nohup'd to a file etc.) --
    # same fix as diagnosis.py/run_diagnosis.py's _main.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", default="diag",
                     help="ledger episodes.parquet phase to fit on (default: diag, "
                          "the 2,400-row diagnosis batch)")
    ap.add_argument("--out", default=None,
                     help="joblib save path (default: config.MAP_MODELS_JOBLIB)")
    ap.add_argument("--report_out", default=None,
                     help="validation report json path (default: "
                          "config.LEDGER_DIR/map_validation_report.json)")
    args = ap.parse_args()

    df = ledger.read("episodes")
    df = df[df["phase"] == args.phase].reset_index(drop=True)
    print(f"map_fit: fitting on {len(df)} phase={args.phase!r} rows "
          f"({df['start_id'].nunique() if len(df) else 0} distinct start_ids)")
    if len(df) == 0:
        print(f"!!! map_fit: zero rows with phase={args.phase!r} in ledger episodes.parquet "
              f"-- nothing to fit. Refusing to proceed.")
        sys.exit(1)

    models = fit(df)
    report = validation_report(models, df)
    print("VALIDATION_REPORT", json.dumps(report, indent=2))

    report_path = (Path(args.report_out) if args.report_out is not None
                   else config.LEDGER_DIR / "map_validation_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"map_fit: wrote validation report to {report_path}")

    # Investigate-before-proceeding gate (task-8-brief.md Step 3 / this
    # module's AUC_GATE_MIN docstring note). `not (auc >= AUC_GATE_MIN)` is
    # deliberately used instead of `auc < AUC_GATE_MIN` so a NaN AUC (e.g. a
    # degenerate all-success/all-fail diag slice with zero label variance --
    # see validation_report's `pd.unique(y).size >= 2` guard) ALSO fails the
    # gate loudly, rather than silently comparing False against a NaN.
    if not (report["auc"] >= AUC_GATE_MIN):
        print(f"!!! GATE FAILED: held-out AUC {report['auc']} < {AUC_GATE_MIN} -- "
              f"investigate before proceeding (wrong join? leaked constants? see "
              f"task-8-brief.md Step 3). Refusing to save a model to "
              f"{args.out or config.MAP_MODELS_JOBLIB} -- run_baseline.sh must NOT "
              f"build E or run the baseline eval against this fit.")
        sys.exit(1)

    path = save(models, path=args.out)
    print(f"map_fit: gate passed (AUC={report['auc']:.4f} >= {AUC_GATE_MIN}) -- saved {path}")


if __name__ == "__main__":
    _main()
