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

Two model families, chosen by held-out validation (owner decision: the two
original models above disagree by construction and ignore the reach ->
grasp -> transport -> place progression the 5 STAGES actually encode):
  - "two_model" (the original design above): a single binary p_success fit
    plus a disconnected 5-class multinomial p_stage fit. Kept as-is, byte-
    identical, as the default `fit()`/`_fit_core` path -- no existing caller
    or test changes behavior.
  - "sequential" (`SequentialStageModel`): four binary L2-logistic
    "gates" (reach/grasp/transport/place), each fit on the SAME
    FEATURE_COLUMNS matrix (same `_build_feature_matrix`/`category_target_
    encoding` call, never forked) but restricted to the rows that survived
    the previous gate (`_gate_populations_and_labels`). `predict_p` is the
    plain product q1*q2*q3*q4 and `predict_stage` composes the same four
    gate probabilities into the 5 STAGES columns -- rows sum to 1 and
    P(success) equals `predict_p` BY CONSTRUCTION, so (unlike the two
    disconnected two_model fits) the two public predictions can never
    disagree.
`MapModels.family` selects which one `predict_p`/`predict_stage` route
through; `fit(df, family=...)` builds either (default "two_model", for
backward compatibility); `compare_families(df)` fits and cross-validates
BOTH on the same GroupKFold-by-start_id folds and picks a winner by held-out
p_success log-loss (see its docstring) -- this is what the CLI (`_main`)
actually runs before saving a production model.

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

# `SequentialStageModel`'s four binary gates, in reach -> grasp -> transport
# -> place progression order (MOTIVATION: each STAGES value is the first
# failed gate in this chain). `_GRASPED_STAGES`/`_TRANSPORTED_STAGES` are the
# STAGES subsets a row's `failure_stage` falls into once it has passed the
# grasp/transport gate respectively -- used by `_gate_populations_and_labels`
# to define each gate's training population (rows that passed the PREVIOUS
# gate) and label (did this row pass THIS gate) directly off the same
# `failure_stage` column the two_model family's p_stage fits on, so both
# families are provably scored against the same ground truth.
GATE_NAMES = ("reach", "grasp", "transport", "place")
_GRASPED_STAGES = frozenset({
    "fail_grasped_no_transport", "fail_reached_sink_no_place", "success",
})
_TRANSPORTED_STAGES = frozenset({"fail_reached_sink_no_place", "success"})

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


# --- continuation-ratio (sequential) stage model -----------------------------

def _gate_populations_and_labels(df: pd.DataFrame) -> dict:
    """{"reach"|"grasp"|"transport"|"place": (population_mask, label)}, both
    boolean ndarrays over ALL of `df` (len == len(df)):
      - `population_mask[i]` is True iff row i survived the PREVIOUS gate
        (reach's population is every row -- there is no "previous" gate);
      - `label[i]` is True iff row i passed THIS gate, meaningful only where
        `population_mask[i]` is True (rows outside a gate's population never
        got a chance to pass or fail it, so their label value is a
        don't-care and callers must always intersect with population_mask
        before using it -- see `_fit_sequential_core`/`validation_report`).
    Derived once from `df["failure_stage"]` so `_fit_sequential_core` (what
    a fold/production fit trains each gate on) and `validation_report`'s
    per-gate held-out AUC (what population/label a held-out row is scored
    against) can never define "reached"/"grasped"/"transported" two
    different ways."""
    stage = df["failure_stage"].astype(str).to_numpy()
    n = len(df)
    reached = stage != "never_reached"
    grasped = np.isin(stage, list(_GRASPED_STAGES))
    transported = np.isin(stage, list(_TRANSPORTED_STAGES))
    placed = stage == "success"
    return {
        "reach": (np.ones(n, dtype=bool), reached),
        "grasp": (reached, grasped),
        "transport": (grasped, transported),
        "place": (transported, placed),
    }


@dataclass
class _Gate:
    """One binary logistic gate inside a `SequentialStageModel`. `pipeline`
    is a fitted (StandardScaler -> `_make_estimator`) Pipeline in the normal
    case -- including the existing degenerate-single-class DummyClassifier
    safety net `_make_estimator` already gives the two_model family. It is
    None only in the stricter degenerate case `_make_estimator` can't cover
    at all: a gate whose training POPULATION is empty (zero rows survived
    the previous gate, e.g. no diag row ever grasped anything) -- sklearn's
    StandardScaler/LogisticRegression/DummyClassifier all require >=1
    training sample and raise on an empty X/y, so this is a distinct
    fallback, not just routed through `_make_estimator`. `constant_p` is the
    fixed P(pass) `positive_proba` then always returns; 0.5 (maximally
    uninformative) is used throughout -- harmless in practice since a
    downstream gate being reachable at all requires the upstream gates'
    q's to be nonzero, and an upstream gate whose population was THIS
    degenerate (nobody ever got here) has itself already predicted ~0 for
    every row (see `_fit_gate`'s zero-row check happening only after the
    previous gate's own fit already saw a single-class or empty y), so this
    constant gets multiplied by a near-zero upstream product in
    `predict_p`/`predict_stage` either way."""
    pipeline: object
    constant_p: float
    n_train: int
    classes_seen: list

    def positive_proba(self, X) -> np.ndarray:
        if self.pipeline is None:
            return np.full(len(X), self.constant_p, dtype=float)
        return _positive_class_proba(self.pipeline, X)


def _fit_gate(X_pop, y_pop: np.ndarray, fallback_p: float = 0.5) -> _Gate:
    """Fit one gate's Pipeline on its (already population-filtered) `X_pop`/
    `y_pop`. Zero rows -> the `_Gate(pipeline=None, ...)` constant fallback
    (see its docstring); otherwise a normal (StandardScaler ->
    `_make_estimator(y_pop)`) Pipeline, fit exactly like `_fit_core`'s
    p_success Pipeline (same estimator constructor, same degenerate-class
    handling) -- only the population it is fit on differs."""
    n = int(len(y_pop))
    if n == 0:
        return _Gate(pipeline=None, constant_p=fallback_p, n_train=0, classes_seen=[])
    pipeline = Pipeline([("scale", StandardScaler()), ("clf", _make_estimator(y_pop))])
    pipeline.fit(X_pop, y_pop)
    return _Gate(pipeline=pipeline, constant_p=fallback_p, n_train=n,
                 classes_seen=list(pipeline.classes_))


@dataclass
class SequentialStageModel:
    """Continuation-ratio stage model: four binary gates (`GATE_NAMES`)
    along the reach -> grasp -> transport -> place progression, each a
    Pipeline over the exact same FEATURE_COLUMNS matrix the two_model
    family's p_success/p_stage fit on, restricted to its own conditional
    population (`_gate_populations_and_labels`). `gates` maps GATE_NAMES ->
    `_Gate`. See module docstring for the compose-to-STAGES formula."""
    gates: dict

    def predict_gate_probs(self, X) -> dict:
        """{"reach": q1, "grasp": q2, "transport": q3, "place": q4}, each
        shape (n,) -- P(pass this gate | x). Every gate can be queried on
        ANY row's features regardless of that row's own true stage; only
        FITTING a gate was restricted to its conditional population."""
        return {name: self.gates[name].positive_proba(X) for name in GATE_NAMES}

    def predict_p(self, X) -> np.ndarray:
        """P(success | x) = q1*q2*q3*q4 -- the plain product of all four
        gates, shape (n,)."""
        q = self.predict_gate_probs(X)
        return q["reach"] * q["grasp"] * q["transport"] * q["place"]

    def predict_stage(self, X) -> np.ndarray:
        """P(stage | x) over STAGES, shape (n, 5); every row sums to
        exactly 1 and column STAGES.index("success") equals `predict_p`'s
        output exactly (both are the same q1*q2*q3*q4 product) -- by
        construction, not by a separate consistency check."""
        q = self.predict_gate_probs(X)
        q1, q2, q3, q4 = q["reach"], q["grasp"], q["transport"], q["place"]
        stage_vals = {
            "success": q1 * q2 * q3 * q4,
            "never_reached": 1.0 - q1,
            "reached_no_grasp": q1 * (1.0 - q2),
            "fail_grasped_no_transport": q1 * q2 * (1.0 - q3),
            "fail_reached_sink_no_place": q1 * q2 * q3 * (1.0 - q4),
        }
        out = np.zeros((len(q1), len(STAGES)), dtype=float)
        for s, idx in _STAGE_INDEX.items():
            out[:, idx] = stage_vals[s]
        return out


# --- MapModels ---------------------------------------------------------------

@dataclass
class MapModels:
    """Fitted difficulty map + stage model. `cat_encoding`/`metadata` are
    plain dict/JSON-ish (joblib-picklable alongside the fitted Pipelines).

    `family` selects which underlying fit `predict_p`/`predict_stage` route
    through -- "two_model" (default, backward-compatible): `p_success` +
    `p_stage`, the original disconnected binary + multinomial fits, both
    populated, `sequential` left None. "sequential": `sequential` (a
    `SequentialStageModel`) populated, `p_success`/`p_stage` left None.
    Exactly one of (`p_success`+`p_stage`) / `sequential` is populated for
    any given `family`; `predict_p`/`predict_stage`'s signature and output
    shapes are IDENTICAL either way -- callers (clustering.py, eval_set.py,
    wells.py) never need to know which family they got."""
    p_success: Pipeline = None
    p_stage: Pipeline = None
    cat_encoding: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    family: str = "two_model"
    sequential: SequentialStageModel = None

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
        if self.family == "sequential":
            return self.sequential.predict_p(X)
        return _positive_class_proba(self.p_success, X)

    def predict_stage(self, df_or_records) -> np.ndarray:
        """p_stage(x): P(stage | x) over STAGES, shape (n, 5); every row sums
        to 1."""
        X = self.predict_features(df_or_records)
        if self.family == "sequential":
            return self.sequential.predict_stage(X)
        proba = self.p_stage.predict_proba(X)
        return _reindex_stage_proba(proba, self.p_stage.classes_)

    def predict_gate_probs(self, df_or_records) -> dict:
        """Sequential-family-only diagnostic: {"reach": q1, "grasp": q2,
        "transport": q3, "place": q4}, each shape (n,) -- see
        `SequentialStageModel.predict_gate_probs`. Raises for the
        two_model family (it has no gates to report)."""
        assert self.family == "sequential", (
            f"predict_gate_probs is only defined for family='sequential', "
            f"got family={self.family!r}"
        )
        X = self.predict_features(df_or_records)
        return self.sequential.predict_gate_probs(X)


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


def _fit_sequential_core(df: pd.DataFrame, k: int = CAT_TE_K) -> MapModels:
    """Shared fit path for the "sequential" family -- the `_fit_core`
    counterpart used by both `fit(df, family="sequential")` (all of df_diag,
    the production model) and `_oof_predict` (one GroupKFold training fold
    at a time). Calls the EXACT SAME `category_target_encoding`/
    `_build_feature_matrix` helpers `_fit_core` calls (never forked), then
    fits each of the four gates (`_fit_gate`) on its own conditional
    population/label (`_gate_populations_and_labels`) instead of `_fit_core`'s
    single all-rows p_success fit + single all-rows multinomial p_stage
    fit."""
    _assert_canonical(df)
    cat_encoding, global_mean = category_target_encoding(df, k=k)
    X = _build_feature_matrix(df, cat_encoding, global_mean)
    pops = _gate_populations_and_labels(df)

    gates = {}
    gate_meta = {}
    for name in GATE_NAMES:
        pop_mask, label = pops[name]
        gate = _fit_gate(X[pop_mask], label[pop_mask].astype(int))
        gates[name] = gate
        gate_meta[name] = {"n_train": gate.n_train, "classes_seen": gate.classes_seen}

    sequential = SequentialStageModel(gates=gates)
    metadata = {
        "global_mean": global_mean,
        "n_rows": int(len(df)),
        "n_categories": int(df["category"].nunique()),
        "feature_columns": list(FEATURE_COLUMNS),
        "raw_feature_columns": list(RAW_FEATURE_COLUMNS),
        "stages": list(STAGES),
        "cat_te_k": k,
        "gate_names": list(GATE_NAMES),
        "gate_meta": gate_meta,
    }
    return MapModels(family="sequential", sequential=sequential,
                      cat_encoding=cat_encoding, metadata=metadata)


def fit(df_diag: pd.DataFrame, family: str = "two_model") -> MapModels:
    """Fit the difficulty map on ALL of `df_diag` -- the production model
    used for inference (Tasks 9/10). `family="two_model"` (default,
    backward-compatible): p_hat_0 (p_success) + p_stage, the original
    disconnected binary + multinomial fits. `family="sequential"`: the
    continuation-ratio `SequentialStageModel` (see module docstring). For an
    honest held-out accuracy estimate, call `validation_report(fit(df_diag,
    family=...), df_diag)` instead of evaluating this fit against df_diag
    directly (see module docstring); `compare_families(df_diag)` fits + cross
    -validates BOTH families and picks a winner in one call."""
    if family == "sequential":
        return _fit_sequential_core(df_diag, k=CAT_TE_K)
    if family == "two_model":
        return _fit_core(df_diag)
    raise ValueError(f"unknown family {family!r}, expected 'two_model' or 'sequential'")


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


def _oof_predict(df: pd.DataFrame, n_splits: int = DEFAULT_N_SPLITS, k: int = CAT_TE_K,
                  family: str = "two_model"):
    """Fold-local out-of-fold predictions: for each GroupKFold(by start_id)
    fold, refit (`_fit_core` for "two_model" / `_fit_sequential_core` for
    "sequential", including a fold-LOCAL cat_te either way) on the training
    rows only and predict on the held-out rows -- never letting a held-out
    row's own label reach its category's shrinkage encoding. Returns
    (oof_p, oof_stage_proba, fold_id, n_splits_eff, oof_gate), where
    `oof_gate` is None for "two_model" and, for "sequential", a
    {GATE_NAMES: (n,) ndarray} of each gate's held-out q -- evaluated on
    EVERY held-out row regardless of that row's own true stage (a fold
    model's gate can always be queried; see `SequentialStageModel.
    predict_gate_probs`), so `validation_report` can later intersect it with
    each gate's true conditional population itself."""
    splits, n_splits_eff = _group_kfold_splits(df, n_splits)
    oof_p = np.full(len(df), np.nan)
    oof_stage = np.full((len(df), len(STAGES)), np.nan)
    fold_id = np.full(len(df), -1, dtype=int)
    oof_gate = {name: np.full(len(df), np.nan) for name in GATE_NAMES} if family == "sequential" else None
    fit_core = _fit_sequential_core if family == "sequential" else _fit_core
    for fi, (tr, va) in enumerate(splits):
        fold_models = fit_core(df.iloc[tr], k=k)
        va_df = df.iloc[va]
        oof_p[va] = fold_models.predict_p(va_df)
        oof_stage[va] = fold_models.predict_stage(va_df)
        fold_id[va] = fi
        if family == "sequential":
            gate_probs = fold_models.predict_gate_probs(va_df)
            for name in GATE_NAMES:
                oof_gate[name][va] = gate_probs[name]
    return oof_p, oof_stage, fold_id, n_splits_eff, oof_gate


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
    `models` supplies the family to refit each fold as (`models.family`) and
    the k it was fit with (metadata["cat_te_k"]), and its own global_mean is
    echoed into the report for pairing/sanity.

    For `models.family == "sequential"` the report additionally carries
    `gate_auc`: {GATE_NAMES: held-out AUC}, each computed ONLY on that
    gate's own true conditional population (`_gate_populations_and_labels`,
    intersected with the held-out mask) -- e.g. the "grasp" gate's AUC is
    scored only over rows that actually reached, never over never_reached
    rows that never got a chance to grasp. A gate whose (held-out,
    population-restricted) labels have fewer than 2 classes -- the
    "degenerate gate population" case (e.g. zero rows ever grasped anything
    in this df) -- reports NaN for that gate rather than raising, the same
    "loud NaN, not a crash" convention `auc` itself already uses above."""
    _assert_canonical(df)
    family = models.family
    k = models.metadata.get("cat_te_k", CAT_TE_K)
    oof_p, oof_stage, fold_id, n_splits_eff, oof_gate = _oof_predict(
        df, n_splits=n_splits, k=k, family=family)
    valid = fold_id >= 0

    y = df["success"].astype(int).to_numpy()[valid]
    p = oof_p[valid]
    ll = float(log_loss(y, p, labels=[0, 1]))
    auc = float(roc_auc_score(y, p)) if pd.unique(y).size >= 2 else float("nan")
    calib = _calibration_bins(p, y.astype(float), CALIBRATION_N_BINS)

    y_stage = df["failure_stage"].astype(str).to_numpy()[valid]
    stage_ll = float(log_loss(y_stage, oof_stage[valid][:, _STAGE_SORT_PERM],
                               labels=list(_STAGES_SORTED)))

    report = {
        "family": family,
        "n": int(valid.sum()),
        "n_groups": int(pd.unique(df["start_id"]).size),
        "n_splits": n_splits_eff,
        "log_loss": ll,
        "auc": auc,
        "calibration_bins": calib,
        "stage_log_loss": stage_ll,
        "global_mean": models.metadata.get("global_mean"),
    }

    if family == "sequential":
        pops = _gate_populations_and_labels(df)
        gate_auc = {}
        for name in GATE_NAMES:
            pop_mask, label = pops[name]
            mask = valid & pop_mask
            yg = label[mask].astype(int)
            qg = oof_gate[name][mask]
            finite = ~np.isnan(qg)
            yg, qg = yg[finite], qg[finite]
            if len(yg) == 0 or pd.unique(yg).size < 2:
                gate_auc[name] = float("nan")
            else:
                gate_auc[name] = float(roc_auc_score(yg, qg))
        report["gate_auc"] = gate_auc

    return report


def compare_families(df: pd.DataFrame, n_splits: int = DEFAULT_N_SPLITS) -> dict:
    """Fits BOTH families (`fit(df, family=...)`, each the full-df production
    fit) and cross-validates each with `validation_report` -- same
    GroupKFold-by-start_id folds, cat_te and gates/models all refit fresh
    per fold for both (no leakage, identical discipline to the single-family
    path). Picks a `winner` by held-out p_success log-loss (a proper scoring
    rule; LOWER is better), ties broken by held-out stage log-loss (also
    lower is better) -- log-loss rather than AUC because AUC only measures
    ranking, not the calibrated probabilities Task 9/10's downstream
    consumers (clustering's p_hat block, eval_set's stratification) actually
    use. Deterministic for a fixed `df`: GroupKFold has no shuffle/random
    state and every LogisticRegression/DummyClassifier fit here is
    deterministic given its (X, y).

    Returns {"two_model": <report>, "sequential": <report>,
    "winner": "two_model"|"sequential"}."""
    _assert_canonical(df)
    report_two = validation_report(fit(df, family="two_model"), df, n_splits=n_splits)
    report_seq = validation_report(fit(df, family="sequential"), df, n_splits=n_splits)

    if report_two["log_loss"] < report_seq["log_loss"]:
        winner = "two_model"
    elif report_seq["log_loss"] < report_two["log_loss"]:
        winner = "sequential"
    else:
        winner = "two_model" if report_two["stage_log_loss"] <= report_seq["stage_log_loss"] else "sequential"

    return {"two_model": report_two, "sequential": report_seq, "winner": winner}


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

    # Family A/B: fit + cross-validate BOTH "two_model" (the original
    # disconnected binary+multinomial fit) and "sequential" (the
    # continuation-ratio gate model) on these same diag rows, and let
    # held-out validation -- not a hardcoded choice -- pick which one this
    # run actually ships (see compare_families's docstring for the
    # log-loss-then-stage-log-loss tiebreak rule).
    comparison = compare_families(df)
    for fam in ("two_model", "sequential"):
        print(f"VALIDATION_REPORT[{fam}]", json.dumps(comparison[fam], indent=2))
    winner = comparison["winner"]
    report = comparison[winner]
    print(f"map_fit: family A/B winner = {winner!r} "
          f"(two_model log_loss={comparison['two_model']['log_loss']:.4f}, "
          f"sequential log_loss={comparison['sequential']['log_loss']:.4f})")

    report_path = (Path(args.report_out) if args.report_out is not None
                   else config.LEDGER_DIR / "map_validation_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(comparison, indent=2, sort_keys=True))
    print(f"map_fit: wrote validation report (both families + winner) to {report_path}")

    # Investigate-before-proceeding gate (task-8-brief.md Step 3 / this
    # module's AUC_GATE_MIN docstring note), now applied to the WINNING
    # family's held-out AUC. `not (auc >= AUC_GATE_MIN)` is deliberately used
    # instead of `auc < AUC_GATE_MIN` so a NaN AUC (e.g. a degenerate
    # all-success/all-fail diag slice with zero label variance -- see
    # validation_report's `pd.unique(y).size >= 2` guard) ALSO fails the
    # gate loudly, rather than silently comparing False against a NaN.
    if not (report["auc"] >= AUC_GATE_MIN):
        print(f"!!! GATE FAILED: winner family {winner!r} held-out AUC {report['auc']} "
              f"< {AUC_GATE_MIN} -- investigate before proceeding (wrong join? leaked "
              f"constants? see task-8-brief.md Step 3). Refusing to save a model to "
              f"{args.out or config.MAP_MODELS_JOBLIB} -- run_baseline.sh must NOT "
              f"build E or run the baseline eval against this fit.")
        sys.exit(1)

    models = fit(df, family=winner)

    path = save(models, path=args.out)
    print(f"map_fit: gate passed (family={winner}, AUC={report['auc']:.4f} >= "
          f"{AUC_GATE_MIN}) -- saved {path}")


if __name__ == "__main__":
    _main()
