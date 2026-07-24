"""Retrieval-channel pull draws with gates (bandit_v1 Task 11).

Design: weakregion/BANDIT_V1_DESIGN.md section 3 ("One pull") + section 1 item 4
(epsilon-exclusion). This module implements exactly the "demos = draw B distinct
demos ..." line of a pull, plus its two applicable gates (novelty, epsilon) --
the third gate the design lists, "quality (light -- pool demos are pre-filtered
successes: length bounds, parse sanity)", is a property of `fx_pool.json`'s
construction (upstream of `pool.build_pool_table`), not something this module
re-checks; it is out of this task's scope.

Zero import-time dependency on Task 10 (`arms.yaml` + the fitted region/cluster
model): this module never imports or reads either. Callers pass in whatever
Task 10 eventually produces as plain data:
  - `regions`: a `pd.Series` indexed by `episode_index` (aligned to, i.e. a
    subset of the index space of, `pool_df["episode_index"]`), whose values are
    arm-name strings. Membership in arm `k` is `regions[regions == k].index`.
  - `e_features`: a `pd.DataFrame` with columns `category, x_rel, y_rel` -- one
    row per Eval-set (E) saved-state start, i.e. `states.start_features()`'s
    output collected over all of E and reduced to those 3 columns.
  - `pool_df`: the pool table from `pool.build_pool_table()` (or an
    equivalent-schema DataFrame in tests), columns per its OUTPUT_COLUMNS,
    notably `in_d0` (bool) which supplies the novelty gate's D0 comparison set
    (the D0 rows of this SAME frame -- no separate D0 features file).

Random arm: the literal string `"random"` (module constant `RANDOM_ARM`,
case-sensitive) is not a `regions` value at all -- it means "candidates = all of
W", ignoring `regions` entirely (design section 2 item 5 / section 0 item 4:
"Random arm: uniform draw over all of W -- always included").

Gates (identical distance rule, different comparison sets -- see
`_conflict_mask`, the one shared implementation both go through so the two
gates cannot drift apart):
  - `eps_conflict`: candidate vs every Eval-set (E) start (design section 1
    item 4). A demo conflicts with an E start iff they share a canonical
    category AND their (x_rel, y_rel) Euclidean distance is < `eps_xy`
    (strict). Conflicting demos are excluded before B is ever assembled --
    a demo must never be a near-duplicate of anything the policy will be
    evaluated on.
  - `novelty`: same distance rule, two comparison sets: (1) every D0 row of
    `pool_df` (a demo must not be a near-duplicate of something already in
    the base training set), checked once per candidate up front, same as
    eps_conflict; and (2) every OTHER demo already kept in the same pull, so
    a pull is not internally redundant -- this half is necessarily sequential
    (it depends on what has been kept so far) and is applied inside FPS
    growth (`_fps_select`), not as a pre-filter.

Draw mechanics (`pull_demos`), per the brief:
  1. `candidates` = W (`pool.well_mask`) intersected with the arm's region (or
     all of W for `"random"`), MINUS any row that fails eps_conflict (vs E) or
     the D0 half of novelty. This is a fixed set for the pull -- everything
     after this point only subsamples from it.
  2. If `len(candidates) > 3*B`: draw 3B of them uniformly at random WITHOUT
     replacement, consuming `rng` -- call this `pool_for_fps`. Otherwise
     `pool_for_fps = candidates` untouched (nothing to narrow).
     This extra draw is the ONLY reason pulls of the same arm differ: farthest-
     point sampling is otherwise a deterministic function of its input set, so
     running FPS straight over the full (unchanging, per-arm) region would
     select the identical B demos every single pull -- silently breaking the
     "redraw every pull, with-replacement across pulls" independence the
     design's scheduler assumes (BANDIT_V1_DESIGN.md section 3/9). Routing the
     only randomness through a uniform pre-draw keeps each pull an independent
     sample while still spending the FPS step's diversity budget on 3B, not
     the (potentially far larger) full region.
  3. Farthest-point-subsample `pool_for_fps` down to B in STANDARDIZED
     (h, w, x_rel, y_rel) space -- standardized using `pool_for_fps`'s own
     mean/std (i.e. over the candidate set that is actually being subsampled
     from THIS pull, not the full pool or the full region), per the brief's
     "standardize over the candidate set; document" instruction. See
     `_standardize`.
  4. The within-pull half of the novelty gate is folded into step 3
     (`_fps_select`): FPS visits points in decreasing farthest-point order and
     only actually KEEPS a visited point if it does not novelty-conflict with
     whatever is already kept; a rejected point is never revisited but also
     never blocks the walk -- growth simply continues past it. This is
     simultaneously the brief's "apply the within-pull rule as a post-filter"
     (a rejected point is exactly one FPS would have kept, filtered out
     because it duplicates an earlier keeper) AND its "refill from remaining
     candidates by the same FPS criterion" (continuing the identical
     farthest-point walk past a rejection instead of stopping at the
     naive top-B is the refill: the walk only stops once B points are KEPT or
     `pool_for_fps` is exhausted). A literal two-pass
     (select-top-B-then-post-filter-then-separately-refill) implementation
     would do the same growth walk with the same skip rule, so the single
     continuous pass is behaviorally equivalent and simpler.
     If `pool_for_fps` is exhausted with fewer than B kept, `pull_demos`
     raises -- a pull that cannot fill B after gating is exactly the "well
     cannot fill an arm" failure the design says should halt the run (section
     9), scoped here to one pull's realized gate losses rather than the
     arms-freeze well-count table.

All-distinct within a pull falls out for free: `pool_for_fps` is drawn without
replacement from `candidates` (itself deduplicated -- `pool_df` is one row per
`episode_index`), and FPS/refill only ever adds an index once (`taken`).
Independent redraws across pulls fall out because `pull_demos` holds no module
state; it is a pure function of its arguments, and pulls of the same arm are
just repeated calls with fresh `rng` draws.
"""
import numpy as np
import pandas as pd

from . import categories, config, pool

RANDOM_ARM = "random"

_KNOB_COLS = ("h", "w", "x_rel", "y_rel")


def _conflict_mask(cand_df: pd.DataFrame, other_df: pd.DataFrame, eps_xy: float) -> np.ndarray:
    """Shared core of both gates: boolean array aligned to `cand_df`'s row
    order, True at row i iff `cand_df.iloc[i]` shares a canonical category with
    AT LEAST ONE row of `other_df` AND its (x_rel, y_rel) Euclidean distance to
    that row is strictly < `eps_xy`. Both `eps_conflict` (vs E) and the D0/
    within-pull halves of `novelty` are this identical rule against a
    different `other_df`, computed here once so they cannot silently diverge.
    Categories are re-canonicalized defensively (categories.canonical_category)
    even though both `pool_df` and `e_features` are expected to already carry
    canonical names by construction -- same paranoid-at-every-entry-point
    convention as diagnosis.py/states.py, cheap and idempotent if redundant."""
    n = len(cand_df)
    if n == 0 or len(other_df) == 0:
        return np.zeros(n, dtype=bool)
    c_cat = cand_df["category"].map(categories.canonical_category).to_numpy()
    o_cat = other_df["category"].map(categories.canonical_category).to_numpy()
    same_cat = c_cat[:, None] == o_cat[None, :]
    dx = cand_df["x_rel"].to_numpy(dtype=float)[:, None] - other_df["x_rel"].to_numpy(dtype=float)[None, :]
    dy = cand_df["y_rel"].to_numpy(dtype=float)[:, None] - other_df["y_rel"].to_numpy(dtype=float)[None, :]
    dist = np.hypot(dx, dy)
    return (same_cat & (dist < eps_xy)).any(axis=1)


def _row_frame(category, x_rel, y_rel) -> pd.DataFrame:
    """One-row DataFrame with just the 3 columns `_conflict_mask` needs, for the
    scalar single-demo gate functions below."""
    return pd.DataFrame({"category": [category], "x_rel": [float(x_rel)], "y_rel": [float(y_rel)]})


def eps_conflict(category: str, x_rel: float, y_rel: float, e_features: pd.DataFrame,
                  eps_xy: float = config.EPS_XY) -> bool:
    """True iff (category, x_rel, y_rel) is an epsilon-conflict with ANY row of
    `e_features` (columns category, x_rel, y_rel -- one row per Eval-set
    start): same canonical category AND hypot(Δx_rel, Δy_rel) < eps_xy
    (strict). Design section 1 item 4's epsilon-exclusion."""
    return bool(_conflict_mask(_row_frame(category, x_rel, y_rel), e_features, eps_xy)[0])


def novelty_conflict(category: str, x_rel: float, y_rel: float, other_df: pd.DataFrame,
                      eps_xy: float = config.EPS_XY) -> bool:
    """Same rule as `eps_conflict`, against a caller-supplied comparison set
    (D0 rows, or the demos already kept so far in a pull) instead of E starts."""
    return bool(_conflict_mask(_row_frame(category, x_rel, y_rel), other_df, eps_xy)[0])


def _standardize(df: pd.DataFrame) -> np.ndarray:
    """Z-score `_KNOB_COLS` (h, w, x_rel, y_rel) using `df`'s OWN mean/std --
    i.e. over whatever candidate set is being subsampled (see `pull_demos`'s
    docstring point 3), not the full pool or the full region. A zero-variance
    column (std == 0 -- e.g. a synthetic single-category test pool with
    constant h) is left un-scaled (divided by 1) after centering rather than
    producing NaN/inf; such a column then contributes 0 to every pairwise
    distance, which is the correct no-information behavior."""
    x = df[list(_KNOB_COLS)].to_numpy(dtype=float)
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (x - mu) / sd


def _fps_select(df: pd.DataFrame, B: int, rng: np.random.Generator, eps_xy: float) -> pd.DataFrame:
    """Farthest-point sampling of up to B rows from `df`, standardized per
    `_standardize`, with the within-pull novelty gate folded into the walk (see
    the module docstring's point 4 for why one pass suffices for
    select+post-filter+refill). Seed point is chosen via `rng.integers`, so the
    entire walk -- and hence `pull_demos`'s output -- is deterministic given
    `rng`'s state (this is what makes two `pull_demos` calls seeded identically
    return identical results, and two calls with different seeds generally
    differ).

    Raises ValueError if `df` is exhausted (every row visited) before B rows
    are kept -- not enough gate-compatible rows existed in this pull's
    candidate set.
    """
    n = len(df)
    if n < B:
        raise ValueError(f"_fps_select: only {n} rows available, need B={B}")
    if n == B:
        # Nothing to subsample -- but still worth checking there is no
        # internal novelty conflict, per the same gate the general path uses.
        df = df.reset_index(drop=True)

    z = _standardize(df)
    taken = np.zeros(n, dtype=bool)   # visited (kept OR gate-rejected) -- never revisited
    kept_idx: list = []

    s0 = int(rng.integers(n))
    taken[s0] = True
    kept_idx.append(s0)
    min_dist = np.linalg.norm(z - z[s0], axis=1)

    while len(kept_idx) < B and not taken.all():
        candidate_dist = np.where(taken, -np.inf, min_dist)
        nxt = int(np.argmax(candidate_dist))
        taken[nxt] = True
        # Update the farthest-point frontier regardless of whether `nxt` is
        # ultimately kept -- the walk explores based on everywhere it has
        # been, not just what survived the gate (see module docstring).
        d_new = np.linalg.norm(z - z[nxt], axis=1)
        min_dist = np.minimum(min_dist, d_new)

        kept_df = df.iloc[kept_idx]
        if _conflict_mask(df.iloc[[nxt]], kept_df, eps_xy)[0]:
            continue  # within-pull novelty conflict: skip, do not add to kept_idx
        kept_idx.append(nxt)

    if len(kept_idx) < B:
        raise ValueError(
            f"_fps_select: pool exhausted with only {len(kept_idx)}/{B} kept "
            f"after within-pull novelty filtering")

    return df.iloc[kept_idx].reset_index(drop=True)


def pull_demos(arm: str, B: int, rng: np.random.Generator, pool_df: pd.DataFrame,
                regions: pd.Series, e_features: pd.DataFrame,
                eps_xy: float = config.EPS_XY) -> list:
    """One pull's worth of B distinct demo episode_index values for `arm`.

    candidates = W ∩ region_arm (or all of W if arm == RANDOM_ARM) minus rows
    that fail eps_conflict (vs e_features) or the D0 half of novelty (vs
    pool_df's own in_d0 rows); then 3B-uniform-then-FPS-to-B with the
    within-pull novelty half folded in, per the module docstring. See there for
    the full mechanics and why with-replacement independence across pulls
    holds. Raises ValueError if the arm has no region membership, no W ∩
    region, or too few gate-surviving candidates to reach B.
    """
    well = pool_df[pool.well_mask(pool_df)]

    if arm == RANDOM_ARM:
        region_df = well
    else:
        region_ids = set(regions.index[regions == arm])
        if not region_ids:
            raise ValueError(f"pull_demos: arm {arm!r} has no episodes in `regions`")
        region_df = well[well["episode_index"].isin(region_ids)]

    if len(region_df) == 0:
        raise ValueError(f"pull_demos: arm {arm!r}: W ∩ region is empty")

    e_fail = _conflict_mask(region_df, e_features, eps_xy)
    d0_df = pool_df[pool_df["in_d0"]]
    d0_fail = _conflict_mask(region_df, d0_df, eps_xy)
    candidates = region_df[~(e_fail | d0_fail)].reset_index(drop=True)

    if len(candidates) < B:
        raise ValueError(
            f"pull_demos: arm {arm!r}: only {len(candidates)} candidates survive "
            f"gating, need B={B}")

    n = len(candidates)
    if n > 3 * B:
        idx = rng.choice(n, size=3 * B, replace=False)
        pool_for_fps = candidates.iloc[idx].reset_index(drop=True)
    else:
        pool_for_fps = candidates

    kept = _fps_select(pool_for_fps, B, rng, eps_xy)
    return kept["episode_index"].astype(int).tolist()


def log_selector_scores(demo_ids: list, pool_df: pd.DataFrame) -> dict:
    """§7.4 logged-not-deployed selector-ablation inputs for one pull's chosen
    `demo_ids`: mean pairwise standardized-knob distance within the chosen set,
    and the mean distance from each chosen demo to its nearest D0 row.

    Standardization uses the FULL `pool_df`'s (h, w, x_rel, y_rel) mean/std --
    deliberately NOT the pull's own candidate-set standardization `pull_demos`
    used internally for FPS -- so scores logged for pulls of different arms
    (which may have drawn from regions with very different candidate-set
    statistics) sit in one common, comparable coordinate system. This is a
    read-only diagnostic computed after the fact from the ledger; it is never
    fed back into selection (design section 7 item 4)."""
    cols = list(_KNOB_COLS)
    mu = pool_df[cols].mean().to_numpy()
    sd = pool_df[cols].std().to_numpy()
    sd = np.where(sd == 0, 1.0, sd)

    chosen = pool_df[pool_df["episode_index"].isin(demo_ids)]
    zc = (chosen[cols].to_numpy(dtype=float) - mu) / sd
    n = len(zc)

    if n >= 2:
        diff = zc[:, None, :] - zc[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(n, k=1)
        mean_pairwise_dist = float(dist[iu].mean())
    else:
        mean_pairwise_dist = float("nan")

    d0 = pool_df[pool_df["in_d0"]]
    if len(d0) > 0 and n > 0:
        zd0 = (d0[cols].to_numpy(dtype=float) - mu) / sd
        d = np.linalg.norm(zc[:, None, :] - zd0[None, :, :], axis=2)
        mean_dist_to_nearest_d0 = float(d.min(axis=1).mean())
    else:
        mean_dist_to_nearest_d0 = float("nan")

    return {
        "n_demos": n,
        "mean_pairwise_dist": mean_pairwise_dist,
        "mean_dist_to_nearest_d0": mean_dist_to_nearest_d0,
    }
