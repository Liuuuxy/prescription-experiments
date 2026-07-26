"""Well membership + the B rule (bandit_v1 Task 10, second half).

Design: weakregion/BANDIT_V1_DESIGN.md section 2 items 5-6 ("Per-arm samplers"
/ "Freeze ... compute the per-arm well-count table ... apply the B rule") and
section 0 item 2 (the B rule itself). Brief: .superpowers/sdd/task-10-brief.md.

Depends on `clustering.py` (for `ZSpec`/`transform_z`, to embed pool rows into
the SAME z-space the diag-fit centroids live in) -- `clustering.py` itself
never imports this module at top level (only lazily, inside its CLI's
`compute_draft`), so this one-directional dependency never cycles.
"""
import numpy as np
import pandas as pd

from . import clustering, config, pool
from .draw import RANDOM_ARM  # the literal "random" string; draw.py owns it


def assign_regions(pool_df: pd.DataFrame, models, arms_spec: dict) -> pd.Series:
    """Nearest-centroid arm assignment for W (`pool.well_mask`) rows of
    `pool_df` ONLY -- D0 rows are excluded entirely (never even appear in the
    returned index), per design section 9's "never touch E/D0 for demo
    selection" family of invariants (D0 must not be scored into an arm at
    all, not just excluded from later draws).

    `arms_spec` is a loaded arms.yaml-shaped dict: `{"arms": [{"name",
    "centroid": {"standardized": [...floats...], ...}, ...}, ...],
    "z_spec": {...}}` (draft or final -- both have this shape; see
    clustering.py). Each W row's feature vector is embedded via
    `clustering.transform_z` using `arms_spec["z_spec"]` (the SAME frozen
    standardization the arms' centroids themselves were computed in) and
    assigned to whichever arm's `centroid.standardized` is nearest in
    Euclidean z-space.

    This function is descriptor-agnostic BY CONSTRUCTION: `z_spec`'s own
    `descriptor` field ("hybrid" -- knob+p_hat+p_stage, 11-dim -- or
    "behavior" -- p_hat+p_stage only, 6-dim, no knob block; see clustering.py's
    module docstring) is read by `clustering.ZSpec.from_dict` and dispatched
    on inside `clustering.transform_z` itself, so a "behavior"-frozen
    arms.yaml is honored automatically here -- `well_df` (this module's own
    variable) is embedded through whichever blocks the frozen `z_spec` names,
    and the resulting `Z`'s width always matches `centroids`' width (both were
    produced by the same frozen `z_spec`), so the nearest-centroid distance
    computation below never needs to know or care which descriptor is live.

    Returns a `pd.Series` indexed by `episode_index` (W rows only), values =
    arm name strings -- exactly the shape `draw.pull_demos`'s `regions`
    parameter expects.
    """
    well_df = pool_df[pool.well_mask(pool_df)].reset_index(drop=True)

    z_spec = clustering.ZSpec.from_dict(arms_spec["z_spec"])
    Z = clustering.transform_z(well_df, models, z_spec)

    arms = arms_spec["arms"]
    if not arms:
        raise ValueError("assign_regions: arms_spec has zero arms")
    names = [a["name"] for a in arms]
    centroids = np.stack([np.asarray(a["centroid"]["standardized"], dtype=float) for a in arms])

    d = np.linalg.norm(Z[:, None, :] - centroids[None, :, :], axis=2)
    nearest = np.argmin(d, axis=1)
    assigned = [names[i] for i in nearest]

    region = pd.Series(assigned, index=well_df["episode_index"].to_numpy(), dtype=object)
    region.index.name = "episode_index"
    return region


def well_table(regions: pd.Series) -> pd.DataFrame:
    """Arm x count table from an `assign_regions`-shaped `regions` Series:
    one row per distinct arm name present in `regions` (its W-membership
    count), PLUS a `"random"` row whose count is `len(regions)` -- Random's
    well is all of W by definition (design item 5), not a cluster membership
    count. Sorted by arm name for deterministic output/printing.
    """
    counts = regions.value_counts()
    rows = [{"arm": str(name), "count": int(cnt)} for name, cnt in counts.items()]
    rows.append({"arm": RANDOM_ARM, "count": int(len(regions))})
    return pd.DataFrame(rows).sort_values("arm").reset_index(drop=True)


def choose_B(well_table_df: pd.DataFrame):
    """The B rule (design section 0 item 2): `B = max{b in
    config.B_CANDIDATES : min over CLUSTER arms (Random excluded from the
    min) of well-count >= 3b}`. Returns `(B, limiting_arm)` -- the largest
    surviving `b`, and the name of the (cluster) arm whose well-count set
    that ceiling (first in `well_table_df`'s row order on a tie).

    Raises ValueError if even the smallest candidate in `config.B_CANDIDATES`
    fails -- per design section 9, a well that cannot fill any arm at any
    candidate B must halt the run, never silently shrink B per-arm.
    """
    cluster_rows = well_table_df[well_table_df["arm"] != RANDOM_ARM]
    if cluster_rows.empty:
        raise ValueError("choose_B: well_table has no non-random (cluster) arms")

    limiting_row = cluster_rows.loc[cluster_rows["count"].idxmin()]
    limiting_arm = str(limiting_row["arm"])
    min_count = int(limiting_row["count"])

    for b in sorted(config.B_CANDIDATES, reverse=True):
        if min_count >= 3 * b:
            return int(b), limiting_arm

    smallest = min(config.B_CANDIDATES)
    raise ValueError(
        f"choose_B: even the smallest B candidate ({smallest}) needs "
        f"{3 * smallest} well members in the limiting arm {limiting_arm!r}, but "
        f"it only has {min_count} -- the well cannot fill any arm at any "
        f"candidate B (design section 9: this must halt the run, never "
        f"silently use a per-arm B or mix channels).")
