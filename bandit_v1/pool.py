"""Build the frozen pool_demos snapshot table (one row per candidate demo episode).

Source: config.FX_POOL_JSON (weakregion/factor_analysis/fx_pool.json), produced by
the earlier factor-analysis recon. That file is a dict, NOT a flat list of rows:
    {"fields": [...], "rows": [[...], ...], "cats": [...], "stats": {...},
     "arm_names": [...], "arm_members": {...}}
The 9,885-row per-demo table lives at d["rows"], with each row a positional list
whose column order is given by d["fields"] ==
    ['i', 'cat', 'h', 'w', 'layout', 'r', 'x', 'y', 'side', 'ambig', 'len']

Deviations from the recon's guessed field mapping (verified empirically, see
bandit_v1/tests/test_pool.py + task-2 report for the inspection commands):
  - 'cat' is NOT the category string itself. It is an integer index into the
    top-level d["cats"] list (81 entries, each {"name", "sr", "n", "h", "w"}).
    category = cats[row_cat_idx]["name"]. Direct rename cat->category (as a
    plain column copy) would have produced integers, not category names.
  - The 'category' column is canonicalized via categories.canonical_category
    (config.CATEGORY_ALIASES) after the cats[...]["name"] lookup above -- task 3's
    fix for the cross-process determinism gate's category-alias mismatch (see
    .superpowers/sdd/task-3-report.md). 17/1516 robocasa mjcf instances are
    registered under two overlapping category names (e.g. "jug_wide_opening" is a
    full subset of "jug"); this table's raw cats[...]["name"] values can contain
    either alias name, so they are folded into their canonical name here to match
    states.py's fingerprint/start_features convention (same alias table, same
    function). Concretely: every "jug_wide_opening" row becomes "jug" and every
    "saucepan_with_lid" row becomes "saucepan" (the latter has 0 rows in this
    table's current data, but is folded for consistency/future-proofing anyway).
  - 'h'/'w' per row equal cats[row_cat_idx]["h"/"w"] exactly (per-category eval
    means broadcast onto every row of that category), not per-episode object
    geometry. Kept as-is per the output schema (columns named h, w).
  - 'side' is stored as int (-1 or 1) in the source rows and is kept as int
    (-1/+1) in the output table (canonical encoding: `side: int in {-1, +1}`
    everywhere in bandit_v1 — never str). Pool-side and eval-side features
    are later joined into one embedding, so an int-vs-str mismatch there
    would silently fail to compare; casting to str on the way out is a bug,
    not a schema requirement.
  - 'i' is a dense bijection over 0..9884 (verified unique == len(rows)) and is
    used directly as episode_index; no re-derivation needed.
  - 'r' and 'ambig' are present in the source fields but are NOT part of the
    required output schema, so they are dropped. Empirically, r == max(|x|,
    |y|) exactly for all 9,885 rows (the radius to the dominant axis), which
    is the key to the side finding below.

Side convention (empirical, verified over all 9,885 rows — see
bandit_v1/tests/test_pool.py::test_row_level_fidelity for spot checks at
i=0,1824,9884): side is NOT a pure function of sign(x) or sign(y) alone.
sign(x_rel) matches side for 71.8% of rows (7,096/9,885); sign(y_rel) matches
for 81.1% of rows (8,015/9,885). The much better predictor is the
dominant-axis rule: side == sign(x_rel) if |x_rel| >= |y_rel| else
sign(y_rel) (i.e. the sign of whichever coordinate has the larger
magnitude, consistent with r == max(|x|,|y|) above) — this matches for
93.8% of rows (9,276/9,885). Any downstream eval-side feature that needs to
derive `side` from (x_rel, y_rel) alone should use this dominant-axis rule,
not sign(x_rel) or sign(y_rel) individually, and should not assume perfect
(100%) agreement — ~6.2% of rows are genuine exceptions (not explained by
the 'ambig' flag: mismatch rate is ~6.1% whether ambig is 0 or 1).

D0 flag: in_d0 is joined from config.ARMS_JSON's "base_episodes" list (400 ints,
a subset of the 9,885 episode_index values) — in_d0 = episode_index in D0.

This is a snapshot table, not an event log, so it is written directly via
df.to_parquet (NOT via ledger.append_rows, which is for append-only event
tables and would be wrong here — a re-run must overwrite, not accumulate).
"""
import json

import pandas as pd

from . import categories, config, ledger

OUTPUT_COLUMNS = [
    "episode_index", "category", "h", "w", "layout",
    "x_rel", "y_rel", "side", "traj_len", "in_d0",
]

POOL_PARQUET = config.LEDGER_DIR / "pool_demos.parquet"
HASHES_JSON = config.LEDGER_DIR / "hashes.json"


def build_pool_table(write: bool = True) -> pd.DataFrame:
    """Load fx_pool.json + arms.json, return the 9,885-row pool table.

    If write=True (default), also writes LEDGER_DIR/pool_demos.parquet and
    merge-updates LEDGER_DIR/hashes.json with fx_pool.json's sha256.
    """
    fx = json.load(open(config.FX_POOL_JSON))
    fields = fx["fields"]
    cats = fx["cats"]

    rows = fx["rows"]
    df = pd.DataFrame(rows, columns=fields)

    out = pd.DataFrame({
        "episode_index": df["i"].astype(int),
        "category": df["cat"].map(lambda c: categories.canonical_category(cats[int(c)]["name"])),
        "h": df["h"].astype(float),
        "w": df["w"].astype(float),
        "layout": df["layout"].astype(int),
        "x_rel": df["x"].astype(float),
        "y_rel": df["y"].astype(float),
        "side": df["side"].astype(int),
        "traj_len": df["len"].astype(int),
    })

    arms = json.load(open(config.ARMS_JSON))
    d0 = set(arms["base_episodes"])
    out["in_d0"] = out["episode_index"].isin(d0)

    out = out[OUTPUT_COLUMNS]

    if write:
        config.LEDGER_DIR.mkdir(parents=True, exist_ok=True)
        tmp = POOL_PARQUET.with_suffix(".tmp.parquet")
        out.to_parquet(tmp, index=False)
        tmp.replace(POOL_PARQUET)

        hashes = {}
        if HASHES_JSON.exists():
            hashes = json.load(open(HASHES_JSON))
        hashes["fx_pool.json"] = ledger.file_hash(config.FX_POOL_JSON)
        tmp_h = HASHES_JSON.with_suffix(".tmp.json")
        with open(tmp_h, "w") as f:
            json.dump(hashes, f, indent=2, sort_keys=True)
        tmp_h.replace(HASHES_JSON)

    return out


def well_mask(df: pd.DataFrame) -> pd.Series:
    """True for pool episodes NOT in D0 (the 'well' available for prescription)."""
    return ~df["in_d0"]
