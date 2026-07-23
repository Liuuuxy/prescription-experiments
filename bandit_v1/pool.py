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
  - 'h'/'w' per row equal cats[row_cat_idx]["h"/"w"] exactly (per-category eval
    means broadcast onto every row of that category), not per-episode object
    geometry. Kept as-is per the output schema (columns named h, w).
  - 'side' is stored as int (-1 or 1) in the source rows. The required output
    schema types it as `side:str`, so it is cast with str() on the way out.
  - 'i' is a dense bijection over 0..9884 (verified unique == len(rows)) and is
    used directly as episode_index; no re-derivation needed.
  - 'r' and 'ambig' are present in the source fields but are NOT part of the
    required output schema, so they are dropped.

D0 flag: in_d0 is joined from config.ARMS_JSON's "base_episodes" list (400 ints,
a subset of the 9,885 episode_index values) — in_d0 = episode_index in D0.

This is a snapshot table, not an event log, so it is written directly via
df.to_parquet (NOT via ledger.append_rows, which is for append-only event
tables and would be wrong here — a re-run must overwrite, not accumulate).
"""
import json

import pandas as pd

from . import config, ledger

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
        "category": df["cat"].map(lambda c: cats[int(c)]["name"]),
        "h": df["h"].astype(float),
        "w": df["w"].astype(float),
        "layout": df["layout"].astype(int),
        "x_rel": df["x"].astype(float),
        "y_rel": df["y"].astype(float),
        "side": df["side"].map(str),
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
