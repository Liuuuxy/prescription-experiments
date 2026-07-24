"""Diagnosis condition selection + saved-state capture (bandit_v1 Task 6).

Design: weakregion/BANDIT_V1_DESIGN.md section 2 item 2 ("Diagnosis batch"): scan
fresh env seeds (base 600000, config.DIAG_ENV_SEED_BASE) and keep a balanced
27-cell grid of N=300 (config.N_DIAG) saved-state conditions --
category-difficulty tercile x height bin x |along-counter offset| bin -- for the
later m=8-rollout diagnosis batch (a subsequent task, not this one). This module
only selects+captures the conditions; it never runs a policy or a rollout.

Pipeline:
  1. `load_tercile_map()` -- category -> difficulty tercile (0=hardest,
     2=easiest), a prior computed once from weakregion/factor_analysis's
     per-category success-rate table and frozen to
     config.DIAG_TERCILE_MAP_JSON (bandit_v1/ledger/diag_tercile_map.json,
     committed) so every re-run/resume uses the identical map. See
     `build_tercile_map` for how it's derived.
  2. `assign_cell(features, tercile_map)` -- pure grid-cell lookup for one
     already-captured start's `states.start_features()` dict.
  3. `select_conditions(...)` -- the scan/capture/keep-discard loop: for each
     candidate seed (in order, starting at seed_base), capture it via
     `states.capture_start` into a `start_%05d` directory, compute its cell, and
     keep it (append to the ledger rows) iff its cell isn't already at
     capacity -- see its docstring for the exact overflow/resume rules.

Category-difficulty prior (`_load_prior_category_rates`): per the task-6 brief,
sourced from weakregion/factor_analysis/fx_episodes.json's top-level "cats" list
([{"name","sr","n","h","w"}, ...] -- one entry per category), the same shape
fx_pool.json's "cats" list has for h/w (which states.py's `_category_hw` already
trusts without re-deriving from raw episodes). "sr" there is NOT recomputable by
re-averaging fx_episodes.json's own "rows": spot-checking every category's
mean(row-level success) against its "cats" entry's "sr" shows a systematic
~0.08-0.09 gap for most categories (e.g. apple: raw mean 0.650 vs sr 0.561) and
is not merely mean-centering (grand_sr=0.465 doesn't explain the gap either) --
"rows" is evidently a different, feature-joined episode slice than whatever
larger/different set "sr" was computed over. Per the brief ("use ... if it has
per-category success rates" -- it does, directly, as "sr"), this module trusts
"cats"[...]["sr"] as-is, unreconciled against "rows", exactly as pool-side code
trusts fx_pool.json's "cats"[...]["h"/"w"]. One category (kettle_non_electric,
n=17) has sr=-0.023, outside a literal [0,1] probability range -- almost
certainly a shrunk/regularized estimate rather than a raw mean -- but this only
affects magnitude, not its (correct) placement at the very bottom of the sorted
difficulty order, so it is used unmodified.

Fallback: if fx_episodes.json is absent or lacks "cats"/"sr", per-category
success is computed directly from config.POOLED_EPISODES_CSV's
`object_category`/`success` columns (mean success per canonicalized category).

Category canonicalization: every category name (from fx_episodes.json's "cats",
pooled_episodes.csv's object_category, or a captured start's
`start_features()["category"]`) is folded through
`categories.canonical_category` (config.CATEGORY_ALIASES) before it is ever used
as a rate-table key, a tercile_map key, or a tercile_map lookup -- same
convention as states.py/pool.py. This matters concretely here: the live env's
"all" obj_groups (PickPlaceCounterToSink, graspable=True, washable=True) can
reach 83 categories, of which fx_episodes.json's prior table only has 81 --
"pot" is genuinely absent (falls back to the middle tercile, per the brief's
clarification) and "saucepan_with_lid" is an ALIAS of a category the prior
table does have ("saucepan") and would incorrectly also fall back to the
middle tercile without canonicalizing first.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import categories, config, states

# --- grid definition -----------------------------------------------------

N_TERCILES = 3
H_BINS = (0.08, 0.212)     # (lo, hi) meters; see _bin3 for the closed/open rule
X_BINS = (0.325, 0.65)     # (lo, hi) meters, applied to |x_rel|
N_CELLS = N_TERCILES * 3 * 3  # 27

ALL_CELLS = tuple(
    (t, h, x) for t in range(N_TERCILES) for h in range(3) for x in range(3)
)


def _bin3(v, lo, hi):
    """3-way bin index for scalar `v` against cutoffs (lo, hi): 0 if v < lo,
    1 if lo <= v <= hi (BOTH edges closed into the middle bin), 2 if v > hi.

    This is the exact rule the brief's boundary cases pin down: h == 0.212
    (the upper h cutoff) lands in the middle bin, and |x_rel| == 0.65 (the
    upper x cutoff) also lands in the middle bin -- i.e. both the lower and
    upper cutoff of the middle bin are closed (inclusive) on the middle bin's
    side, so a value exactly on either boundary never falls into an outer
    bin. Only strictly-less-than-lo or strictly-greater-than-hi reach bins 0/2.
    """
    if v < lo:
        return 0
    if v <= hi:
        return 1
    return 2


# --- category-difficulty prior / tercile map ------------------------------

def _prior_rates_from_fx_episodes():
    """Per-canonical-category (success-rate-weighted-mean, total_n) from
    fx_episodes.json's "cats" list, or None if the file/field is absent. Merges
    alias pairs (e.g. jug_wide_opening -> jug) via an n-weighted average of
    their "sr" values -- see module docstring for why "sr" itself is trusted
    unreconciled."""
    if not config.FX_EPISODES_JSON.exists():
        return None
    d = json.load(open(config.FX_EPISODES_JSON))
    cats = d.get("cats")
    if not cats:
        return None
    merged = {}  # canonical category -> [sr*n running sum, n running sum]
    for c in cats:
        if "sr" not in c or "n" not in c:
            return None
        cat = categories.canonical_category(c["name"])
        n = c["n"]
        succ = c["sr"] * n
        if cat in merged:
            merged[cat][0] += succ
            merged[cat][1] += n
        else:
            merged[cat] = [succ, n]
    return {cat: s / n for cat, (s, n) in merged.items()}


def _prior_rates_from_pooled_csv():
    """Fallback per-canonical-category mean(success) from
    config.POOLED_EPISODES_CSV's object_category/success columns."""
    df = pd.read_csv(config.POOLED_EPISODES_CSV)
    df["category"] = df["object_category"].map(categories.canonical_category)
    return df.groupby("category")["success"].mean().to_dict()


def _load_prior_category_rates():
    """Dispatcher: fx_episodes.json's per-category "sr" if available, else the
    pooled_episodes.csv fallback. Both return {canonical_category: rate}."""
    rates = _prior_rates_from_fx_episodes()
    if rates is None:
        rates = _prior_rates_from_pooled_csv()
    return rates


def build_tercile_map(write=True):
    """Compute the category-difficulty tercile map from the prior per-category
    rate table (`_load_prior_category_rates`): sort categories ascending by
    rate (ties broken by category name, for a fully deterministic split), then
    split into 3 contiguous groups via `numpy.array_split` (as equal-sized as
    the category count allows). tercile 0 = hardest (lowest prior success
    rate) third, tercile 2 = easiest (highest) third.

    If write (default True), freezes the map to config.DIAG_TERCILE_MAP_JSON
    (bandit_v1/ledger/diag_tercile_map.json) so `load_tercile_map` and every
    later re-run/resume of `select_conditions` see the identical map,
    regardless of whether the source files change afterward.
    """
    rates = _load_prior_category_rates()
    cats_sorted = sorted(rates.keys(), key=lambda c: (rates[c], c))
    groups = np.array_split(np.asarray(cats_sorted, dtype=object), N_TERCILES)
    tercile_map = {}
    for tercile_idx, group in enumerate(groups):
        for cat in group:
            tercile_map[str(cat)] = int(tercile_idx)

    if write:
        config.DIAG_TERCILE_MAP_JSON.parent.mkdir(parents=True, exist_ok=True)
        tmp = config.DIAG_TERCILE_MAP_JSON.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(tercile_map, indent=2, sort_keys=True))
        tmp.replace(config.DIAG_TERCILE_MAP_JSON)

    return tercile_map


def load_tercile_map():
    """Load the frozen tercile map from config.DIAG_TERCILE_MAP_JSON, building
    (and writing) it first if it doesn't exist yet. Prefer this over calling
    `build_tercile_map` directly so a re-run always sees the SAME map even if
    the underlying prior-rate source files were later regenerated."""
    if not config.DIAG_TERCILE_MAP_JSON.exists():
        return build_tercile_map(write=True)
    return json.loads(config.DIAG_TERCILE_MAP_JSON.read_text())


# --- pure grid assignment --------------------------------------------------

def assign_cell(features, tercile_map):
    """(tercile, hbin, xbin) for one `states.start_features()`-shaped dict
    (needs only "category", "h", "x_rel"). `features["category"]` is
    canonicalized via categories.canonical_category before the tercile_map
    lookup (defensive: start_features already canonicalizes, but tercile_map
    keys are always canonical names, so a raw alias name must never be used to
    look them up directly). Categories absent from tercile_map (after
    canonicalization) default to tercile 1 (middle) -- the prior table's
    coverage is not exhaustive over every category the live env can reach (see
    module docstring: "pot" is a real example). hbin/xbin come from `_bin3`
    against H_BINS/X_BINS (xbin uses |x_rel|, not the signed value, since the
    grid dimension is offset magnitude, not side)."""
    cat = categories.canonical_category(features["category"])
    tercile = tercile_map.get(cat, 1)
    hbin = _bin3(features["h"], *H_BINS)
    xbin = _bin3(abs(features["x_rel"]), *X_BINS)
    return (tercile, hbin, xbin)


def _cell_label(cell):
    """Parquet/log-friendly string for a (tercile, hbin, xbin) tuple, e.g.
    (0, 1, 2) -> "t0h1x2". Kept as a single string column (not a nested
    list/tuple) so diag_conditions.parquet's "cell" column is a plain,
    groupby-able categorical value."""
    t, h, x = cell
    return f"t{t}h{h}x{x}"


def _nearest_undersubscribed(cell, counts, target, all_cells=ALL_CELLS):
    """Among cells with counts[c] < target, the "nearest" one to `cell` under
    the brief's rule ("same tercile first, then adjacent bins"): rank
    candidates by (tercile distance, hbin+xbin Manhattan distance, cell tuple)
    ascending, lexicographically -- i.e. minimize |Δtercile| FIRST (so any
    same-tercile undersubscribed cell always beats any different-tercile one,
    no matter how far apart in h/x), then within the smallest tercile-distance
    tier minimize |Δhbin|+|Δxbin|, then break remaining ties by the cells'
    natural (tercile, hbin, xbin) sort order for full determinism. Returns
    None only if no cell is undersubscribed (can't happen mid-scan while
    len(rows) < n, since sum(target over 27 cells) = 27*ceil(n/27) > n by
    construction -- see select_conditions)."""
    t0, h0, x0 = cell
    candidates = [c for c in all_cells if counts.get(c, 0) < target]
    if not candidates:
        return None

    def key(c):
        t, h, x = c
        return (abs(t - t0), abs(h - h0) + abs(x - x0), c)

    return min(candidates, key=key)


# --- capture loop -----------------------------------------------------------

def select_conditions(n=None, seed_base=None, out_dir=None, max_scan=None,
                       write_ledger=True, tercile_map=None, log=print):
    """Scan env seeds seed_base, seed_base+1, ... capturing each candidate via
    `states.capture_start` into `out_dir/start_%05d` (the scan index, zero
    padded to 5 digits -- NOT the accepted count, so re-running resumes
    correctly, see below), keeping a subset that fills a balanced 27-cell grid
    (category-difficulty tercile x height bin x |x_rel| bin -- see
    `assign_cell`) up to `n` total kept conditions.

    Per-cell target = ceil(n / 27) (12 for n=300); since 27*12=324 > 300, not
    every cell reaches the target -- the loop simply stops as soon as `n`
    total conditions are kept, so under-filled cells are an expected, logged
    outcome (the brief's own "hard cells may be sparse" note), not a bug.

    Fill rule: a freshly-captured candidate is kept in its OWN natural cell
    whenever that cell's count is still below target (this naturally
    round-robins across cells over the course of the scan, since consecutive
    seeds land in essentially arbitrary cells -- there is no explicit
    per-cell iteration order). Once the scan has covered >= max_scan
    candidates (default None -> ceil(2000 * n / config.N_DIAG), i.e. the
    brief's literal 2000 at n=config.N_DIAG=300, scaled down proportionally
    for a smaller `n` such as the --out_check dry-run's n=20 -- see below for
    why a smaller n needs its own, smaller threshold) AND a candidate's
    natural cell is already at target, it is instead redirected into the
    NEAREST undersubscribed cell (`_nearest_undersubscribed`: same tercile
    first, then smallest h/x distance) rather than discarded -- this is the
    brief's overflow rule, and lets rare/hard natural cells (which may never
    naturally fill from scanning alone) still receive their share of the
    total via reassignment of surplus captures from oversubscribed cells.
    Before max_scan, a full-cell candidate is simply discarded (not counted,
    not written to the ledger) -- its start_%05d directory is left on disk
    (capture is already-paid-for work; skipping recapture is exactly what
    makes resuming cheap).

    Why max_scan scales with n: empirically (see task-6-report.md), this
    task's live object/placement distribution is heavily skewed across the
    27 cells -- e.g. h lands in bin 0 (h<0.08) ~66% of the time and only
    ~6% in bin 2 (h>0.212), so corner cells (tall + far-offset, etc.) are
    genuinely rare and a strict per-cell target of 1 (n=20's
    ceil(20/27)=1) can go 100+ scans without a first hit on some of them.
    A fixed max_scan=2000 regardless of n would make a small-n dry run
    scan just as long as the real n=300 run before overflow ever kicks in
    to close the gap -- scaling by n/config.N_DIAG keeps the "how much
    overscanning are we willing to tolerate before forcing balance"
    ratio constant (2000/300 ~= 6.7x) instead of constant in absolute
    seed count.

    Unbinnable candidates (features["h"] is None -- a category absent from
    FX_POOL_JSON's h/w table, e.g. the untracked "pot" category, see module
    docstring) are always discarded regardless of scan position: there is no
    cell to assign them to, so they are never candidates for overflow either.

    Resumable: for each scan index i, if `out_dir/start_{i:05d}` already
    exists (has a fingerprint.json), its features are re-read via
    `states.start_features` instead of re-capturing (capture_start is the
    expensive ~10-20s step; feature parsing from already-written JSON is
    not) -- so a re-run after an interruption picks up exactly where the
    previous run left off, replaying the same accept/discard/overflow
    decisions deterministically (counts are rebuilt from scratch each run by
    re-scanning from i=0, not persisted separately).

    Writes `config.LEDGER_DIR/diag_conditions.parquet` (columns: start_id,
    seed, every `states.start_features` field, cell) only once, at the end,
    with all `n` kept rows -- iff write_ledger (default True; the
    --out_check dry-run passes write_ledger=False so it never touches the
    real ledger table). Logs (via `log`, default `print`) ONE line per
    scanned candidate (seed, start_id, category, resulting cell, kept/
    discarded, running kept-count) as it happens -- not just a final summary
    -- specifically so a long nohup'd run's log file grows continuously and
    `tail -f`/`grep` can confirm early progress without waiting for
    completion (captures are the expensive ~seconds-per-step here; a
    same-process df.to_parquet only happens once, at the very end). Also
    logs the realized per-cell histogram before returning -- per the brief,
    a sparse-cell histogram is itself a finding, not just a debugging aid.

    Returns the kept-conditions DataFrame (also the frame written to the
    ledger, when write_ledger).
    """
    n = config.N_DIAG if n is None else n
    seed_base = config.DIAG_ENV_SEED_BASE if seed_base is None else seed_base
    out_dir = Path(config.DIAG_DIR if out_dir is None else out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if max_scan is None:
        max_scan = max(1, math.ceil(2000 * n / config.N_DIAG))

    if tercile_map is None:
        tercile_map = load_tercile_map()

    target_per_cell = math.ceil(n / N_CELLS)
    counts = {c: 0 for c in ALL_CELLS}
    rows = []

    i = 0
    n_unbinnable = 0
    # Safety valve only -- see _nearest_undersubscribed's docstring for why
    # this should never actually trigger before `rows` reaches `n`.
    hard_scan_cap = max_scan * 10
    while len(rows) < n and i < hard_scan_cap:
        seed = seed_base + i
        start_dir = out_dir / f"start_{i:05d}"
        if start_dir.exists() and (start_dir / "fingerprint.json").exists():
            feats = states.start_features(start_dir)
        else:
            states.capture_start(seed, start_dir)
            feats = states.start_features(start_dir)

        if feats["h"] is None:
            n_unbinnable += 1
            log(f"[{i:05d}] seed={seed} start_id={start_dir.name} "
                f"category={feats['category']} UNBINNABLE (no h/w) -- "
                f"discarded, kept={len(rows)}/{n}")
            i += 1
            continue

        cell = assign_cell(feats, tercile_map)
        accept = False
        overflowed = False
        if counts[cell] < target_per_cell:
            accept = True
        elif i >= max_scan:
            nearest = _nearest_undersubscribed(cell, counts, target_per_cell)
            if nearest is not None:
                cell = nearest
                accept = True
                overflowed = True

        if accept:
            counts[cell] += 1
            rows.append({
                "start_id": start_dir.name,
                "seed": seed,
                **feats,
                "cell": _cell_label(cell),
            })
        tag = "KEPT(overflow)" if overflowed else ("KEPT" if accept else "discarded")
        log(f"[{i:05d}] seed={seed} start_id={start_dir.name} "
            f"category={feats['category']} cell={_cell_label(cell)} "
            f"{tag}, kept={len(rows)}/{n}")
        i += 1

    df = pd.DataFrame(rows)

    hist = {_cell_label(c): counts[c] for c in ALL_CELLS if counts[c] > 0}
    log(f"select_conditions: kept {len(df)}/{n} conditions from {i} scanned "
        f"seeds ({n_unbinnable} unbinnable, target/cell={target_per_cell}, "
        f"{sum(1 for c in ALL_CELLS if counts[c] > 0)}/{N_CELLS} cells "
        f"nonempty, min/max nonempty cell = "
        f"{min(hist.values()) if hist else 0}/{max(hist.values()) if hist else 0})")
    log(f"select_conditions: per-cell histogram: {hist}")

    if write_ledger:
        config.LEDGER_DIR.mkdir(parents=True, exist_ok=True)
        tmp = (config.LEDGER_DIR / "diag_conditions.parquet").with_suffix(".tmp.parquet")
        df.to_parquet(tmp, index=False)
        tmp.replace(config.LEDGER_DIR / "diag_conditions.parquet")

    return df


def _main():
    # Reconfigure stdout to line-buffer regardless of how this is invoked
    # (nohup'd to a file, piped through tee, etc. all fully-buffer a
    # non-tty stdout by default in CPython, which would otherwise delay
    # every log line -- including the per-capture progress lines -- until
    # process exit or an internal buffer flush).
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass  # Python < 3.7, not a concern in this repo's env

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=config.N_DIAG)
    ap.add_argument("--out_check", action="store_true",
                     help="use config.DIAG_CHECK_DIR (a scratch dir) instead of "
                          "config.DIAG_DIR, and never write the real ledger "
                          "table -- for the dry-run sanity check only.")
    ap.add_argument("--seed_base", type=int, default=config.DIAG_ENV_SEED_BASE)
    ap.add_argument("--max_scan", type=int, default=None,
                     help="default: ceil(2000 * n / config.N_DIAG) -- see "
                          "select_conditions's docstring for why this scales "
                          "with n instead of always being the brief's literal "
                          "2000.")
    args = ap.parse_args()

    out_dir = config.DIAG_CHECK_DIR if args.out_check else config.DIAG_DIR
    select_conditions(n=args.n, seed_base=args.seed_base, out_dir=out_dir,
                       max_scan=args.max_scan, write_ledger=not args.out_check)


if __name__ == "__main__":
    _main()
