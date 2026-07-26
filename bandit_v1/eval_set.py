"""Eval set E + baseline-eval aggregation (bandit_v1 Task 9).

Design: weakregion/BANDIT_V1_DESIGN.md section 1 item 4 ("Eval set E"), item 5
("Baseline b"), item 6 ("Noise floor sigma_e"). Brief: .superpowers/sdd/
task-9-brief.md.

Two halves:
  1. **Eval set E construction** (`stratify`, `write_manifest`, `build_E`):
     scan fresh env seeds from config.E_ENV_SEED_BASE (500000 -- disjoint from
     every prior eval/diagnosis seed base per the design doc: 0, 1-137,
     100000, 300000, 600000), capture `n_candidates` (default 300) saved
     starts (states.capture_start, same env-touching mechanics as
     diagnosis.py's condition capture -- no GPU/policy server needed), score
     each candidate's p_hat_0 via a fitted `map_fit.MapModels.predict_p`
     (features = ONLY [category, h, w, x_rel, y_rel, side], per map_fit's
     commensurability contract), keep `config.N_E` (150) of them stratified
     ~50/50/50 by p_hat_0 TERCILE OF THE CANDIDATE POOL (design: "strata are
     relative to p_hat_0" -- i.e. computed from whatever ~300 candidates were
     actually scanned, not a fixed absolute p_hat threshold), DELETE every
     non-selected candidate's capture directory, and freeze+hash the manifest
     (ledger/E_manifest.parquet, ledger/hashes.json).

  2. **Baseline-eval aggregation** (`eval_checkpoint`, `_aggregate_eval_rows`,
     `per_start_flip_table`, `append_baseline_to_config_yaml`): run a served
     policy checkpoint over the frozen 150-start manifest, `repeats` times,
     through the SAME shared bandit_v1.rollout.run engine every other phase
     (diag/pull) already uses (phase="eval") -- then reduce the raw per-
     episode rows into the per-repeat / per-stratum means pull.run_pull's
     `eval_fn` seam expects. See pull.py's module docstring + `compute_delta`
     for the exact contract `eval_checkpoint`'s return dict is built to match
     byte-for-byte (`per_repeat_means`: list[float] len==repeats;
     `per_stratum_means`: {stratum: list[float] len==repeats}) -- plus a few
     convenience keys (`mean`, `per_stratum_per_repeat`, `per_stratum_mean`)
     pull.compute_delta simply ignores.

`eval_checkpoint`'s signature is `(policy_port, policy_id, arm, pull_id,
repeats=...)`: `policy_id` and `pull_id` are explicit, SEPARATE parameters --
never one value silently reused for both -- resolving the doubling flagged in
pull.py's own `run_pull` (task-12-report.md's "Concerns": it currently calls
`eval_fn(port, pull_id, arm, pull_id)`, passing `pull_id` for both slots since
nothing else was available to give `policy_id` at the time pull.py was
written). A caller with a genuinely distinct policy identity (e.g.
run_baseline.sh's "pi0_baseline", which is not itself a pull_id) can now pass
it through cleanly; pull.py's existing call site is unaffected by this (still
correct -- policy_id happening to equal pull_id for an actual bandit pull is a
reasonable choice, not a bug this module needs to fix at that call site).

Everything in this module is CPU/filesystem-only except the two things this
task's brief explicitly defers to a later, human-reviewed run: `build_E`'s
capture loop needs a live robocasa env (states.capture_start -- no GPU/policy
server, same as diagnosis.py) and `eval_checkpoint`'s rollout.run call needs a
served policy (GPU + openpi_client websocket). Every test in
test_eval_set.py monkeypatches both away; the real build + baseline eval are
DEFERRED to `run_baseline.sh`, which is NOT launched by this task (see its own
docstring for why -- it requires Task 7's diagnosis batch to finish and its
results to be reviewed first).
"""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import config, ledger, rollout, states

STRATA = ("hard", "mid", "easy")  # ascending p_hat_0 order (design doc's "easy/mid/hard")
EVAL_PHASE = "eval"

# The 9 states.start_features() fields carried through into the manifest,
# unchanged (no re-engineering here -- map_fit.predict_features does its own
# feature engineering from these raw columns at score time).
FEATURE_COLUMNS = ["category", "h", "w", "x_rel", "y_rel", "side", "yaw", "layout_id", "style_id"]
MANIFEST_COLUMNS = ["start_id", "seed", *FEATURE_COLUMNS, "p_hat", "stratum"]


# =============================================================================
# 1. Eval set E construction
# =============================================================================

def stratify(cands_df: pd.DataFrame, n=150) -> pd.DataFrame:
    """Select `n` rows of `cands_df` (which must have a "p_hat" column),
    stratified ~50/50/50 by p_hat TERCILE OF `cands_df` ITSELF (design:
    "strata are relative to p_hat_0" -- i.e. computed from whatever candidate
    pool was actually scanned, not a fixed absolute threshold). Returns a
    copy of the selected rows (every original column preserved) with one new
    "stratum" column added, valued in STRATA ("hard" = lowest-p_hat third,
    "mid", "easy" = highest-p_hat third).

    Deterministic under a fixed `cands_df` row order: p_hat is ranked with a
    STABLE (mergesort) ascending sort, so p_hat ties keep the candidates'
    original relative row order rather than an arbitrary sort-algorithm
    tiebreak. Within each tercile group (`numpy.array_split` over the sorted
    rank order -- same "as equal as possible" convention as
    diagnosis.build_tercile_map), `n // 3` candidates are picked at EVENLY
    SPACED rank positions across that group (`numpy.linspace` over the
    group's index positions, rounded to the nearest integer) rather than
    clustered at one edge of the tercile -- so the design's "p_hat ranges
    printed per stratum" (brief Step 3) shows a real spread across each
    tercile, not a single point near its boundary.

    `n` must be evenly divisible by len(STRATA) (3) -- the design's literal
    50/50/50 split, not an approximately-equal one -- raises ValueError
    otherwise. Raises ValueError if `cands_df` has fewer than `n` rows: given
    that precondition, EVERY tercile group is guaranteed (by
    `numpy.array_split`'s "first `m % k` groups get one extra element, the
    rest get exactly `m // k`" rule) to have at least `n // 3` candidates, so
    no separate per-tercile-too-small check is needed or reachable.
    """
    if "p_hat" not in cands_df.columns:
        raise ValueError("stratify: cands_df must have a 'p_hat' column")
    if n % len(STRATA) != 0:
        raise ValueError(
            f"stratify: n={n} must be evenly divisible by {len(STRATA)} "
            f"(the design's literal 50/50/50 split)")
    n_per = n // len(STRATA)
    if len(cands_df) < n:
        raise ValueError(f"stratify: only {len(cands_df)} candidates, need >= {n}")

    p_hat = cands_df["p_hat"].to_numpy(dtype=float)
    order = np.argsort(p_hat, kind="mergesort")  # stable ascending: ties keep original row order
    groups = np.array_split(order, len(STRATA))  # ascending p_hat -> [hard, mid, easy]

    selected = []
    for stratum, idxs in zip(STRATA, groups):
        pick_pos = np.unique(np.round(np.linspace(0, len(idxs) - 1, n_per)).astype(int))
        if len(pick_pos) != n_per:
            raise ValueError(
                f"stratify: evenly-spaced pick for stratum {stratum!r} collapsed to "
                f"{len(pick_pos)} unique rank positions (wanted {n_per} out of "
                f"{len(idxs)} candidates in this tercile) -- group too small "
                f"relative to n_per for a clean spread")
        chosen_idx = idxs[pick_pos]
        sub = cands_df.iloc[chosen_idx].copy()
        sub["stratum"] = stratum
        selected.append(sub)

    return pd.concat(selected, ignore_index=True)


def _delete_dirs(paths) -> None:
    """rmtree every path in `paths` that exists. Used to discard non-selected
    build_E candidate capture directories once stratification has decided
    which 150 to keep -- capture is the expensive step, so this only ever
    deletes already-captured, already-scored-and-rejected work."""
    for p in paths:
        p = Path(p)
        if p.exists():
            shutil.rmtree(p)


def write_manifest(selected_df: pd.DataFrame, path=None) -> Path:
    """Write the frozen E manifest (default config.LEDGER_DIR/E_manifest.parquet)
    with columns exactly MANIFEST_COLUMNS, and merge-update `hashes.json`
    (alongside the manifest, same convention as pool.py's build_pool_table)
    with the written file's sha256 under the key "E_manifest.parquet".
    Atomic write (tmp-then-replace, matching every other bandit_v1 ledger
    writer). `config.LEDGER_DIR` is read fresh at call time (not cached at
    import time), so a test's `monkeypatch.setattr(config, "LEDGER_DIR", ...)`
    is honored -- see ledger.py/pool.py for the same gotcha this avoids.

    Raises ValueError if `selected_df` is missing any MANIFEST_COLUMNS column
    (a wrong/incomplete stratify() output reaching here should surface
    immediately, not silently write a truncated manifest).
    """
    path = Path(config.LEDGER_DIR) / "E_manifest.parquet" if path is None else Path(path)
    missing = [c for c in MANIFEST_COLUMNS if c not in selected_df.columns]
    if missing:
        raise ValueError(f"write_manifest: selected_df missing columns {missing}")

    out = selected_df[MANIFEST_COLUMNS].reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    out.to_parquet(tmp, index=False)
    tmp.replace(path)

    hashes_path = path.parent / "hashes.json"
    hashes = json.load(open(hashes_path)) if hashes_path.exists() else {}
    hashes[path.name] = ledger.file_hash(path)
    tmp_h = hashes_path.with_suffix(".tmp.json")
    with open(tmp_h, "w") as f:
        json.dump(hashes, f, indent=2, sort_keys=True)
    tmp_h.replace(hashes_path)

    return path


def build_E(models, n_candidates=300, n=None, seed_base=None, out_dir=None, log=print) -> pd.DataFrame:
    """Scan fresh env seeds starting at `seed_base` (default
    config.E_ENV_SEED_BASE), capture `n_candidates` (default 300) saved
    starts into `out_dir` (default config.E_DIR) via
    `states.capture_start`/`states.start_features` (resumable: a start_dir
    that already has a fingerprint.json is re-read, not re-captured -- same
    convention as diagnosis.select_conditions), score every binnable
    candidate's p_hat_0 via `models.predict_p`, stratify down to `n` (default
    config.N_E=150) via `stratify` above, DELETE every non-selected
    candidate's capture directory, and write+hash the manifest
    (`write_manifest`).

    Unlike diagnosis.select_conditions's early-stopping balanced-grid scan,
    this always captures the FULL `n_candidates` before stratifying -- the
    whole point of "terciles of the candidate pool" is that they need the
    complete candidate distribution first; there is no meaningful notion of
    "keep as soon as a stratum fills" here.

    Candidates whose category has no h/w in FX_POOL_JSON (states.
    start_features returns h=None) are "unbinnable" for p_hat_0 scoring (no
    numeric feature to score) -- discarded immediately and never written to
    the manifest, same situation as diagnosis.py's identical case; their
    capture directory is deleted along with every other non-selected one.

    Logs one line per scanned candidate (so a long nohup'd run's log grows
    continuously under `tail -f`, matching diagnosis.select_conditions's
    convention) plus a final per-stratum p_hat range summary (brief Step 3's
    "p_hat ranges printed per stratum").

    Returns the `n`-row selected+stratified DataFrame (the same frame written
    to the manifest).
    """
    n = config.N_E if n is None else n
    seed_base = config.E_ENV_SEED_BASE if seed_base is None else seed_base
    out_dir = Path(config.E_DIR if out_dir is None else out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    n_unbinnable = 0
    for i in range(n_candidates):
        seed = seed_base + i
        start_id = f"start_{i:05d}"
        start_dir = out_dir / start_id
        if start_dir.exists() and (start_dir / "fingerprint.json").exists():
            feats = states.start_features(start_dir)
        else:
            states.capture_start(seed, start_dir)
            feats = states.start_features(start_dir)

        if feats["h"] is None:
            n_unbinnable += 1
            log(f"[{i:05d}] seed={seed} start_id={start_id} category={feats['category']} "
                f"UNBINNABLE (no h/w) -- discarded")
            continue

        rows.append({"start_id": start_id, "seed": seed, **feats})
        log(f"[{i:05d}] seed={seed} start_id={start_id} category={feats['category']} captured")

    cands_df = pd.DataFrame(rows)
    if len(cands_df) == 0:
        raise ValueError("build_E: zero binnable candidates captured -- nothing to score/stratify")

    cands_df = cands_df.reset_index(drop=True)
    cands_df["p_hat"] = models.predict_p(cands_df)

    selected = stratify(cands_df, n=n)

    keep_ids = set(selected["start_id"])
    to_delete = [out_dir / f"start_{i:05d}" for i in range(n_candidates)
                 if f"start_{i:05d}" not in keep_ids]
    _delete_dirs(to_delete)
    log(f"build_E: kept {len(selected)}/{len(cands_df)} binnable candidates "
        f"({n_unbinnable} unbinnable discarded out of {n_candidates} scanned, "
        f"{len(to_delete)} non-selected capture dirs deleted)")

    for stratum in STRATA:
        sub = selected[selected["stratum"] == stratum]
        log(f"build_E: stratum={stratum} n={len(sub)} "
            f"p_hat range=[{sub['p_hat'].min():.4f}, {sub['p_hat'].max():.4f}]")

    write_manifest(selected)
    return selected


def load_manifest(path=None) -> pd.DataFrame:
    """Load the frozen E manifest (default config.LEDGER_DIR/E_manifest.parquet,
    read fresh at call time)."""
    path = Path(config.LEDGER_DIR) / "E_manifest.parquet" if path is None else Path(path)
    return pd.read_parquet(path)


# =============================================================================
# 2. Baseline-eval aggregation
# =============================================================================

def eval_checkpoint(policy_port, policy_id, arm, pull_id, repeats=None, *,
                     host="127.0.0.1", manifest=None, manifest_path=None,
                     run_fn=None, workers=None, log=print) -> dict:
    """Evaluate the checkpoint served at (host, policy_port) on the frozen
    150-start eval set E, `repeats` independent passes (default
    config.EVAL_REPEATS=3). This is the ONE function pull.run_pull's
    `eval_fn` seam is meant to be, and the ONE function run_baseline.sh's
    baseline eval calls too -- see this module's docstring for the
    policy_id/pull_id doubling this signature resolves.

    Runs entirely through the shared bandit_v1.rollout.run engine
    (phase="eval"), so every episode lands in ledger table "episodes" exactly
    like every diag/pull episode -- this function does no ledger writing of
    its own beyond what rollout.run already does; it only aggregates the rows
    `run_fn` (default rollout.run) returns.

    `workers` (bandit_v1 rollout-speedup #1, OPT-IN -- default None): None or
    1 leaves the serial `rollout.run` path completely unchanged (the only
    behavior this function had before `workers` existed); an int > 1 instead
    routes through `parallel_eval.run_parallel` with that many worker
    subprocesses, sharding the 150 E-set starts round-robin across them (see
    parallel_eval.py's module docstring for why this preserves the warm-
    restore speedup and how ledger writes stay concurrency-safe). Ignored
    entirely when `run_fn` is explicitly supplied (see below) -- an explicit
    `run_fn` always wins, exactly as before `workers` existed, so every
    existing run_fn-based test is unaffected.

    `manifest`/`manifest_path` let a caller supply an already-loaded manifest
    DataFrame or an alternate path (default: `load_manifest()` against
    config.LEDGER_DIR); `run_fn` is the same kind of injectable seam
    test_rollout.py/test_run_diagnosis.py already use elsewhere in this
    codebase, letting every test here run with no live env, GPU, or policy
    server.

    Returns (see pull.compute_delta's docstring for the two keys it actually
    reads -- kept byte-identical in shape here):
      {"per_repeat_means": [float]*repeats,      # mean success over all 150
                                                   # E-set starts, one entry
                                                   # per repeat
       "mean": float,                             # np.mean(per_repeat_means)
       "per_stratum_means": {stratum: [float]*repeats, ...},  # EXACT shape
                                                   # pull.compute_delta's
                                                   # docstring documents --
                                                   # per-repeat VECTOR (not
                                                   # already reduced), one per
                                                   # stratum in STRATA
       "per_stratum_per_repeat": {stratum: [float]*repeats, ...},  # identical
                                                   # content under an
                                                   # unambiguous name (the
                                                   # key above is a
                                                   # per-repeat vector, which
                                                   # its own "_means" name
                                                   # can mislead)
       "per_stratum_mean": {stratum: float, ...}}  # already-reduced scalar
                                                   # per stratum (mean over
                                                   # repeats) -- what
                                                   # run_baseline.sh records
                                                   # as config.yaml's
                                                   # per-stratum `b`

    Raises ValueError (loud, never a silently-averaged partial slice or a
    silent NaN) if any manifest start_id is missing from -- or any row is
    duplicated within, or any extra start_id appears in -- a given repeat's
    rollout.run rows; also if `manifest` itself has zero starts for one or
    more of STRATA (an entirely-missing stratum). See `_aggregate_eval_rows`.
    """
    repeats = config.EVAL_REPEATS if repeats is None else repeats
    if manifest is None:
        manifest = load_manifest(manifest_path)
    if "start_id" not in manifest.columns or "stratum" not in manifest.columns:
        raise ValueError("eval_checkpoint: manifest must have 'start_id' and 'stratum' columns")
    if run_fn is None:
        if workers is not None and workers > 1:
            from . import parallel_eval as _parallel_eval
            run_fn = lambda *a, **kw: _parallel_eval.run_parallel(*a, workers=workers, **kw)
        else:
            run_fn = rollout.run

    e_dir = config.E_DIR
    start_dirs = [e_dir / sid for sid in manifest["start_id"]]
    result_rows = run_fn(host, policy_port, start_dirs, repeats, phase=EVAL_PHASE,
                          policy_id=policy_id, arm=arm, pull_id=pull_id)

    return _aggregate_eval_rows(result_rows, manifest, repeats)


def _aggregate_eval_rows(result_rows, manifest: pd.DataFrame, repeats: int) -> dict:
    """Pure aggregation: `result_rows` (rollout.run's row list, or any
    DataFrame-constructible equivalent -- e.g. a test's fake) -> the dict
    `eval_checkpoint` returns. Requires EVERY manifest start_id to appear
    EXACTLY ONCE at EVERY repeat_idx in [0, repeats) -- raises ValueError
    naming exactly which repeat and which start_ids are missing/duplicated/
    unexpected, rather than silently averaging over however many rows
    actually showed up (a partial-data average or an empty-slice NaN would
    both hide a real rollout/serving failure instead of surfacing it).

    Also requires every member of STRATA to have >= 1 start in `manifest`
    itself (checked before any per-stratum reduction) -- an entirely-missing
    stratum would otherwise mean() an empty slice into a silent NaN rather
    than surfacing the malformed manifest; raises ValueError naming exactly
    which stratum/strata are missing."""
    df = pd.DataFrame(result_rows)
    if len(df) == 0:
        raise ValueError(
            f"eval_checkpoint: rollout.run returned zero rows for the E-set eval "
            f"(expected {len(manifest)} starts x {repeats} repeats)")
    for col in ("start_id", "repeat_idx", "success"):
        if col not in df.columns:
            raise ValueError(f"eval_checkpoint: rollout.run rows missing required column {col!r}")

    start_to_stratum = dict(zip(manifest["start_id"], manifest["stratum"]))
    expected_starts = set(manifest["start_id"])

    manifest_strata = set(manifest["stratum"].unique())
    missing_strata = [s for s in STRATA if s not in manifest_strata]
    if missing_strata:
        raise ValueError(
            f"eval_checkpoint: manifest has zero starts for stratum/strata "
            f"{missing_strata} (STRATA={STRATA}) -- cannot compute a per-stratum "
            f"mean for an entirely-missing stratum")

    per_repeat_means = []
    per_stratum_per_repeat = {s: [] for s in STRATA}

    for r in range(repeats):
        sub = df[df["repeat_idx"] == r]

        dup = sub.loc[sub["start_id"].duplicated(keep=False), "start_id"].unique()
        if len(dup):
            raise ValueError(
                f"eval_checkpoint: repeat {r} has duplicate rows for start_ids: "
                f"{sorted(dup)[:10]}{'...' if len(dup) > 10 else ''}")

        got_starts = set(sub["start_id"])
        missing = expected_starts - got_starts
        if missing:
            raise ValueError(
                f"eval_checkpoint: repeat {r} is missing {len(missing)}/"
                f"{len(expected_starts)} manifest starts: {sorted(missing)[:10]}"
                f"{'...' if len(missing) > 10 else ''}")
        extra = got_starts - expected_starts
        if extra:
            raise ValueError(
                f"eval_checkpoint: repeat {r} has rows for start_ids not in the "
                f"manifest: {sorted(extra)[:10]}{'...' if len(extra) > 10 else ''}")

        per_repeat_means.append(float(sub["success"].astype(float).mean()))

        sub_strat = sub.assign(stratum=sub["start_id"].map(start_to_stratum))
        for s in STRATA:
            ssub = sub_strat[sub_strat["stratum"] == s]
            per_stratum_per_repeat[s].append(float(ssub["success"].astype(float).mean()))

    per_stratum_mean = {s: float(np.mean(v)) for s, v in per_stratum_per_repeat.items()}

    return {
        "per_repeat_means": per_repeat_means,
        "mean": float(np.mean(per_repeat_means)),
        "per_stratum_means": {s: list(v) for s, v in per_stratum_per_repeat.items()},
        "per_stratum_per_repeat": {s: list(v) for s, v in per_stratum_per_repeat.items()},
        "per_stratum_mean": per_stratum_mean,
    }


def per_start_flip_table(df_eval: pd.DataFrame) -> pd.DataFrame:
    """Per-start flip diagnostic (design doc's noise-floor QA artifact, brief
    Step 4's "per-start flip table"): for each start_id, how many of its
    repeat_idx rows succeeded vs failed, and whether the outcome ever
    DISAGREES across repeats (a "flip" -- e.g. success on one repeat, failure
    on another). `df_eval` is any DataFrame with start_id/success columns
    (typically a `ledger.read("episodes")` slice already filtered to one
    phase/policy_id, e.g. run_baseline.sh's 3-repeat baseline eval).

    Returns one row per start_id: n_repeats, n_success (int), flip (bool,
    True iff n_success is neither 0 nor n_repeats -- i.e. a genuine outcome
    disagreement for that start)."""
    g = df_eval.groupby("start_id")["success"].agg(n_repeats="count", n_success="sum")
    g["n_success"] = g["n_success"].astype(int)
    g["flip"] = (g["n_success"] > 0) & (g["n_success"] < g["n_repeats"])
    return g.reset_index()


def append_baseline_to_config_yaml(result: dict, path=None, *, checkpoint_id=None) -> Path:
    """Append a `baseline:` block (b, per_stratum_b, sigma_e_eval,
    per_repeat_means, repeats, checkpoint_id, written_at) to config.yaml.

    Deliberately APPENDS plain text rather than doing a
    yaml.safe_load-then-dump round-trip of the whole file: config.yaml is a
    human-authored, richly commented file (see bandit_v1/config.py's own
    "frozen constants ... never edit mid-run" convention, and Task 5/7's
    additions) -- a generic YAML dumper re-serializing the ENTIRE file would
    silently discard every comment. Only the NEW block is built via
    `yaml.safe_dump` (for correct syntax/indentation/quoting), then appended
    after a short comment header; every byte already in the file is
    untouched.

    `sigma_e_eval` = sample std (ddof=1) of `result["per_repeat_means"]` --
    the design's "std of the 3 repeat means" noise-floor measurement (0.0 for
    a degenerate single-repeat call, where a sample std is undefined).
    """
    path = Path(config.LEDGER_DIR) / "config.yaml" if path is None else Path(path)
    per_repeat = [float(x) for x in result["per_repeat_means"]]
    sigma_e_eval = float(np.std(per_repeat, ddof=1)) if len(per_repeat) > 1 else 0.0

    block = {
        "baseline": {
            "b": float(result["mean"]),
            "per_stratum_b": {s: float(result["per_stratum_mean"][s]) for s in STRATA},
            "sigma_e_eval": sigma_e_eval,
            "per_repeat_means": per_repeat,
            "repeats": len(per_repeat),
            "checkpoint_id": str(checkpoint_id) if checkpoint_id is not None else None,
            "written_at": datetime.now(timezone.utc).isoformat(),
        }
    }
    header = (
        "\n# bandit_v1 Task 9: pi_0 baseline b on the frozen 150-start eval set E\n"
        "# (config.EVAL_REPEATS independent repeats), written once by\n"
        "# run_baseline.sh -- never edited mid-run, same frozen-constant convention\n"
        "# as the rest of this file (bandit_v1/config.py's docstring).\n"
    )
    dumped = yaml.safe_dump(block, sort_keys=False, default_flow_style=False)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(header)
        f.write(dumped)
    return path


# =============================================================================
# CLI
# =============================================================================

def _main():
    import argparse
    import sys

    # Line-buffer stdout regardless of invocation (nohup'd to a file etc.) --
    # same fix as diagnosis.py/run_diagnosis.py's _main.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    build_p = sub.add_parser("build-e", help="scan+capture+stratify the 150-start eval set E")
    build_p.add_argument("--n_candidates", type=int, default=300)
    build_p.add_argument("--n", type=int, default=config.N_E)

    eval_p = sub.add_parser("eval-baseline", help="serve must already be up; runs eval_checkpoint "
                                                   "and records b/per-stratum-b/sigma_e_eval")
    eval_p.add_argument("--host", default="127.0.0.1")
    eval_p.add_argument("--port", type=int, required=True)
    eval_p.add_argument("--policy_id", default="pi0_baseline")
    eval_p.add_argument("--repeats", type=int, default=config.EVAL_REPEATS)
    eval_p.add_argument("--checkpoint_id", default=None)
    eval_p.add_argument("--workers", type=int, default=None,
                         help="rollout-speedup #1 (opt-in): >1 runs eval_checkpoint's rollout "
                              "over this many parallel worker subprocesses against the same "
                              "served policy (see parallel_eval.py); omitted/1 is the original "
                              "serial rollout.run path, unchanged")

    args = ap.parse_args()

    if args.cmd == "build-e":
        from . import map_fit
        models = map_fit.load()
        build_E(models, n_candidates=args.n_candidates, n=args.n)

    elif args.cmd == "eval-baseline":
        result = eval_checkpoint(args.port, args.policy_id, None, args.policy_id,
                                  repeats=args.repeats, host=args.host, workers=args.workers)
        print("BASELINE_MEAN", result["mean"])
        print("BASELINE_PER_STRATUM", json.dumps(result["per_stratum_mean"]))

        result_path = config.LEDGER_DIR / "baseline_eval_result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2))
        print(f"eval_set: wrote {result_path}")

        cfg_path = append_baseline_to_config_yaml(result, checkpoint_id=args.checkpoint_id)
        print(f"eval_set: appended baseline block to {cfg_path}")

        df_eval = ledger.read("episodes")
        df_eval = df_eval[(df_eval["phase"] == EVAL_PHASE) & (df_eval["policy_id"] == args.policy_id)]
        flips = per_start_flip_table(df_eval)
        n_flips = int(flips["flip"].sum())
        print(f"BASELINE_FLIP_COUNT {n_flips}/{len(flips)} starts flipped across "
              f"{args.repeats} repeats ({(n_flips / len(flips) if len(flips) else float('nan')):.1%})")
        flip_path = config.LEDGER_DIR / "baseline_flip_table.parquet"
        flips.to_parquet(flip_path, index=False)
        print(f"eval_set: wrote {flip_path}")


if __name__ == "__main__":
    _main()
