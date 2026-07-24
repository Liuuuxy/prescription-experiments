"""Diagnosis-batch chunked/resumable rollout driver (bandit_v1 Task 7).

Paired with run_diagnosis.sh, which owns the wait-for-preconditions loop and
serving pi_0 (scripts/serve_policy.py); once the server is confirmed up, the
shell script invokes this module once (`python -m bandit_v1.run_diagnosis
--port 8124`) to drive all 300 starts x config.M_DIAG repeats = 2,400 episodes
of ledger table "episodes" (phase="diag", policy_id="pi0"), in chunks of 25
starts at a time, through the shared bandit_v1.rollout.run() engine.

Resume model: `done_pairs()` re-queries the ledger fresh immediately before
EVERY chunk (not just once at process startup), so:
  - a rerun of this module (or of run_diagnosis.sh as a whole, e.g. after a
    crash or a preempted GPU) only ever redoes the still-missing
    (start_id, repeat_idx) pairs of whichever chunk was interrupted -- never a
    whole-chunk replay, and never a completed episode;
  - within a single run, a chunk fully satisfied by rows an EARLIER chunk in
    this same process happened to also write (shouldn't normally happen since
    chunks partition start_ids, but is harmless either way) is detected and
    skipped without calling rollout.run at all.
`rollout.run`'s own per-episode ledger append (already crash-safe -- see
rollout.py) is what makes each individual chunk itself safely interruptible
mid-way, not just chunk-to-chunk; `skip_pairs` (rollout.py's new optional
parameter, added for this task) is what lets a chunk resume at per-episode
granularity instead of being forced to redo from its own start.

Sanity guard (brief's Step 5): `load_ordered_start_ids` asserts
diag_conditions.parquet has EXACTLY config.N_DIAG (300) rows, all with
distinct start_id values, before this module ever calls rollout.run -- the one
hard gate against an incomplete or corrupted condition set silently driving a
partial diagnosis batch. start_id order is the parquet's own row order (== the
order select_conditions kept them in), per the brief's "order start_ids from
the parquet". The parquet's start_id column is the sole authority for which
300 of bandit_v1/states/diag/'s directories to roll out -- that directory also
holds discarded scanned-but-not-kept captures (see task-6-report.md), which
this module never touches because it only ever builds start_dirs from
start_ids read out of the parquet.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from . import config, ledger, rollout

PHASE = "diag"
POLICY_ID = "pi0"
CHUNK_SIZE = 25
DEFAULT_PORT = 8124


def load_ordered_start_ids(parquet_path=None):
    """Ordered list of start_id strings from diag_conditions.parquet (row
    order == select_conditions's keep order). Asserts exactly config.N_DIAG
    rows with config.N_DIAG distinct start_id values -- see module docstring's
    "Sanity guard" section for why this is a hard precondition, not a
    warning."""
    p = Path(parquet_path) if parquet_path is not None else (config.LEDGER_DIR / "diag_conditions.parquet")
    df = pd.read_parquet(p)
    ids = df["start_id"].tolist()
    n_unique = df["start_id"].nunique()
    assert len(ids) == config.N_DIAG and n_unique == config.N_DIAG, (
        f"{p} sanity check failed: {len(ids)} rows, {n_unique} unique "
        f"start_ids (expected exactly {config.N_DIAG} of each)")
    return ids


def chunked(seq, size):
    """Split `seq` into consecutive lists of length <= size, in order (last
    chunk may be shorter)."""
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def done_pairs(phase=PHASE, policy_id=POLICY_ID):
    """Set of (start_id, repeat_idx) tuples already present in ledger table
    "episodes" for this (phase, policy_id), re-read fresh from disk on every
    call (see module docstring for why staleness would break the resume
    guarantee). Empty set if the ledger table doesn't exist yet (first-ever
    invocation, before any diag episode has been written)."""
    try:
        df = ledger.read("episodes")
    except FileNotFoundError:
        return set()
    d = df[(df["phase"] == phase) & (df["policy_id"] == policy_id)]
    return set(zip(d["start_id"], d["repeat_idx"]))


def run_all(host, port, repeats=None, chunk_size=CHUNK_SIZE, log=print):
    """Full resumable diagnosis batch: load+sanity-check the 300 start_ids
    (`load_ordered_start_ids`), split into chunks of `chunk_size` (default 25),
    and for each chunk call `rollout.run(..., phase="diag", policy_id="pi0",
    skip_pairs=done_pairs())`. A chunk whose every (start_id, repeat_idx) pair
    is already in the ledger is skipped without calling rollout.run (no env/
    client construction wasted on a no-op chunk). Logs one line per chunk
    (already-done count, new episodes run, success count, elapsed, running
    ledger total for this phase/policy_id) plus a final summary line -- so a
    long nohup'd run's log grows continuously and progress is checkable via
    `tail -f`/`grep` without waiting for completion, per this task's brief."""
    repeats = config.M_DIAG if repeats is None else repeats
    start_ids = load_ordered_start_ids()
    chunks = chunked(start_ids, chunk_size)
    log(f"run_all: {len(start_ids)} start_ids in {len(chunks)} chunks of "
        f"<= {chunk_size}, repeats={repeats} phase={PHASE} policy_id={POLICY_ID}")

    for ci, chunk_ids in enumerate(chunks):
        chunk_t0 = time.time()
        done = done_pairs()
        total_pairs = len(chunk_ids) * repeats
        n_already = sum(1 for sid in chunk_ids for r in range(repeats) if (sid, r) in done)
        if n_already >= total_pairs:
            log(f"chunk {ci + 1}/{len(chunks)}: {n_already}/{total_pairs} pairs "
                f"already in ledger -- skipping entirely (0 new episodes)")
            continue

        start_dirs = [config.DIAG_DIR / sid for sid in chunk_ids]
        rows = rollout.run(host, port, start_dirs, repeats, phase=PHASE,
                            policy_id=POLICY_ID, skip_pairs=done)
        elapsed = round(time.time() - chunk_t0, 1)
        n_success = sum(1 for r in rows if r["success"])
        d = ledger.read("episodes")
        cum_total = len(d[(d["phase"] == PHASE) & (d["policy_id"] == POLICY_ID)])
        log(f"chunk {ci + 1}/{len(chunks)}: {len(chunk_ids)} starts, "
            f"{n_already}/{total_pairs} already done, ran {len(rows)} new "
            f"episodes ({n_success}/{len(rows)} success), {elapsed}s, "
            f"cumulative diag rows in ledger={cum_total}")

    d = ledger.read("episodes")
    final_total = len(d[(d["phase"] == PHASE) & (d["policy_id"] == POLICY_ID)])
    final_sr = d[(d["phase"] == PHASE) & (d["policy_id"] == POLICY_ID)]["success"].mean()
    log(f"run_all: all {len(chunks)} chunks processed -- "
        f"{final_total} total diag/pi0 rows, success_rate={final_sr:.4f}")


def _main():
    # Line-buffer stdout regardless of invocation (nohup'd to a file): a
    # non-tty stdout is fully-buffered by default in CPython, which would
    # otherwise delay every per-chunk progress line until process exit --
    # same fix as diagnosis.py's _main, same reason.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--repeats", type=int, default=config.M_DIAG)
    ap.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    args = ap.parse_args()

    run_all(args.host, args.port, repeats=args.repeats, chunk_size=args.chunk_size)


if __name__ == "__main__":
    _main()
