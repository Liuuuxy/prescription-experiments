"""Append-only parquet ledger. All analyses read ONLY from here."""
import fcntl
import hashlib
import os
import time
from pathlib import Path
import pandas as pd
from .config import LEDGER_DIR as _LEDGER_DIR

LEDGER_DIR = _LEDGER_DIR

# Cross-process append safety (task-ledgerlock-report.md): tonight's incident
# was two writers -- an external bulk-append process and this runner's own
# per-episode append loop -- both doing read-concat-write-tmp-rename against
# the SAME table at the SAME time. Two failure modes from that: (1) a row
# committed between the OTHER writer's read and its own write is silently
# lost (lost-update: the later writer's read-then-concat never saw it), and
# (2) both writers shared one fixed tmp filename, so one writer's
# `tmp.replace(p)` could make the other's tmp vanish out from under it mid-
# write, surfacing as a raw FileNotFoundError. append_rows below closes (1)
# with a real cross-process mutex (fcntl.flock on a sibling `<table>.lock`
# file, held across the entire read-concat-write-rename) and closes (2) as
# belt-and-braces by giving every writer's tmp file its own pid so two
# writers -- even a rogue one that somehow bypasses the lock -- can never
# target the same tmp path.
#
# flock is advisory and per-(inode, open-file-description): it serializes
# every process that goes through THIS function, which is the only writer
# this whole ledger package exposes for a shared table (append_rows_to_path,
# below, is a deliberately separate contract -- see its own docstring).
#
# LOCK_TIMEOUT_S bounds the wait: a lock that is never released (a genuinely
# dead holder, a stuck process) must not hang the caller -- a live runner or
# eval process -- forever. Implemented as LOCK_NB + a backing-off retry loop
# rather than SIGALRM: this function is called from inside worker/runner
# processes that may already use signals or run on non-main threads, where
# SIGALRM either can't be delivered or would stomp on someone else's handler.
LOCK_TIMEOUT_S = 120.0
_LOCK_POLL_START_S = 0.02
_LOCK_POLL_MAX_S = 1.0

def _path(table: str) -> Path:
    return Path(LEDGER_DIR) / f"{table}.parquet"

def _lock_path(table: str) -> Path:
    return Path(LEDGER_DIR) / f"{table}.lock"

class _TableLock:
    """Cross-process exclusive lock on LEDGER_DIR/<table>.lock, held for the
    whole read-concat-write-rename in append_rows. `timeout`/`poll_start`
    default to the module-level LOCK_TIMEOUT_S/_LOCK_POLL_START_S constants
    -- read at __init__ time (not frozen as a function-default at import
    time), so a test's `monkeypatch.setattr(ledger, "LOCK_TIMEOUT_S", ...)`
    is honored by every subsequent append_rows call, not just ones after a
    module reload.

    On timeout, raises TimeoutError naming the table and a best-effort
    "holder hint" (whatever the current lock file contains -- the pid/host
    of whoever last acquired it, written the moment they acquired it, below)
    -- loud and named, never a silent indefinite hang."""

    def __init__(self, table: str, timeout: float = None, poll_start: float = None):
        self.table = table
        self.timeout = LOCK_TIMEOUT_S if timeout is None else timeout
        self.poll = _LOCK_POLL_START_S if poll_start is None else poll_start
        self._fh = None

    def __enter__(self):
        path = _lock_path(self.table)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a+")
        deadline = time.monotonic() + self.timeout
        poll = self.poll
        while True:
            try:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    holder = self._holder_hint()
                    self._fh.close()
                    self._fh = None
                    raise TimeoutError(
                        f"ledger.append_rows: timed out after {self.timeout}s waiting for "
                        f"the '{self.table}' table lock ({path}) -- currently held by "
                        f"{holder}. A live lock always means a live writer (flock releases "
                        "automatically on process exit/crash), so do not remove the lock "
                        "file without first confirming the holder is actually dead."
                    )
                time.sleep(poll)
                poll = min(poll * 1.5, _LOCK_POLL_MAX_S)
        # Record ourselves as the current holder so the NEXT waiter (if any)
        # gets a useful hint instead of stale/empty content.
        try:
            self._fh.seek(0)
            self._fh.truncate()
            self._fh.write(f"pid={os.getpid()} host={os.uname().nodename} "
                            f"acquired={time.time():.0f}\n")
            self._fh.flush()
        except OSError:
            pass  # best-effort hint only -- never fail the lock over this
        return self

    def _holder_hint(self) -> str:
        try:
            self._fh.seek(0)
            content = self._fh.read().strip()
            return content if content else "(unknown -- lock file empty)"
        except OSError:
            return "(unknown -- could not read lock file)"

    def __exit__(self, exc_type, exc, tb):
        if self._fh is not None:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None
        return False

def append_rows(table: str, rows: list) -> None:
    Path(LEDGER_DIR).mkdir(parents=True, exist_ok=True)
    with _TableLock(table):
        new = pd.DataFrame(rows)
        p = _path(table)
        if p.exists():
            new = pd.concat([pd.read_parquet(p), new], ignore_index=True)
        tmp = p.parent / f"{p.stem}.tmp.{os.getpid()}.parquet"  # unique per writer (belt-and-braces)
        new.to_parquet(tmp, index=False)
        tmp.replace(p)                       # atomic on same fs

def read(table: str) -> pd.DataFrame:
    return pd.read_parquet(_path(table))

def append_rows_to_path(path, rows: list) -> None:
    """Same atomic (tmp-then-replace) read-modify-write behavior append_rows
    had before it grew a cross-process lock (task-ledgerlock-report.md), but
    targeting an arbitrary `path` instead of a LEDGER_DIR/<table>.parquet
    table path, and deliberately NOT locked itself: this is bandit_v1's
    parallel-rollout seam (parallel_eval.py), where each worker subprocess
    calls this against its OWN shard file (bandit_v1/ledger/shards/<run-tag>_
    <worker>.parquet) -- N workers never touch the same file at once by
    construction (distinct paths, one per worker), so there is no shared
    state here for a lock to protect. Only the parent, after every worker has
    exited, merges shards into "episodes" via a single append_rows call,
    which IS now lock-protected. A no-op (leaves no file) when `rows` is
    empty, so a worker that completes zero episodes before crashing never
    creates an empty shard file -- merge_shards below simply has nothing to
    read for it."""
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if path.exists():
        new = pd.concat([pd.read_parquet(path), new], ignore_index=True)
    tmp = path.with_suffix(".tmp.parquet")
    new.to_parquet(tmp, index=False)
    tmp.replace(path)                       # atomic on same fs

def read_path(path) -> pd.DataFrame:
    """Read an arbitrary parquet path (not a LEDGER_DIR/<table>.parquet table
    name) -- the read half of the append_rows_to_path shard seam above."""
    return pd.read_parquet(Path(path))

def file_hash(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
