"""Append-only parquet ledger. All analyses read ONLY from here."""
import hashlib
from pathlib import Path
import pandas as pd
from .config import LEDGER_DIR as _LEDGER_DIR

LEDGER_DIR = _LEDGER_DIR

def _path(table: str) -> Path:
    return Path(LEDGER_DIR) / f"{table}.parquet"

def append_rows(table: str, rows: list) -> None:
    Path(LEDGER_DIR).mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    p = _path(table)
    if p.exists():
        new = pd.concat([pd.read_parquet(p), new], ignore_index=True)
    tmp = p.with_suffix(".tmp.parquet")
    new.to_parquet(tmp, index=False)
    tmp.replace(p)                       # atomic on same fs

def read(table: str) -> pd.DataFrame:
    return pd.read_parquet(_path(table))

def append_rows_to_path(path, rows: list) -> None:
    """Same atomic (tmp-then-replace) read-modify-write behavior as
    append_rows, but targeting an arbitrary `path` instead of a
    LEDGER_DIR/<table>.parquet table path -- append_rows itself is completely
    untouched (still the same whole-"episodes"-table semantics, still not
    concurrency-safe, exactly as before). This is bandit_v1's parallel-rollout
    seam (parallel_eval.py): each worker subprocess calls this against its OWN
    shard file (bandit_v1/ledger/shards/<run-tag>_<worker>.parquet), so N
    workers never touch the same file at once -- only the parent, after every
    worker has exited, merges shards into "episodes" via a single append_rows
    call. A no-op (leaves no file) when `rows` is empty, so a worker that
    completes zero episodes before crashing never creates an empty shard
    file -- merge_shards below simply has nothing to read for it."""
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
