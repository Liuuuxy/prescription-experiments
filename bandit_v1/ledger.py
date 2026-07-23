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

def file_hash(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
