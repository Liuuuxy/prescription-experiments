import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

from bandit_v1 import config, ledger

def test_append_and_read_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER_DIR", tmp_path)
    ledger.append_rows("episodes", [{"episode_id": "e1", "success": True, "x_rel": 0.1}])
    ledger.append_rows("episodes", [{"episode_id": "e2", "success": False, "x_rel": -0.2}])
    df = ledger.read("episodes")
    assert list(df["episode_id"]) == ["e1", "e2"]
    assert df.shape[0] == 2

def test_file_hash_stable(tmp_path):
    p = tmp_path / "f.txt"; p.write_text("abc")
    assert ledger.file_hash(p) == ledger.file_hash(p)
    assert len(ledger.file_hash(p)) == 64


def test_append_rows_to_path_roundtrip_and_accumulates(tmp_path):
    p = tmp_path / "shards" / "worker0.parquet"
    ledger.append_rows_to_path(p, [{"start_id": "s0", "repeat_idx": 0}])
    ledger.append_rows_to_path(p, [{"start_id": "s0", "repeat_idx": 1}])

    df = ledger.read_path(p)
    assert list(zip(df["start_id"], df["repeat_idx"])) == [("s0", 0), ("s0", 1)]


def test_append_rows_to_path_empty_rows_is_a_noop(tmp_path):
    p = tmp_path / "shards" / "worker0.parquet"
    ledger.append_rows_to_path(p, [])
    assert not p.exists()


def test_append_rows_to_path_does_not_touch_ledger_dir_or_table_files(tmp_path, monkeypatch):
    """append_rows_to_path is a completely separate write path from
    append_rows -- writing to an arbitrary shard path must never create or
    modify anything under the "episodes" table."""
    monkeypatch.setattr(ledger, "LEDGER_DIR", tmp_path / "ledger")
    shard_path = tmp_path / "elsewhere" / "shard.parquet"
    ledger.append_rows_to_path(shard_path, [{"start_id": "s0", "repeat_idx": 0}])

    assert shard_path.exists()
    assert not (tmp_path / "ledger" / "episodes.parquet").exists()


# =============================================================================
# Cross-process concurrency (task-ledgerlock-report.md): tonight's incident
# was two real writers racing append_rows's read-concat-write-tmp-rename
# against the SAME table at the SAME time.
# =============================================================================

def test_append_rows_concurrent_subprocesses_no_loss_no_dupes(tmp_path):
    """Two REAL OS subprocesses (not threads -- see _stress_worker_append.py:
    each is a genuine `python -m ...` child process, its own interpreter, its
    own GIL, the only way to actually exercise a cross-PROCESS fcntl.flock),
    each calling append_rows("episodes", ...) 50 times in a tight loop against
    the SAME ledger dir at the same time. Without the lock this is exactly
    tonight's incident shape (read-concat-write races: a row committed by one
    writer between the other's read and write is silently dropped, and/or a
    shared tmp filename collision surfaces as FileNotFoundError). With the
    lock: exactly 100 rows land, every (proc, i) pair exactly once -- no
    loss, no dupes."""
    n_per_proc = 50
    procs = [
        subprocess.Popen(
            [sys.executable, "-m", "bandit_v1.tests._stress_worker_append",
             str(tmp_path), str(proc_idx), str(n_per_proc)],
            cwd=str(config.REPO),
        )
        for proc_idx in range(2)
    ]
    for p in procs:
        rc = p.wait(timeout=120)
        assert rc == 0, f"stress worker subprocess exited {rc}"

    df = pd.read_parquet(tmp_path / "episodes.parquet")
    assert df.shape[0] == 2 * n_per_proc

    got = set(zip(df["proc"].astype(str), df["i"].astype(int)))
    expected = {(str(proc_idx), i) for proc_idx in range(2) for i in range(n_per_proc)}
    assert got == expected, (
        f"missing: {expected - got}, unexpected/duplicated beyond expected: {got - expected}, "
        f"row count {df.shape[0]} vs expected {len(expected)}"
    )
    assert not df.duplicated(subset=["proc", "i"]).any()


def test_append_rows_lock_timeout_raises_loudly_when_held_by_another_process(tmp_path, monkeypatch):
    """A lock held by ANOTHER process (not released, e.g. a stuck/dead
    writer) must not hang this process's append_rows forever -- it must
    raise a loud, named TimeoutError instead. LOCK_TIMEOUT_S is monkeypatched
    down so the test doesn't take 120s; the holder subprocess signals a
    `ready` marker only after it actually holds the lock, so this test never
    races the acquire (a flaky pass/fail depending on scheduling)."""
    monkeypatch.setattr(ledger, "LEDGER_DIR", tmp_path)
    monkeypatch.setattr(ledger, "LOCK_TIMEOUT_S", 0.3)

    ready_marker = tmp_path / "holder_ready"
    holder = subprocess.Popen(
        [sys.executable, "-m", "bandit_v1.tests._stress_lock_holder",
         str(tmp_path), "5", str(ready_marker)],
        cwd=str(config.REPO),
    )
    try:
        deadline = time.monotonic() + 10
        while not ready_marker.exists():
            assert time.monotonic() < deadline, "holder subprocess never signaled ready"
            time.sleep(0.02)

        with pytest.raises(TimeoutError, match="episodes"):
            ledger.append_rows("episodes", [{"episode_id": "should_not_land"}])
    finally:
        holder.wait(timeout=10)

    # Lock released once the holder exits -- a normal append now succeeds.
    ledger.append_rows("episodes", [{"episode_id": "e1"}])
    df = ledger.read("episodes")
    assert list(df["episode_id"]) == ["e1"]
