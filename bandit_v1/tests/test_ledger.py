import pandas as pd
from bandit_v1 import ledger

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
