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
