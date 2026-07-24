"""Tests for bandit_v1/run_diagnosis.py (Task 7's chunked/resumable driver).

Pure-logic + monkeypatched-integration tests only -- no real env, policy
server, or GPU. `rollout.run` is faked in the run_all tests (its own
skip_pairs behavior is covered by test_rollout.py) so these tests isolate
run_diagnosis.py's own chunking/resume bookkeeping.
"""
import pandas as pd
import pytest

from bandit_v1 import config, ledger, rollout, run_diagnosis


def test_chunked_splits_with_remainder():
    assert run_diagnosis.chunked([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]


def test_chunked_exact_multiple():
    assert run_diagnosis.chunked([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]


def test_chunked_empty():
    assert run_diagnosis.chunked([], 5) == []


def test_load_ordered_start_ids_preserves_parquet_row_order(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "N_DIAG", 3)
    p = tmp_path / "diag_conditions.parquet"
    pd.DataFrame({"start_id": ["start_00007", "start_00002", "start_00099"]}).to_parquet(p)

    ids = run_diagnosis.load_ordered_start_ids(parquet_path=p)

    assert ids == ["start_00007", "start_00002", "start_00099"]


def test_load_ordered_start_ids_rejects_wrong_row_count(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "N_DIAG", 3)
    p = tmp_path / "diag_conditions.parquet"
    pd.DataFrame({"start_id": ["start_00001", "start_00002"]}).to_parquet(p)

    with pytest.raises(AssertionError):
        run_diagnosis.load_ordered_start_ids(parquet_path=p)


def test_load_ordered_start_ids_rejects_duplicate_start_id(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "N_DIAG", 3)
    p = tmp_path / "diag_conditions.parquet"
    # 3 rows but only 2 distinct start_ids -- must fail the uniqueness half
    # of the sanity guard even though the row COUNT matches N_DIAG.
    pd.DataFrame({"start_id": ["start_00001", "start_00001", "start_00002"]}).to_parquet(p)

    with pytest.raises(AssertionError):
        run_diagnosis.load_ordered_start_ids(parquet_path=p)


def test_done_pairs_empty_when_no_ledger_table(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER_DIR", tmp_path)
    assert run_diagnosis.done_pairs() == set()


def test_done_pairs_filters_by_phase_and_policy_id(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER_DIR", tmp_path)
    ledger.append_rows("episodes", [
        {"phase": "diag", "policy_id": "pi0", "start_id": "s0", "repeat_idx": 0},
        {"phase": "diag", "policy_id": "pi0", "start_id": "s0", "repeat_idx": 1},
        {"phase": "diag", "policy_id": "other", "start_id": "s1", "repeat_idx": 0},
        {"phase": "smoke", "policy_id": "pi0", "start_id": "s2", "repeat_idx": 0},
    ])

    assert run_diagnosis.done_pairs() == {("s0", 0), ("s0", 1)}


def test_run_all_skips_fully_done_chunk_and_resumes_partial_chunk(tmp_path, monkeypatch):
    """4 start_ids, chunk_size=2 -> 2 chunks. Chunk 0 (s0, s1) has s0 fully
    done already but s1 untouched (partial -> must still call rollout.run,
    with skip_pairs covering exactly s0's 2 rows). Chunk 1 (s2, s3) is fully
    done already (both starts, both repeats) -> must NOT call rollout.run at
    all."""
    monkeypatch.setattr(ledger, "LEDGER_DIR", tmp_path)
    monkeypatch.setattr(run_diagnosis, "load_ordered_start_ids",
                         lambda: ["s0", "s1", "s2", "s3"])

    # Pre-seed: s0 fully done (chunk 0, partial); s2+s3 fully done (chunk 1, complete).
    ledger.append_rows("episodes", [
        {"phase": "diag", "policy_id": "pi0", "start_id": "s0", "repeat_idx": 0, "success": True},
        {"phase": "diag", "policy_id": "pi0", "start_id": "s0", "repeat_idx": 1, "success": True},
        {"phase": "diag", "policy_id": "pi0", "start_id": "s2", "repeat_idx": 0, "success": True},
        {"phase": "diag", "policy_id": "pi0", "start_id": "s2", "repeat_idx": 1, "success": True},
        {"phase": "diag", "policy_id": "pi0", "start_id": "s3", "repeat_idx": 0, "success": True},
        {"phase": "diag", "policy_id": "pi0", "start_id": "s3", "repeat_idx": 1, "success": True},
    ])

    calls = []

    def fake_run(host, port, start_dirs, repeats, phase, policy_id, arm=None,
                 pull_id=None, skip_pairs=None):
        calls.append({
            "start_ids": [sd.name for sd in start_dirs],
            "skip_pairs": set(skip_pairs) if skip_pairs is not None else None,
        })
        new_rows = []
        for sd in start_dirs:
            for r in range(repeats):
                if skip_pairs and (sd.name, r) in skip_pairs:
                    continue
                row = {"phase": phase, "policy_id": policy_id, "start_id": sd.name,
                       "repeat_idx": r, "success": True}
                new_rows.append(row)
                ledger.append_rows("episodes", [row])
        return new_rows

    monkeypatch.setattr(rollout, "run", fake_run)

    run_diagnosis.run_all("127.0.0.1", 1, repeats=2, chunk_size=2, log=lambda *a: None)

    # Only chunk 0 triggers a rollout.run call; chunk 1 (s2,s3) was fully done.
    # skip_pairs passed through is done_pairs()'s FULL (unfiltered-by-chunk)
    # result -- harmless since rollout.run only ever matches entries against
    # its own start_dirs' names -- so it also includes the pre-seeded s2/s3
    # pairs, not just s0's.
    assert len(calls) == 1
    assert calls[0]["start_ids"] == ["s0", "s1"]
    assert calls[0]["skip_pairs"] == {
        ("s0", 0), ("s0", 1), ("s2", 0), ("s2", 1), ("s3", 0), ("s3", 1),
    }

    d = ledger.read("episodes")
    d = d[(d["phase"] == "diag") & (d["policy_id"] == "pi0")]
    assert len(d) == 8  # 4 pre-seeded (s0 x2, s2+s3 x4) + 2 newly run (s1 x2)
    assert set(zip(d["start_id"], d["repeat_idx"])) == {
        ("s0", 0), ("s0", 1), ("s1", 0), ("s1", 1),
        ("s2", 0), ("s2", 1), ("s3", 0), ("s3", 1),
    }
