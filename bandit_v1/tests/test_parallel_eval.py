"""Tests for bandit_v1/parallel_eval.py (rollout-speedup #1: N parallel rollout
workers against one policy server, with concurrency-safe ledger writes).

No GPU/env/server anywhere here: `rollout.run` is always monkeypatched to a
cheap synthetic stand-in (same pattern test_rollout.py/test_run_diagnosis.py/
test_eval_set.py already use), and `run_parallel`'s `spawn_fn` seam replaces
real subprocess creation with an in-process fake that either (a) calls
`run_worker_inline` directly -- exercising the exact same code a real worker
subprocess runs, synchronously, in this test process -- or (b) simulates a
crashed worker by writing a partial shard file and returning a nonzero
"process".
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from bandit_v1 import config, ledger, parallel_eval


class _FakeProc:
    """Stand-in for subprocess.Popen: only `.wait()` is ever used by
    run_parallel (accepts and ignores `timeout=`, matching the real
    Popen.wait signature run_parallel now always calls with)."""
    def __init__(self, returncode=0):
        self.returncode = returncode

    def wait(self, timeout=None):
        return self.returncode


def _fake_rollout_run(host, port, start_dirs, repeats, phase, policy_id, arm=None,
                       pull_id=None, skip_pairs=None, episodes_sink=None):
    """Synthetic rollout.run stand-in: builds one row per (start_id,
    repeat_idx) pair not in skip_pairs, sinking each through episodes_sink --
    exactly the shape run_worker_inline needs to exercise its own
    skip_pairs-reconstruction + sink-wiring for real."""
    rows = []
    sink = episodes_sink if episodes_sink is not None else (lambda row: None)
    for sd in start_dirs:
        sid = Path(sd).name
        for r in range(repeats):
            if skip_pairs and (sid, r) in skip_pairs:
                continue
            row = {"start_id": sid, "repeat_idx": r, "success": True,
                   "phase": phase, "policy_id": policy_id, "arm": arm, "pull_id": pull_id}
            rows.append(row)
            sink(row)
    return rows


# =============================================================================
# shard_round_robin: pure sharding logic
# =============================================================================

def test_shard_round_robin_covers_all_exactly_once_preserves_order_and_balances():
    starts = [f"s{i}" for i in range(10)]
    shards = parallel_eval.shard_round_robin(starts, 3)

    assert len(shards) == 3
    flat = [s for shard in shards for s in shard]
    assert sorted(flat) == sorted(starts)
    assert len(flat) == len(starts)  # exactly once each, nothing dropped/duplicated
    sizes = [len(s) for s in shards]
    assert max(sizes) - min(sizes) <= 1  # round-robin balance
    for i, shard in enumerate(shards):
        assert shard == starts[i::3]  # each shard is a strict round-robin slice


def test_shard_round_robin_more_workers_than_starts_leaves_some_shards_empty():
    shards = parallel_eval.shard_round_robin(["a", "b"], 5)
    assert len(shards) == 5
    assert [s for s in shards if s] == [["a"], ["b"]]


def test_shard_round_robin_rejects_workers_lt_1():
    with pytest.raises(ValueError):
        parallel_eval.shard_round_robin(["a"], 0)


# =============================================================================
# run_worker_inline: spec -> rollout.run call -> shard file
# =============================================================================

def test_run_worker_inline_forwards_args_reconstructs_skip_pairs_and_sinks_to_shard(
        tmp_path, monkeypatch):
    calls = []

    def fake_run(host, port, start_dirs, repeats, phase, policy_id, arm=None,
                 pull_id=None, skip_pairs=None, episodes_sink=None):
        calls.append(dict(host=host, port=port, start_dirs=[Path(p).name for p in start_dirs],
                           repeats=repeats, phase=phase, policy_id=policy_id, arm=arm,
                           pull_id=pull_id, skip_pairs=skip_pairs))
        return _fake_rollout_run(host, port, start_dirs, repeats, phase, policy_id,
                                  arm=arm, pull_id=pull_id, skip_pairs=skip_pairs,
                                  episodes_sink=episodes_sink)

    monkeypatch.setattr(parallel_eval.rollout, "run", fake_run)

    shard_path = tmp_path / "shard0.parquet"
    spec = {
        "policy_host": "h", "policy_port": 123,
        "start_dirs": [str(tmp_path / "start_000"), str(tmp_path / "start_001")],
        "repeats": 2, "phase": "eval", "policy_id": "pi0_baseline",
        "arm": "targeted", "pull_id": "pull_x_j1",
        "skip_pairs": [["start_000", 0]],
        "shard_path": str(shard_path),
    }

    parallel_eval.run_worker_inline(spec)

    assert len(calls) == 1
    c = calls[0]
    assert c["host"] == "h" and c["port"] == 123
    assert c["start_dirs"] == ["start_000", "start_001"]
    assert c["arm"] == "targeted" and c["pull_id"] == "pull_x_j1"
    assert c["skip_pairs"] == {("start_000", 0)}

    assert shard_path.exists()
    df = ledger.read_path(shard_path)
    assert set(zip(df["start_id"], df["repeat_idx"])) == {("start_000", 1), ("start_001", 0), ("start_001", 1)}


def test_run_worker_inline_empty_skip_pairs_becomes_none():
    """An empty skip_pairs list (every pair in this shard still to run) must
    reconstruct to None, not an empty-but-truthy set -- matches rollout.run's
    own `is not None` check semantics, but verify the seam does not
    accidentally pass a container that changes behavior."""
    seen = {}

    def fake_run(host, port, start_dirs, repeats, phase, policy_id, arm=None,
                 pull_id=None, skip_pairs=None, episodes_sink=None):
        seen["skip_pairs"] = skip_pairs

    import bandit_v1.parallel_eval as pe
    orig = pe.rollout.run
    pe.rollout.run = fake_run
    try:
        pe.run_worker_inline({
            "policy_host": "h", "policy_port": 1, "start_dirs": [], "repeats": 1,
            "phase": "eval", "policy_id": "pi0", "skip_pairs": [], "shard_path": "/tmp/x.parquet",
        })
    finally:
        pe.rollout.run = orig

    assert seen["skip_pairs"] is None


# =============================================================================
# merge_shards
# =============================================================================

def test_merge_shards_unions_rows_appends_once_and_cleans_up(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER_DIR", tmp_path / "ledger")
    shard_dir = tmp_path / "ledger" / "shards"
    p0 = shard_dir / "tag_0.parquet"
    p1 = shard_dir / "tag_1.parquet"
    ledger.append_rows_to_path(p0, [{"start_id": "s0", "repeat_idx": 0, "success": True}])
    ledger.append_rows_to_path(p1, [
        {"start_id": "s1", "repeat_idx": 0, "success": False},
        {"start_id": "s1", "repeat_idx": 1, "success": True},
    ])

    append_calls = []
    orig_append = ledger.append_rows

    def counting_append(table, rows):
        append_calls.append((table, len(rows)))
        orig_append(table, rows)

    monkeypatch.setattr(ledger, "append_rows", counting_append)

    rows = parallel_eval.merge_shards([p0, p1])

    assert len(rows) == 3
    assert append_calls == [("episodes", 3)]  # exactly ONE append_rows call for the whole merge
    assert not p0.exists() and not p1.exists()  # shard files cleaned up
    d = ledger.read("episodes")
    assert len(d) == 3
    assert set(zip(d["start_id"], d["repeat_idx"])) == {("s0", 0), ("s1", 0), ("s1", 1)}


def test_merge_shards_skips_missing_files(tmp_path, monkeypatch):
    """A worker that crashed before completing even one episode never created
    a shard file at all -- merge must simply skip it, not error."""
    monkeypatch.setattr(ledger, "LEDGER_DIR", tmp_path / "ledger")
    shard_dir = tmp_path / "ledger" / "shards"
    p0 = shard_dir / "tag_0.parquet"
    p_missing = shard_dir / "tag_1.parquet"
    ledger.append_rows_to_path(p0, [{"start_id": "s0", "repeat_idx": 0, "success": True}])

    rows = parallel_eval.merge_shards([p0, p_missing])

    assert len(rows) == 1
    assert not p0.exists()
    assert len(ledger.read("episodes")) == 1


def test_merge_shards_all_missing_returns_empty_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER_DIR", tmp_path / "ledger")
    rows = parallel_eval.merge_shards([tmp_path / "ledger" / "shards" / "nope.parquet"])
    assert rows == []
    assert not (tmp_path / "ledger" / "episodes.parquet").exists()


# =============================================================================
# run_parallel: end-to-end sharding + merge (subprocess spawn faked)
# =============================================================================

def test_run_parallel_covers_all_pairs_incl_skip_pairs_and_keeps_repeats_together(
        tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LEDGER_DIR", tmp_path / "ledger")
    monkeypatch.setattr(ledger, "LEDGER_DIR", tmp_path / "ledger")
    monkeypatch.setattr(parallel_eval.rollout, "run", _fake_rollout_run)

    start_dirs = [tmp_path / f"start_{i:03d}" for i in range(5)]
    skip_pairs = {("start_000", 0), ("start_002", 1), ("start_004", 2)}

    assignments = []  # [(spec_path, [start_id, ...]), ...] -- one entry per worker actually spawned

    def fake_spawn(spec_path, log_path):
        spec = json.loads(Path(spec_path).read_text())
        assignments.append((str(spec_path), [Path(sd).name for sd in spec["start_dirs"]]))
        parallel_eval.run_worker_inline(spec)
        return _FakeProc(0)

    rows = parallel_eval.run_parallel(
        "host", 1, start_dirs, repeats=3, phase="diag", policy_id="pi0",
        workers=2, skip_pairs=skip_pairs, spawn_fn=fake_spawn,
        log_dir=tmp_path / "logs", spec_dir=tmp_path / "specs")

    expected = {(f"start_{i:03d}", r) for i in range(5) for r in range(3)} - skip_pairs
    assert {(row["start_id"], row["repeat_idx"]) for row in rows} == expected
    assert len(rows) == len(expected)

    d = ledger.read("episodes")
    assert set(zip(d["start_id"], d["repeat_idx"])) == expected
    assert len(d) == len(expected)  # exactly once each -- no duplicate merge

    # every start's repeats all landed in exactly ONE worker's spec (never
    # split across two workers, which would defeat the warm-restore speedup).
    owner = {}
    for spec_path, sids in assignments:
        for sid in sids:
            owner.setdefault(sid, set()).add(spec_path)
    assert set(owner) == {f"start_{i:03d}" for i in range(5)}
    assert all(len(v) == 1 for v in owner.values())

    # shard files are cleaned up after a fully-successful merge
    assert list((tmp_path / "ledger" / "shards").glob("*.parquet")) == []


def test_run_parallel_worker_crash_keeps_partial_shard_and_raises_with_log_tail(
        tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LEDGER_DIR", tmp_path / "ledger")
    monkeypatch.setattr(ledger, "LEDGER_DIR", tmp_path / "ledger")
    monkeypatch.setattr(parallel_eval.rollout, "run", _fake_rollout_run)

    start_dirs = [tmp_path / f"start_{i:03d}" for i in range(4)]  # workers=2 -> shard0=[0,2], shard1=[1,3]

    call_idx = {"n": 0}

    def fake_spawn(spec_path, log_path):
        idx = call_idx["n"]
        call_idx["n"] += 1
        spec = json.loads(Path(spec_path).read_text())
        if idx == 0:
            # Simulate: this worker completed ONE episode, then crashed before
            # the rest of its shard -- the partial row is already durably on
            # disk (its own shard file), exactly like a real worker's
            # per-episode episodes_sink call would have left it.
            partial_sid = Path(spec["start_dirs"][0]).name
            ledger.append_rows_to_path(
                Path(spec["shard_path"]),
                [{"start_id": partial_sid, "repeat_idx": 0, "success": False}])
            Path(log_path).write_text("Traceback (most recent call last):\nRuntimeError: boom\n")
            return _FakeProc(1)
        parallel_eval.run_worker_inline(spec)
        return _FakeProc(0)

    with pytest.raises(RuntimeError, match=r"1/2 worker\(s\) failed") as excinfo:
        parallel_eval.run_parallel(
            "host", 1, start_dirs, repeats=2, phase="eval", policy_id="pi0",
            workers=2, spawn_fn=fake_spawn, log_dir=tmp_path / "logs", spec_dir=tmp_path / "specs")

    assert "boom" in str(excinfo.value)  # the dead worker's log tail is surfaced in the error

    # surviving + partial shards were still merged: worker0's 1 partial row +
    # worker1's full 2 starts x 2 repeats = 4 rows -> 5 total, nothing lost.
    d = ledger.read("episodes")
    assert len(d) == 5
    assert list((tmp_path / "ledger" / "shards").glob("*.parquet")) == []  # both read shards cleaned up


class _HangingProc:
    """Stand-in for a worker subprocess that never exits on its own: the
    FIRST `.wait(timeout=...)` call always raises TimeoutExpired (simulating
    a genuinely hung rollout); `.kill()` flips a flag so the SECOND wait
    (run_parallel's post-kill reap) succeeds, mirroring how a real killed
    process eventually becomes waitable."""
    def __init__(self):
        self.killed = False
        self.kill_called = False
        self.wait_calls = 0

    def wait(self, timeout=None):
        self.wait_calls += 1
        if not self.killed:
            raise subprocess.TimeoutExpired(cmd="worker", timeout=timeout)
        return -9

    def kill(self):
        self.kill_called = True
        self.killed = True


def test_run_parallel_hung_worker_times_out_is_killed_and_surfaces_in_error(
        tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LEDGER_DIR", tmp_path / "ledger")
    monkeypatch.setattr(ledger, "LEDGER_DIR", tmp_path / "ledger")
    monkeypatch.setattr(parallel_eval.rollout, "run", _fake_rollout_run)

    start_dirs = [tmp_path / f"start_{i:03d}" for i in range(4)]  # workers=2 -> shard0=[0,2], shard1=[1,3]
    hung = _HangingProc()

    def fake_spawn(spec_path, log_path):
        spec = json.loads(Path(spec_path).read_text())
        if spec["start_dirs"] == [str(Path(sd)) for sd in start_dirs[0::2]]:
            # this is worker 0 (the one we hang) -- it DID write one episode
            # to its shard before hanging, matching a real worker's
            # per-episode durability (see the crash test above).
            partial_sid = Path(spec["start_dirs"][0]).name
            ledger.append_rows_to_path(
                Path(spec["shard_path"]),
                [{"start_id": partial_sid, "repeat_idx": 0, "success": True}])
            Path(log_path).write_text("stuck waiting on policy server...\n")
            return hung
        parallel_eval.run_worker_inline(spec)
        return _FakeProc(0)

    with pytest.raises(RuntimeError, match=r"1/2 worker\(s\) failed") as excinfo:
        parallel_eval.run_parallel(
            "host", 1, start_dirs, repeats=2, phase="eval", policy_id="pi0",
            workers=2, spawn_fn=fake_spawn, log_dir=tmp_path / "logs", spec_dir=tmp_path / "specs",
            timeout=0.01)

    assert "TIMED OUT" in str(excinfo.value)
    assert "stuck waiting on policy server" in str(excinfo.value)
    assert hung.kill_called is True         # breached worker was killed
    assert hung.wait_calls == 2             # one timed-out wait, one post-kill reap

    # the hung worker's partial shard + the surviving worker's full shard
    # were both still merged (a timeout never silently discards completed
    # episodes, same guarantee as a nonzero-exit crash).
    d = ledger.read("episodes")
    assert len(d) == 5
    assert list((tmp_path / "ledger" / "shards").glob("*.parquet")) == []


def test_run_parallel_default_timeout_is_10800s():
    assert parallel_eval.DEFAULT_WORKER_TIMEOUT_S == 10800


def test_run_parallel_workers_none_or_one_falls_back_to_plain_rollout_run(monkeypatch):
    calls = []

    def fake_run(host, port, start_dirs, repeats, phase, policy_id, arm=None,
                 pull_id=None, skip_pairs=None):
        calls.append((host, port, list(start_dirs), repeats, phase, policy_id, skip_pairs))
        return [{"ok": True}]

    monkeypatch.setattr(parallel_eval.rollout, "run", fake_run)

    def fail_spawn(*a, **k):
        raise AssertionError("spawn_fn must not be called when workers<=1")

    rows1 = parallel_eval.run_parallel("h", 1, ["a", "b"], repeats=2, phase="eval",
                                        policy_id="pi0", workers=1, spawn_fn=fail_spawn)
    rows_none = parallel_eval.run_parallel("h", 1, ["a"], repeats=1, phase="eval",
                                            policy_id="pi0", workers=None, spawn_fn=fail_spawn)

    assert rows1 == [{"ok": True}] and rows_none == [{"ok": True}]
    assert len(calls) == 2
    assert calls[0] == ("h", 1, ["a", "b"], 2, "eval", "pi0", None)


def test_run_parallel_passes_skip_pairs_through_unfiltered_to_every_worker_spec(tmp_path):
    """Matches run_diagnosis.py's own convention (test_run_diagnosis.py's
    resume test): the full skip_pairs set is written into every worker's spec
    verbatim, not pre-filtered down to that worker's own start_ids -- harmless
    since rollout.run only ever matches entries against its own start_dirs."""
    specs_seen = []

    def fake_spawn(spec_path, log_path):
        specs_seen.append(json.loads(Path(spec_path).read_text()))
        return _FakeProc(0)

    skip_pairs = {("s0", 0), ("s1", 1)}
    parallel_eval.run_parallel(
        "h", 1, ["s0", "s1"], repeats=2, phase="diag", policy_id="pi0",
        workers=2, skip_pairs=skip_pairs, spawn_fn=fake_spawn,
        log_dir=tmp_path / "logs", spec_dir=tmp_path / "specs")

    assert len(specs_seen) == 2
    for spec in specs_seen:
        assert sorted(map(tuple, spec["skip_pairs"])) == sorted(skip_pairs)


# =============================================================================
# _spawn_worker: subprocess argv/env construction (Popen itself faked)
# =============================================================================

def test_spawn_worker_builds_expected_argv_and_creates_log_file(tmp_path, monkeypatch):
    captured = {}

    class FakePopen:
        def __init__(self, argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs

    monkeypatch.setattr(parallel_eval.subprocess, "Popen", FakePopen)

    spec_path = tmp_path / "spec.json"
    log_path = tmp_path / "sub" / "worker0.log"
    parallel_eval._spawn_worker(spec_path, log_path)

    assert captured["argv"][0] == sys.executable
    assert captured["argv"][1:] == ["-m", "bandit_v1.parallel_eval", "--spec", str(spec_path)]
    assert captured["kwargs"]["cwd"] == str(config.REPO)
    assert log_path.exists()  # parent dir created + log file opened for the child to write into


# =============================================================================
# _worker_env: per-worker thread-count caps + unbuffered output
# (task-importhang-report.md)
# =============================================================================

def test_worker_env_sets_thread_caps_and_unbuffered_by_default(monkeypatch):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "LP_NUM_THREADS", "PYTHONUNBUFFERED"):
        monkeypatch.delenv(k, raising=False)

    env = parallel_eval._worker_env()

    assert env["OMP_NUM_THREADS"] == parallel_eval._WORKER_THREAD_CAP
    assert env["OPENBLAS_NUM_THREADS"] == parallel_eval._WORKER_THREAD_CAP
    assert env["MKL_NUM_THREADS"] == parallel_eval._WORKER_THREAD_CAP
    assert env["NUMEXPR_NUM_THREADS"] == parallel_eval._WORKER_THREAD_CAP
    assert env["LP_NUM_THREADS"] == parallel_eval._WORKER_THREAD_CAP
    assert env["PYTHONUNBUFFERED"] == "1"


def test_worker_env_does_not_override_a_caller_supplied_value(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "17")
    monkeypatch.setenv("PYTHONUNBUFFERED", "0")

    env = parallel_eval._worker_env()

    assert env["OMP_NUM_THREADS"] == "17"      # caller's explicit value wins
    assert env["PYTHONUNBUFFERED"] == "0"


def test_worker_env_forwards_unrelated_vars_as_is(monkeypatch):
    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

    env = parallel_eval._worker_env()

    assert env["MUJOCO_GL"] == "osmesa"
    assert env["CUDA_VISIBLE_DEVICES"] == "1"


def test_spawn_worker_uses_worker_env(tmp_path, monkeypatch):
    captured = {}

    class FakePopen:
        def __init__(self, argv, **kwargs):
            captured["kwargs"] = kwargs

    monkeypatch.setattr(parallel_eval.subprocess, "Popen", FakePopen)
    sentinel_env = {"SENTINEL": "yes"}
    monkeypatch.setattr(parallel_eval, "_worker_env", lambda: sentinel_env)

    parallel_eval._spawn_worker(tmp_path / "spec.json", tmp_path / "worker0.log")

    assert captured["kwargs"]["env"] is sentinel_env


# =============================================================================
# run_worker_inline progress prints (task-importhang-report.md): a worker's
# log must show visible forward progress, not just the import-time warnings,
# so a genuinely stuck worker is distinguishable from a slow-but-alive one.
# =============================================================================

def test_run_worker_inline_prints_start_marker_and_per_episode_progress(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(parallel_eval.rollout, "run", _fake_rollout_run)

    shard_path = tmp_path / "shard0.parquet"
    spec = {
        "policy_host": "h", "policy_port": 123,
        "start_dirs": [str(tmp_path / "start_000"), str(tmp_path / "start_001")],
        "repeats": 2, "phase": "eval", "policy_id": "pi0",
        "skip_pairs": [], "shard_path": str(shard_path),
    }

    parallel_eval.run_worker_inline(spec)

    out = capsys.readouterr().out
    assert "starting rollout.run" in out
    # 2 starts x 2 repeats = 4 episodes -> 4 "written to shard" progress lines
    assert out.count("written to shard") == 4
    assert "done, 4 new episode(s)" in out


def test_run_worker_inline_prints_start_marker_even_when_everything_is_skipped(
        tmp_path, monkeypatch, capsys):
    """A resumed worker whose whole shard is already done (skip_pairs covers
    every pair) never calls `sink` at all -- the start-marker print must
    still fire so the log is never silent from t=0, even in this case."""
    monkeypatch.setattr(parallel_eval.rollout, "run", _fake_rollout_run)

    spec = {
        "policy_host": "h", "policy_port": 123,
        "start_dirs": [str(tmp_path / "start_000")],
        "repeats": 1, "phase": "eval", "policy_id": "pi0",
        "skip_pairs": [["start_000", 0]], "shard_path": str(tmp_path / "shard0.parquet"),
    }

    parallel_eval.run_worker_inline(spec)

    out = capsys.readouterr().out
    assert "starting rollout.run" in out
    assert "done, 0 new episode(s)" in out
