"""Tests for bandit_v1/rollout.py's new `skip_pairs` resume parameter (Task 7).

Only `run()`'s loop/bookkeeping is under test here -- `_rollout_one` (the real
env/policy interaction) is monkeypatched to a cheap stub, and
`states.make_env_for_rollout`/`_wcp.WebsocketClientPolicy` are monkeypatched so
no real mujoco env or websocket connection is ever constructed. `states.
start_features` runs for real against synthetic start dirs (same pattern as
test_states.py), so the row-shape/feature-merge part of `run()` is still
exercised genuinely.
"""
import json

import pandas as pd

from bandit_v1 import config, ledger, rollout


class _DummyEnv:
    def reset(self):
        pass


def _write_synthetic_start(start_dir, cat="jar"):
    start_dir.mkdir(parents=True, exist_ok=True)
    ep_meta = {
        "layout_id": 1, "style_id": 1,
        "object_cfgs": [{"name": "obj", "info": {"cat": cat, "mjcf_path": "/x/model.xml"}}],
        "init_robot_base_pos": [0.0, 0.0, 0.0],
        "init_robot_base_ori": [0.0, 0.0, 0.0],
    }
    fp = {
        "category": cat, "instance": "/x/model.xml", "layout_id": 1, "style_id": 1,
        "obj_xyz": [0.1, 0.1, 0.5], "base_xy": [0.0, 0.0],
    }
    (start_dir / "ep_meta.json").write_text(json.dumps(ep_meta))
    (start_dir / "fingerprint.json").write_text(json.dumps(fp))


def _patch_common(monkeypatch, tmp_path, calls):
    monkeypatch.setattr(ledger, "LEDGER_DIR", tmp_path / "ledger")
    monkeypatch.setattr(rollout, "get_task_horizon", lambda task: 5)
    monkeypatch.setattr(rollout.states, "make_env_for_rollout", lambda: _DummyEnv())
    monkeypatch.setattr(rollout.states, "close_env", lambda env: None)
    monkeypatch.setattr(rollout._wcp, "WebsocketClientPolicy", lambda host, port: object())

    def fake_rollout_one(env, client, start_dir, horizon):
        calls.append(start_dir.name)
        return dict(success=True, failure_stage="success", ee_min_dist=0.01,
                    max_lift=0.1, min_sink_dist=0.05, steps=10)
    monkeypatch.setattr(rollout, "_rollout_one", fake_rollout_one)


def test_run_default_behavior_unchanged_when_skip_pairs_omitted(tmp_path, monkeypatch):
    calls = []
    _patch_common(monkeypatch, tmp_path, calls)
    d0 = tmp_path / "start_000"
    _write_synthetic_start(d0)

    rows = rollout.run("host", 1, [d0], repeats=3, phase="diag", policy_id="pi0")

    assert len(rows) == 3
    assert [r["repeat_idx"] for r in rows] == [0, 1, 2]
    assert calls == ["start_000", "start_000", "start_000"]
    assert len(ledger.read("episodes")) == 3


def test_run_skips_pairs_already_present_without_calling_rollout_one(tmp_path, monkeypatch):
    calls = []
    _patch_common(monkeypatch, tmp_path, calls)
    d0 = tmp_path / "start_000"
    d1 = tmp_path / "start_001"
    _write_synthetic_start(d0)
    _write_synthetic_start(d1, cat="mug")

    # start_000/repeat0 already "done"; everything else still to run.
    skip_pairs = {("start_000", 0)}
    rows = rollout.run("host", 1, [d0, d1], repeats=2, phase="diag", policy_id="pi0",
                        skip_pairs=skip_pairs)

    ran = sorted((r["start_id"], r["repeat_idx"]) for r in rows)
    assert ran == [("start_000", 1), ("start_001", 0), ("start_001", 1)]
    assert len(rows) == 3
    assert calls.count("start_000") == 1  # only repeat 1 actually ran
    assert calls.count("start_001") == 2
    assert len(ledger.read("episodes")) == 3


def test_run_episodes_sink_default_still_appends_to_ledger_episodes_table(tmp_path, monkeypatch):
    """episodes_sink omitted (the default, None) must be byte-identical to
    behavior before this parameter existed: every row still lands in ledger
    table "episodes" via ledger.append_rows, one call per row."""
    calls = []
    _patch_common(monkeypatch, tmp_path, calls)
    d0 = tmp_path / "start_000"
    _write_synthetic_start(d0)

    append_calls = []
    orig_append = ledger.append_rows

    def counting_append(table, rows):
        append_calls.append((table, len(rows)))
        orig_append(table, rows)
    monkeypatch.setattr(ledger, "append_rows", counting_append)

    rows = rollout.run("host", 1, [d0], repeats=2, phase="diag", policy_id="pi0")

    assert len(rows) == 2
    assert append_calls == [("episodes", 1), ("episodes", 1)]  # per-episode, unchanged
    assert len(ledger.read("episodes")) == 2


def test_run_episodes_sink_override_bypasses_ledger_append_entirely(tmp_path, monkeypatch):
    """This is the seam parallel_eval.py's workers use: when episodes_sink is
    supplied, rollout.run must route every completed row through it INSTEAD
    of ledger.append_rows("episodes", ...) -- the shared table must be
    untouched."""
    calls = []
    _patch_common(monkeypatch, tmp_path, calls)
    d0 = tmp_path / "start_000"
    d1 = tmp_path / "start_001"
    _write_synthetic_start(d0)
    _write_synthetic_start(d1, cat="mug")

    sunk = []
    rows = rollout.run("host", 1, [d0, d1], repeats=2, phase="diag", policy_id="pi0",
                        episodes_sink=lambda row: sunk.append(row))

    assert len(rows) == 4
    assert len(sunk) == 4
    assert sunk == rows  # same row dicts, in completion order
    assert not (tmp_path / "ledger" / "episodes.parquet").exists()  # shared table never written


def test_run_skips_entire_start_when_all_its_repeats_are_done(tmp_path, monkeypatch):
    calls = []
    _patch_common(monkeypatch, tmp_path, calls)
    d0 = tmp_path / "start_000"
    _write_synthetic_start(d0)

    skip_pairs = {("start_000", 0), ("start_000", 1)}
    rows = rollout.run("host", 1, [d0], repeats=2, phase="diag", policy_id="pi0",
                        skip_pairs=skip_pairs)

    assert rows == []
    assert calls == []
    assert not (tmp_path / "ledger" / "episodes.parquet").exists()
