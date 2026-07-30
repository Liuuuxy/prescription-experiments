"""Tests for bandit_v1/run_race.py (Task 14 step 3: nulls -> successive-
elimination race, resumable).

Pure/monkeypatched only -- no real GPU, policy server, training subprocess,
or live env anywhere here. `run_one`/`read_pulls_fn`/`claimable_fn`/
`sleep_fn` are exactly the seams run_race.py exposes for this; a small
in-memory `FakeLedger` (below) stands in for the real pulls.parquet so the
round-sequencing logic (resume-mid-round, elimination honored, T-cap stop,
null-phase skip) can be driven by synthetic data exactly like
test_scheduler.py's own synthetic pulls_df fixtures.
"""
import threading
import time

import numpy as np
import pandas as pd
import pytest
import yaml

from bandit_v1 import config, pull, run_race as rr


PULLS_COLS = ["pull_id", "arm", "round_j", "delta", "status"]


def _empty_pulls_df():
    return pd.DataFrame(columns=PULLS_COLS)


def _pulls(rows):
    """Same shape convention as test_scheduler.py's `_pulls`: rows are
    (arm, round_j, delta[, status]) tuples, status defaults to "ok"."""
    out = []
    for i, r in enumerate(rows):
        arm, round_j, delta = r[0], r[1], r[2]
        status = r[3] if len(r) > 3 else "ok"
        out.append({"pull_id": f"{arm}_j{round_j}", "arm": arm, "round_j": round_j,
                     "delta": delta, "status": status})
    return pd.DataFrame(out, columns=PULLS_COLS)


class FakeLedger:
    """In-memory stand-in for the real pulls ledger + pull.run_pull: every
    `run_one(spec)` call appends exactly one row (status "ok" unless
    `fail_set` says otherwise), mirroring run_pull's real contract, without
    touching disk. `delta_fn(arm, round_j) -> float` supplies the row's
    delta; `preseed` rows (already-existing ledger state, e.g. simulating a
    crash mid-round) are NEVER touched by `run_one` -- only appended-to."""

    def __init__(self, delta_fn=None, fail_set=None, preseed=None):
        self.rows = list(preseed) if preseed else []
        self.delta_fn = delta_fn if delta_fn is not None else (lambda arm, j: 0.0)
        self.fail_set = fail_set or set()
        self.calls = []

    def read(self):
        return pd.DataFrame(self.rows, columns=PULLS_COLS) if self.rows else _empty_pulls_df()

    def run_one(self, spec):
        arm, j = spec["arm"], spec["j"]
        self.calls.append((arm, j))
        if (arm, j) in self.fail_set:
            row = {"pull_id": f"{arm}_j{j}", "arm": arm, "round_j": j, "delta": None, "status": "failed"}
        else:
            row = {"pull_id": f"{arm}_j{j}", "arm": arm, "round_j": j,
                   "delta": float(self.delta_fn(arm, j)), "status": "ok"}
        self.rows.append(row)
        return row


def _quiet_log(*a):
    pass


# =============================================================================
# preconditions
# =============================================================================

def _write_cfg(path, baseline=None):
    doc = {}
    if baseline is not None:
        doc["baseline"] = baseline
    path.write_text(yaml.safe_dump(doc, sort_keys=False))


def test_preconditions_status_all_missing(tmp_path):
    status = rr.preconditions_status(cfg_path=tmp_path / "no_such.yaml",
                                       e_manifest_path=tmp_path / "no_such.parquet")
    assert status == {"baseline_ready": False, "e_manifest_ready": False, "ready": False}


def test_preconditions_status_partial_baseline_block_not_ready(tmp_path):
    cfg = tmp_path / "config.yaml"
    _write_cfg(cfg, baseline={"b": 0.5, "per_stratum_b": None, "sigma_e_eval": 0.02})
    status = rr.preconditions_status(cfg_path=cfg, e_manifest_path=tmp_path / "missing.parquet")
    assert status["baseline_ready"] is False


def test_preconditions_status_ready_when_both_present(tmp_path):
    cfg = tmp_path / "config.yaml"
    _write_cfg(cfg, baseline={"b": 0.5, "per_stratum_b": {"easy": 0.6}, "sigma_e_eval": 0.02})
    manifest = tmp_path / "E_manifest.parquet"
    manifest.write_text("x")
    status = rr.preconditions_status(cfg_path=cfg, e_manifest_path=manifest)
    assert status == {"baseline_ready": True, "e_manifest_ready": True, "ready": True}


def test_wait_for_preconditions_polls_then_succeeds(tmp_path):
    cfg = tmp_path / "config.yaml"
    manifest = tmp_path / "E_manifest.parquet"
    sleeps = {"n": 0}

    def fake_sleep(secs):
        sleeps["n"] += 1
        if sleeps["n"] == 2:
            _write_cfg(cfg, baseline={"b": 0.1, "per_stratum_b": {}, "sigma_e_eval": 0.0})
            manifest.write_text("x")

    status = rr.wait_for_preconditions(cfg_path=cfg, e_manifest_path=manifest,
                                        sleep_fn=fake_sleep, log=_quiet_log)
    assert status["ready"] is True
    assert sleeps["n"] == 2


def test_wait_for_preconditions_raises_timeout_after_max_polls(tmp_path):
    with pytest.raises(TimeoutError):
        rr.wait_for_preconditions(
            cfg_path=tmp_path / "missing.yaml", e_manifest_path=tmp_path / "missing.parquet",
            sleep_fn=lambda secs: None, log=_quiet_log, max_polls=3)


def test_load_baseline_and_load_frozen_B(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({
        "baseline": {"b": 0.42, "per_stratum_b": {"easy": 0.7, "hard": 0.2}, "sigma_e_eval": 0.03},
        "arms_freeze": {"B": 200},
    }))
    b, per_stratum, sigma_e_eval = rr.load_baseline(cfg_path=cfg)
    assert b == pytest.approx(0.42)
    assert per_stratum == {"easy": 0.7, "hard": 0.2}
    assert sigma_e_eval == pytest.approx(0.03)
    assert rr.load_frozen_B(cfg_path=cfg) == 200


def test_frozen_arm_names_index_order_plus_random():
    arms_spec = {
        "random_arm": True,
        "arms": [
            {"index": 1, "name": "easy_band"},
            {"index": 0, "name": "tall_vessel_grasp_fail"},
            {"index": 2, "name": "mid_band"},
        ],
    }
    assert rr.frozen_arm_names(arms_spec) == [
        "tall_vessel_grasp_fail", "easy_band", "mid_band", "random"]


# =============================================================================
# noise_floor note (append-once guard)
# =============================================================================

def test_append_noise_floor_to_config_yaml_writes_block_and_guards_against_double_append(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("existing: true\n")

    rr.append_noise_floor_to_config_yaml([0.01, -0.02], 0.05, path=cfg, log=_quiet_log)
    doc = yaml.safe_load(cfg.read_text())
    assert doc["existing"] is True
    assert doc["noise_floor"]["null_deltas"] == [0.01, -0.02]
    assert doc["noise_floor"]["sigma_e"] == pytest.approx(0.05)

    before = cfg.read_text()
    rr.append_noise_floor_to_config_yaml([0.01, -0.02], 0.05, path=cfg, log=_quiet_log)  # same values: no-op
    after = cfg.read_text()
    assert after == before  # identical values: pure no-op
    # mismatched values must be REPLACED loudly, not silently kept
    rr.append_noise_floor_to_config_yaml([0.99, 0.99], 999.0, path=cfg, log=_quiet_log)
    doc = yaml.safe_load(cfg.read_text())
    assert doc["noise_floor"]["sigma_e"] == pytest.approx(999.0)
    assert doc["noise_floor"]["null_deltas"] == [0.99, 0.99]


# =============================================================================
# GPU claimability
# =============================================================================

def test_gpu_free_mib_or_none_returns_value_from_query():
    assert rr.gpu_free_mib_or_none(0, query=lambda g: 12345) == 12345


def test_gpu_free_mib_or_none_returns_none_when_query_raises():
    def broken(g):
        raise RuntimeError("nvidia-smi broken")
    assert rr.gpu_free_mib_or_none(0, query=broken) is None


def test_gpu_claimable_none_is_treated_as_claimable():
    assert rr.gpu_claimable(0, query=lambda g: (_ for _ in ()).throw(RuntimeError("x"))) is True


def test_gpu_claimable_insufficient_memory_is_not_claimable():
    assert rr.gpu_claimable(0, need_mib=70000, query=lambda g: 100) is False


def test_both_slots_claimable_requires_both():
    assert rr.both_slots_claimable(query=lambda g: 999999) is True
    assert rr.both_slots_claimable(
        query=lambda g: 999999 if g == 0 else 100) is False


# =============================================================================
# run_batch_two_wide
# =============================================================================

def test_run_batch_two_wide_sequential_when_not_claimable():
    order = []

    def run_one(spec):
        order.append(spec["slot"])
        return spec["slot"]

    out = rr.run_batch_two_wide([{"slot": "a"}, {"slot": "b"}], run_one,
                                 claimable_fn=lambda: False, log=_quiet_log, sleep_fn=lambda s: None)
    assert order == ["a", "b"]
    assert out == ["a", "b"]


def test_run_batch_two_wide_runs_concurrently_when_both_claimable():
    # A 2-party barrier only releases if BOTH run_one calls reach it at
    # roughly the same time -- proves genuine concurrency (a sequential
    # execution would have the first call block forever waiting for a
    # second that hasn't started yet).
    barrier = threading.Barrier(2, timeout=2)
    seen = []

    def run_one(spec):
        barrier.wait()
        seen.append(spec["slot"])
        return spec["slot"]

    out = rr.run_batch_two_wide([{"slot": "a"}, {"slot": "b"}], run_one,
                                 claimable_fn=lambda: True, log=_quiet_log, sleep_fn=lambda s: None)
    assert set(seen) == {"a", "b"}
    assert set(out) == {"a", "b"}


def test_run_batch_two_wide_sequential_mode_would_break_a_barrier():
    """Sanity-checks the barrier technique itself: with claimable_fn=False
    (forced sequential), the first call's barrier.wait() must time out
    (BrokenBarrierError) since the second call never starts until the first
    returns -- confirming the concurrent test above is actually exercising
    concurrency, not something the barrier can't distinguish."""
    barrier = threading.Barrier(2, timeout=0.3)

    def run_one(spec):
        barrier.wait()
        return spec["slot"]

    with pytest.raises(threading.BrokenBarrierError):
        rr.run_batch_two_wide([{"slot": "a"}, {"slot": "b"}], run_one,
                               claimable_fn=lambda: False, log=_quiet_log, sleep_fn=lambda s: None)


def test_run_batch_two_wide_all_specs_attempted_even_if_one_raises():
    completed = []

    def run_one(spec):
        if spec["slot"] == "a":
            raise RuntimeError("boom")
        completed.append(spec["slot"])
        return spec["slot"]

    with pytest.raises(RuntimeError, match="boom"):
        rr.run_batch_two_wide([{"slot": "a"}, {"slot": "b"}], run_one,
                               claimable_fn=lambda: True, log=_quiet_log, sleep_fn=lambda s: None)
    assert completed == ["b"]  # sibling still ran despite the other's exception


def test_run_batch_two_wide_odd_length_trailing_item_runs_alone():
    order = []

    def run_one(spec):
        order.append(spec["slot"])
        return spec["slot"]

    out = rr.run_batch_two_wide([{"slot": "a"}, {"slot": "b"}, {"slot": "c"}], run_one,
                                 claimable_fn=lambda: True, log=_quiet_log, sleep_fn=lambda s: None)
    assert set(order) == {"a", "b", "c"}
    assert set(out) == {"a", "b", "c"}


# =============================================================================
# Phase NULLS
# =============================================================================

def test_missing_null_rounds_various_states():
    assert rr.missing_null_rounds(_empty_pulls_df()) == [1, 2]
    assert rr.missing_null_rounds(_pulls([("null", 1, 0.01)])) == [2]
    assert rr.missing_null_rounds(_pulls([("null", 1, 0.01), ("null", 2, -0.01)])) == []
    # a failed row alone does not count as done
    assert rr.missing_null_rounds(_pulls([("null", 1, None, "failed")])) == [1, 2]


def test_run_null_phase_skips_when_both_already_ok(tmp_path):
    fl = FakeLedger(preseed=_pulls([("null", 1, 0.01), ("null", 2, -0.02)]).to_dict("records"))
    sigma_e = rr.run_null_phase(fl.read, fl.run_one, sigma_e_eval=0.03,
                                cfg_path=tmp_path / "config.yaml", log=_quiet_log)
    assert fl.calls == []  # never re-pulled
    assert sigma_e == pytest.approx(max(float(np.std([0.01, -0.02], ddof=1)), 0.03))


def test_run_null_phase_runs_only_the_missing_round(tmp_path):
    fl = FakeLedger(preseed=_pulls([("null", 1, 0.01)]).to_dict("records"),
                     delta_fn=lambda arm, j: -0.015)
    rr.run_null_phase(fl.read, fl.run_one, sigma_e_eval=0.0, log=_quiet_log, cfg_path=tmp_path / 'config.yaml')
    assert fl.calls == [("null", 2)]


def test_run_null_phase_computes_sigma_e_as_max_of_null_std_and_eval_floor(tmp_path):
    fl = FakeLedger(delta_fn=lambda arm, j: {1: 0.20, 2: -0.20}[j])
    sigma_e = rr.run_null_phase(fl.read, fl.run_one, sigma_e_eval=0.01,
                                 claimable_fn=lambda: False, log=_quiet_log, cfg_path=tmp_path / 'config.yaml', sleep_fn=lambda s: None)
    expected_null_std = float(np.std([0.20, -0.20], ddof=1))
    assert expected_null_std > 0.01
    assert sigma_e == pytest.approx(expected_null_std)
    assert fl.calls == [("null", 1), ("null", 2)]


def test_run_null_phase_logs_loud_warning_when_delta_exceeds_threshold(tmp_path):
    logs = []
    fl = FakeLedger(delta_fn=lambda arm, j: 0.2 if j == 1 else 0.0)
    rr.run_null_phase(fl.read, fl.run_one, sigma_e_eval=0.0, log=logs.append, cfg_path=tmp_path / 'config.yaml')
    assert any("LOUD WARNING" in line for line in logs)


def test_run_null_phase_raises_when_still_fewer_than_2_ok_after_running(tmp_path):
    fl = FakeLedger(fail_set={("null", 1), ("null", 2)})
    with pytest.raises(RuntimeError, match="HUMAN INTERVENTION"):
        rr.run_null_phase(fl.read, fl.run_one, sigma_e_eval=0.0, log=_quiet_log, cfg_path=tmp_path / 'config.yaml')


def test_run_null_phase_two_wide_when_claimable(tmp_path):
    fl = FakeLedger()
    # a 2-party barrier proves the 2 missing null pulls actually ran
    # concurrently when claimable_fn says both slots are free.
    barrier = threading.Barrier(2, timeout=2)
    orig_run_one = fl.run_one

    def barriered_run_one(spec):
        barrier.wait()
        return orig_run_one(spec)

    rr.run_null_phase(fl.read, barriered_run_one, sigma_e_eval=0.0,
                       claimable_fn=lambda: True, log=_quiet_log, cfg_path=tmp_path / 'config.yaml')
    assert set(fl.calls) == {("null", 1), ("null", 2)}


def test_run_null_phase_routes_specs_through_resolve_sticky_slots(tmp_path, monkeypatch):
    """Integration point for the sticky-slot fix (task-stickyslot-report.md):
    run_null_phase must hand its alternation-assigned specs through
    pull.resolve_sticky_slots before dispatch, so a resumed run's index-based
    slot pick can never override an arm's real, already-complete checkpoint
    slot. Exercised end-to-end against real checkpoint_looks_complete/
    ckpt_final_dir (config.OPENPI monkeypatched to tmp_path), not by mocking
    resolve_sticky_slots itself."""
    monkeypatch.setattr(config, "OPENPI", tmp_path / "openpi")
    # null_j1's real checkpoint already lives under slot "b" -- the
    # single-missing-round alternation (index 0) would otherwise pick "a".
    ckpt_dir = pull.ckpt_final_dir("pi0_ppc2sink_bandit_b", "null_j1")
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / pull.CKPT_METADATA_FILE).write_text("{}")
    (ckpt_dir / "params").mkdir()
    (ckpt_dir / "params" / "x").write_text("x")

    seen_slots = []
    fl = FakeLedger(preseed=_pulls([("null", 2, -0.01)]).to_dict("records"))

    def run_one(spec):
        seen_slots.append(spec["slot"])
        return fl.run_one(spec)

    logs = []
    rr.run_null_phase(fl.read, run_one, sigma_e_eval=0.0, log=logs.append,
                       cfg_path=tmp_path / "config.yaml")

    assert seen_slots == ["b"]  # forced away from the "a" alternation would have picked
    assert any("sticky slot: null_j1 -> slot b (existing checkpoint)" in m for m in logs)


# =============================================================================
# resume/round-reconstruction helpers (pure)
# =============================================================================

def test_ok_arms_at_round():
    df = _pulls([("A", 3, 0.1), ("B", 3, 0.2, "failed"), ("A", 4, 0.1)])
    assert rr._ok_arms_at_round(df, 3) == {"A"}
    assert rr._ok_arms_at_round(df, 4) == {"A"}
    assert rr._ok_arms_at_round(df, 5) == set()
    assert rr._ok_arms_at_round(_empty_pulls_df(), 3) == set()


def test_current_round_and_roster_bootstraps_at_race_first_round_when_empty():
    j, alive = rr._current_round_and_roster(_empty_pulls_df(), ["A", "B", "random"],
                                             sigma_e=0.01, delta=0.1, t_max=16, log=_quiet_log)
    assert j == rr.RACE_FIRST_ROUND
    assert alive == ["A", "B", "random"]


def test_current_round_and_roster_resume_mid_round_recovers_full_prior_roster():
    """2 of 3 arms already ok at round 3 (simulated crash after a partial
    round); the 3rd must still be considered `alive` for round 3 -- NOT
    silently dropped/eliminated just because it hasn't been pulled yet."""
    df = _pulls([("A", 3, 0.10), ("B", 3, 0.10)])   # "C" missing entirely
    j, alive = rr._current_round_and_roster(df, ["A", "B", "C"], sigma_e=0.01,
                                             delta=0.1, t_max=16, log=_quiet_log)
    assert j == 3
    assert set(alive) == {"A", "B", "C"}


def test_current_round_and_roster_advances_once_round_is_fully_complete():
    # sigma_e deliberately wide so n=1 CIs overlap heavily -- no decisive
    # separation yet, so this exercises the "round complete -> advance"
    # branch rather than an immediate done=True.
    df = _pulls([("A", 3, 0.10), ("B", 3, 0.05)])
    j, alive = rr._current_round_and_roster(df, ["A", "B"], sigma_e=1.0,
                                             delta=0.1, t_max=16, log=_quiet_log)
    assert j == 4
    assert set(alive) == {"A", "B"}


def test_current_round_and_roster_reports_done_when_scheduler_says_so():
    df = _pulls([("A", 3, 0.30), ("B", 3, 0.05), ("A", 4, 0.31), ("B", 4, 0.04),
                 ("A", 5, 0.29), ("B", 5, 0.06)])
    j, decision = rr._current_round_and_roster(df, ["A", "B"], sigma_e=0.01,
                                                delta=0.1, t_max=16, log=_quiet_log)
    assert j is None
    assert decision["done"] is True
    assert decision["survivors"] == ["A"]


# =============================================================================
# run_race_phase: the full resumable loop
# =============================================================================

def test_run_race_phase_clean_separation_converges_to_single_survivor():
    per_round = {"hi": [0.30, 0.31, 0.29], "mid": [0.05, 0.06, 0.04], "lo": [-0.05, -0.04, -0.06]}

    def delta_fn(arm, j):
        return per_round[arm][(j - rr.RACE_FIRST_ROUND) % 3]

    fl = FakeLedger(delta_fn=delta_fn)
    decision = rr.run_race_phase(sigma_e=0.01, read_pulls_fn=fl.read, run_one=fl.run_one,
                                  all_arms=["hi", "mid", "lo"], claimable_fn=lambda: False,
                                  log=_quiet_log, sleep_fn=lambda s: None)
    assert decision["survivors"] == ["hi"]
    assert decision["done"] is True
    assert "mid" in decision["eliminated"]
    assert "lo" in decision["eliminated"]
    # every arm was pulled the same number of rounds up through elimination
    n_rounds = decision["eliminated"]["mid"] - rr.RACE_FIRST_ROUND + 1
    assert fl.calls.count(("hi", rr.RACE_FIRST_ROUND)) == 1
    assert len([c for c in fl.calls if c[0] == "mid"]) == n_rounds
    assert len([c for c in fl.calls if c[0] == "lo"]) == n_rounds
    # mid/lo never pulled again after their elimination round
    assert all(j <= decision["eliminated"]["mid"] for (a, j) in fl.calls if a == "mid")


def test_run_race_phase_t_cap_stops_with_multiple_survivors():
    fl = FakeLedger(delta_fn=lambda arm, j: {"A": 0.10, "B": 0.11}[arm])
    decision = rr.run_race_phase(sigma_e=0.05, read_pulls_fn=fl.read, run_one=fl.run_one,
                                  all_arms=["A", "B"], claimable_fn=lambda: False,
                                  t_max=4, log=_quiet_log, sleep_fn=lambda s: None)
    assert decision["done"] is True
    assert set(decision["survivors"]) == {"A", "B"}
    assert decision["eliminated"] == {}
    assert len(fl.calls) == 4  # exactly t_max ok pulls, no more


def test_run_race_phase_routes_specs_through_resolve_sticky_slots(tmp_path, monkeypatch):
    """Same wiring check as run_null_phase's, for Phase RACE's own dispatch
    site -- the ACTUAL site tonight's round-3 stall hit (task-stickyslot-
    report.md): easy_band's real checkpoint already lives under slot "b",
    but the bootstrap round's index-based alternation (2 arms, index 0)
    would otherwise pick "a" for it."""
    monkeypatch.setattr(config, "OPENPI", tmp_path / "openpi")
    ckpt_dir = pull.ckpt_final_dir("pi0_ppc2sink_bandit_b", "easy_band_j3")
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / pull.CKPT_METADATA_FILE).write_text("{}")
    (ckpt_dir / "params").mkdir()
    (ckpt_dir / "params" / "x").write_text("x")

    seen_slots = {}
    fl = FakeLedger()

    def run_one(spec):
        seen_slots[spec["arm"]] = spec["slot"]
        return fl.run_one(spec)

    logs = []
    rr.run_race_phase(sigma_e=0.01, read_pulls_fn=fl.read, run_one=run_one,
                       all_arms=["easy_band", "mid_band"], claimable_fn=lambda: False,
                       t_max=2, log=logs.append, sleep_fn=lambda s: None)

    assert seen_slots["easy_band"] == "b"  # forced away from the "a" alternation would have picked
    assert seen_slots["mid_band"] == "b"   # no checkpoint anywhere yet -- alternation's own pick stands
    assert any("sticky slot: easy_band_j3 -> slot b (existing checkpoint)" in m for m in logs)


def test_run_batch_two_wide_should_stop_fn_checked_before_each_pull():
    """The literal acceptance test named in the fix brief: cap reached after
    the 2nd of 4 planned pulls -> exactly 0 further run_one calls (not just
    "eventually stops", but stops with zero extra dispatches)."""
    calls = []
    count = {"n": 0}

    def run_one(spec):
        calls.append(spec["slot"])
        count["n"] += 1
        return spec["slot"]

    out = rr.run_batch_two_wide(
        [{"slot": "a"}, {"slot": "b"}, {"slot": "c"}, {"slot": "d"}],
        run_one, claimable_fn=lambda: False, log=_quiet_log,
        should_stop_fn=lambda: count["n"] >= 2, sleep_fn=lambda s: None)

    assert calls == ["a", "b"]   # c, d never started
    assert out == ["a", "b"]


def test_run_batch_two_wide_should_stop_fn_none_never_stops_batch():
    # default behavior (should_stop_fn=None) is unchanged: no early exit.
    calls = []
    rr.run_batch_two_wide([{"slot": "a"}, {"slot": "b"}, {"slot": "c"}],
                           lambda spec: calls.append(spec["slot"]),
                           claimable_fn=lambda: False, log=_quiet_log, sleep_fn=lambda s: None)
    assert calls == ["a", "b", "c"]


def test_run_race_phase_t_cap_checked_mid_round_stops_with_zero_further_calls():
    """run_race_phase-level version of the same acceptance test: t_max=2,
    4 arms planned this round -- after the 2nd ok pull the cap is hit, so
    the 3rd and 4th planned pulls must NEVER be attempted."""
    fl = FakeLedger(delta_fn=lambda arm, j: 0.1)
    decision = rr.run_race_phase(sigma_e=0.05, read_pulls_fn=fl.read, run_one=fl.run_one,
                                  all_arms=["A", "B", "C", "D"], claimable_fn=lambda: False,
                                  t_max=2, log=_quiet_log, sleep_fn=lambda s: None)
    assert fl.calls == [("A", rr.RACE_FIRST_ROUND), ("B", rr.RACE_FIRST_ROUND)]
    assert decision["done"] is True


def test_run_race_phase_t_cap_stops_mid_round_after_elimination_shrinks_roster():
    """Race-runner review Finding 2's exact overshoot scenario: 15 ok pulls
    already banked (1 below t_max=16) with a live 3-arm roster. Before this
    fix, the whole round (3 more pulls) always ran to completion regardless
    of budget, landing at 18 (2 over budget, since the cap was only
    re-checked BETWEEN whole rounds). Now the round stops the instant the
    16th ok pull lands -- exactly 1 of that round's 3 planned pulls runs."""
    preseed_rows = []
    for k in range(5):   # rounds RACE_FIRST_ROUND..+4, 3 tied arms each = 15 ok pulls
        j = rr.RACE_FIRST_ROUND + k
        preseed_rows += [(arm, j, 0.10) for arm in ("A", "B", "C")]
    fl = FakeLedger(preseed=_pulls(preseed_rows).to_dict("records"), delta_fn=lambda arm, j: 0.10)

    # sigma_e=1.0 (deliberately huge): all 3 arms are tied at mean 0.10, so
    # nothing is ever eliminated -- the race can ONLY end via t_max here,
    # isolating the T-cap behavior from elimination logic.
    decision = rr.run_race_phase(sigma_e=1.0, read_pulls_fn=fl.read, run_one=fl.run_one,
                                  all_arms=["A", "B", "C"], claimable_fn=lambda: False,
                                  t_max=16, log=_quiet_log, sleep_fn=lambda s: None)

    assert len(fl.calls) == 1   # NOT the full 3-arm round -- stops right at t_max
    assert decision["done"] is True


def test_run_race_phase_resumes_eval_failed_row_by_repulling_that_arm():
    """An eval_failed row (pull.py's new eval-except ledger row) must be
    treated exactly like a missing/failed pull for resume purposes -- the
    arm gets re-pulled this round; an arm that's genuinely `ok` is not."""
    preseed = _pulls([("A", rr.RACE_FIRST_ROUND, None, "eval_failed"),
                       ("B", rr.RACE_FIRST_ROUND, 0.10)])
    fl = FakeLedger(preseed=preseed.to_dict("records"), delta_fn=lambda arm, j: 0.10)
    rr.run_race_phase(sigma_e=0.05, read_pulls_fn=fl.read, run_one=fl.run_one,
                       all_arms=["A", "B"], claimable_fn=lambda: False,
                       t_max=3, log=_quiet_log, sleep_fn=lambda s: None)
    assert ("A", rr.RACE_FIRST_ROUND) in fl.calls
    assert ("B", rr.RACE_FIRST_ROUND) not in fl.calls


def test_run_race_phase_resumes_mid_round_never_re_pulls_already_ok_arms():
    preseed = _pulls([("A", rr.RACE_FIRST_ROUND, 0.10), ("B", rr.RACE_FIRST_ROUND, 0.10)])
    fl = FakeLedger(preseed=preseed.to_dict("records"),
                     delta_fn=lambda arm, j: 0.10)
    rr.run_race_phase(sigma_e=0.05, read_pulls_fn=fl.read, run_one=fl.run_one,
                       all_arms=["A", "B", "C"], claimable_fn=lambda: False,
                       t_max=3, log=_quiet_log, sleep_fn=lambda s: None)
    assert ("C", rr.RACE_FIRST_ROUND) in fl.calls
    assert ("A", rr.RACE_FIRST_ROUND) not in fl.calls
    assert ("B", rr.RACE_FIRST_ROUND) not in fl.calls


def test_run_race_phase_already_done_on_entry_returns_immediately():
    df = _pulls([("A", 3, 0.30), ("B", 3, 0.05), ("A", 4, 0.31), ("B", 4, 0.04),
                 ("A", 5, 0.29), ("B", 5, 0.06)])
    fl = FakeLedger(preseed=df.to_dict("records"))
    decision = rr.run_race_phase(sigma_e=0.01, read_pulls_fn=fl.read, run_one=fl.run_one,
                                  all_arms=["A", "B"], claimable_fn=lambda: False, log=_quiet_log, sleep_fn=lambda s: None)
    assert decision["done"] is True
    assert fl.calls == []  # nothing left to pull


def test_run_race_phase_halts_loudly_when_arm_never_produces_ok_pull():
    fl = FakeLedger(fail_set={("bad", rr.RACE_FIRST_ROUND)},
                     delta_fn=lambda arm, j: 0.1)
    with pytest.raises(RuntimeError, match="HUMAN INTERVENTION"):
        rr.run_race_phase(sigma_e=0.05, read_pulls_fn=fl.read, run_one=fl.run_one,
                           all_arms=["good", "bad"], claimable_fn=lambda: False, log=_quiet_log, sleep_fn=lambda s: None)
    assert ("bad", rr.RACE_FIRST_ROUND) in fl.calls
    # never silently advances past the failed arm to a later round
    assert all(j == rr.RACE_FIRST_ROUND for (a, j) in fl.calls)


# =============================================================================
# dry_run_report (read-only)
# =============================================================================

_ARMS_SPEC = {"random_arm": True, "arms": [{"index": 0, "name": "A"}, {"index": 1, "name": "B"}]}


def test_dry_run_report_waiting_on_preconditions(tmp_path):
    out = rr.dry_run_report(cfg_path=tmp_path / "missing.yaml",
                             e_manifest_path=tmp_path / "missing.parquet", log=_quiet_log)
    assert out["phase"] == "waiting_preconditions"
    assert out["ready"] is False


def test_dry_run_report_nulls_missing(tmp_path):
    cfg = tmp_path / "config.yaml"
    _write_cfg(cfg, baseline={"b": 0.5, "per_stratum_b": {}, "sigma_e_eval": 0.02})
    manifest = tmp_path / "E_manifest.parquet"
    manifest.write_text("x")

    out = rr.dry_run_report(read_pulls_fn=_empty_pulls_df, cfg_path=cfg,
                             e_manifest_path=manifest, log=_quiet_log)
    assert out == {"phase": "nulls", "missing": [1, 2]}


def test_dry_run_report_race_in_progress(tmp_path):
    cfg = tmp_path / "config.yaml"
    _write_cfg(cfg, baseline={"b": 0.5, "per_stratum_b": {}, "sigma_e_eval": 0.02})
    manifest = tmp_path / "E_manifest.parquet"
    manifest.write_text("x")

    df = _pulls([("null", 1, 0.01), ("null", 2, -0.01), ("A", 3, 0.10)])
    out = rr.dry_run_report(read_pulls_fn=lambda: df, cfg_path=cfg, e_manifest_path=manifest,
                             arms_spec=_ARMS_SPEC, log=_quiet_log)
    assert out["phase"] == "race"
    assert out["round"] == 3
    assert set(out["to_pull"]) == {"B", "random"}


def test_dry_run_report_never_writes_anything(tmp_path):
    cfg = tmp_path / "config.yaml"
    _write_cfg(cfg, baseline={"b": 0.5, "per_stratum_b": {}, "sigma_e_eval": 0.02})
    manifest = tmp_path / "E_manifest.parquet"
    manifest.write_text("x")
    before = cfg.read_text()

    rr.dry_run_report(read_pulls_fn=_empty_pulls_df, cfg_path=cfg,
                       e_manifest_path=manifest, log=_quiet_log)
    assert cfg.read_text() == before
    assert not (tmp_path / "pulls.parquet").exists()


# =============================================================================
# _make_eval_fn: parallel-by-default (task-ledgerlock-report.md, 2026-07-29)
# + always-resumable (emergency null-takeover fix; resume=True closes the
# "killed mid-eval, rerun redoes/loses rows" gap regardless of worker count).
# EVAL_WORKERS was serial-only (None) while two prerequisites were missing:
# a supervised real parallel wave (now done -- easy_band_j3, 4 workers, 171
# rollouts, clean merge) and a concurrency-safe ledger.append_rows (now
# landed -- cross-process fcntl.flock, see ledger.py/test_ledger.py). Both
# now hold, so the module default flipped to 4 -- see run_race.py's
# EVAL_WORKERS comment for the full history and reasoning.
# =============================================================================

def test_eval_workers_module_default_is_4_parallel():
    """The loud, hard-stop assertion this whole fix hinges on: whatever
    EVAL_WORKERS is set to, it must be 4 (parallel) until a human
    deliberately revisits it -- a regression here would silently re-enable
    the exact hang this fix was written to stop.

    Once serial-only, note well: since 2026-07-29 (task-ledgerlock-
    report.md) the prerequisites that kept this at None -- one supervised
    parallel wave + a concurrency-safe ledger.append_rows -- have both
    landed, so the flipped-to-4 value is the current, deliberate contract,
    not a regression to guard against."""
    assert rr.EVAL_WORKERS == 4


def test_make_eval_fn_default_workers_is_4_and_resume_always_true(monkeypatch):
    seen = {}

    def fake_eval_checkpoint(port, policy_id, arm, pull_id, workers=None, resume=False):
        seen.update(port=port, policy_id=policy_id, arm=arm, pull_id=pull_id,
                    workers=workers, resume=resume)
        return "RESULT"

    monkeypatch.setattr(rr.eval_set, "eval_checkpoint", fake_eval_checkpoint)

    eval_fn = rr._make_eval_fn()
    out = eval_fn(9999, "null_j1", "null", "null_j1")

    assert out == "RESULT"
    assert seen == {"port": 9999, "policy_id": "null_j1", "arm": "null", "pull_id": "null_j1",
                     "workers": 4, "resume": True}


def test_make_eval_fn_explicit_workers_override_still_forces_resume(monkeypatch):
    """A caller that overrides `workers` away from the module default (e.g.
    a one-off serial fallback) must still get resume=True -- resumability is
    not tied to the serial/parallel choice either way."""
    seen = {}

    def fake_eval_checkpoint(port, policy_id, arm, pull_id, workers=None, resume=False):
        seen.update(workers=workers, resume=resume)
        return "RESULT"

    monkeypatch.setattr(rr.eval_set, "eval_checkpoint", fake_eval_checkpoint)

    eval_fn = rr._make_eval_fn(workers=None)
    eval_fn(9999, "some_pull", "targeted", "some_pull")

    assert seen == {"workers": None, "resume": True}


def test_main_default_workers_is_eval_workers_default(monkeypatch):
    """main()'s own `workers=EVAL_WORKERS` default must still resolve to 4
    -- guards against the default drifting apart from the module constant
    if either is edited independently in the future."""
    import inspect
    assert inspect.signature(rr.main).parameters["workers"].default == 4


# =============================================================================
# main(): top-level wiring order (submodule loads monkeypatched)
# =============================================================================

def test_main_wires_gradient_note_preconditions_nulls_and_race_in_order(tmp_path, monkeypatch):
    order = []

    cfg = tmp_path / "ledger" / "config.yaml"
    cfg.parent.mkdir(parents=True)
    _write_cfg(cfg, baseline={"b": 0.5, "per_stratum_b": {"easy": 0.6, "mid": 0.5, "hard": 0.4},
                              "sigma_e_eval": 0.01})
    doc = yaml.safe_load(cfg.read_text())
    doc["arms_freeze"] = {"B": 200}
    cfg.write_text(yaml.safe_dump(doc, sort_keys=False))
    (tmp_path / "ledger" / "E_manifest.parquet").write_text("x")

    monkeypatch.setattr(rr.config, "LEDGER_DIR", tmp_path / "ledger")

    def fake_append_note(log=lambda *a: None):
        order.append("gradient_note")
    monkeypatch.setattr(rr.pull, "append_gradient_analysis_note_to_config_yaml", fake_append_note)

    def fake_wait(**kw):
        order.append("preconditions")
    monkeypatch.setattr(rr, "wait_for_preconditions", fake_wait)

    monkeypatch.setattr(rr.pool, "build_pool_table", lambda write=False: "POOL")
    monkeypatch.setattr(rr.map_fit, "load", lambda: "MODELS")
    monkeypatch.setattr(rr.clustering, "load_arms_yaml", lambda: _ARMS_SPEC)
    monkeypatch.setattr(rr.wells, "assign_regions", lambda pool_df, models, spec: "REGIONS")
    monkeypatch.setattr(rr.eval_set, "load_manifest", lambda: "E_FEATURES")

    fl = FakeLedger(delta_fn=lambda arm, j: 0.1)

    def fake_null_phase(read_pulls_fn, run_one, sigma_e_eval, **kw):
        order.append("nulls")
        return 0.05

    def fake_race_phase(sigma_e, read_pulls_fn, run_one, all_arms, **kw):
        order.append("race")
        return {"done": True, "survivors": ["A"]}

    monkeypatch.setattr(rr, "run_null_phase", fake_null_phase)
    monkeypatch.setattr(rr, "run_race_phase", fake_race_phase)

    decision = rr.main(log=_quiet_log, read_pulls_fn=fl.read, run_pull_fn=lambda *a, **k: None,
                        claimable_fn=lambda: False, sleep_fn=lambda s: None)

    assert order == ["gradient_note", "preconditions", "nulls", "race"]
    assert decision == {"done": True, "survivors": ["A"]}
