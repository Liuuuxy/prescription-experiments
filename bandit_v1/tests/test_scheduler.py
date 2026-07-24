"""Tests for bandit_v1/scheduler.py (Task 14, step 1 -- pure module only).

Covers the brief's four required scenarios (a)-(d) plus the two additional
contract points named in the task instructions: the "null" arm is a noise
measurement excluded everywhere -- including the t_max budget -- and rows
whose status isn't "ok" (e.g. "smoke", "failed") must never count. No live
env, no ledger I/O, no filesystem: decide() is a pure function of its
pulls_df argument, so every test below builds a synthetic DataFrame by hand.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from bandit_v1 import scheduler


def _pulls(rows):
    """rows: list of (arm, round_j, delta[, status]) tuples; status defaults
    to "ok". Returns a DataFrame with the committed pulls.parquet columns
    this module actually reads (pull_id, arm, round_j, delta, status)."""
    out = []
    for i, r in enumerate(rows):
        arm, round_j, delta = r[0], r[1], r[2]
        status = r[3] if len(r) > 3 else "ok"
        out.append({"pull_id": i, "arm": arm, "round_j": round_j,
                     "delta": delta, "status": status})
    return pd.DataFrame(out)


def _z(delta=0.1):
    return float(norm.ppf(1 - delta / 2))


# --- (a) clearly-separated arms -> single survivor within 3 rounds. --------

def test_clearly_separated_arms_single_survivor_within_3_rounds():
    per_round = [(0.30, 0.05, -0.05), (0.31, 0.06, -0.04), (0.29, 0.04, -0.06)]
    rows = []
    for j, (hi, mid, lo) in enumerate(per_round, start=1):
        rows += [("hi", j, hi), ("mid", j, mid), ("lo", j, lo)]
    df = _pulls(rows)

    d = scheduler.decide(df, sigma_e=0.01, delta=0.1, t_max=16)

    assert d["survivors"] == ["hi"]
    assert d["done"] is True
    assert d["next_round"] is None
    assert d["eliminated"] == {"mid": 3, "lo": 3}
    assert [r[0] for r in d["ranking"]] == ["hi", "mid", "lo"]


# --- (b) all-equal arms -> no elimination, budget exhaustion, done=True ----
#     at t_max with full ranking.

def test_all_equal_arms_budget_exhaustion_no_elimination():
    same_deltas = [0.10, 0.12, 0.08]
    rows = [(arm, j, v) for arm in ("A", "B", "C")
            for j, v in enumerate(same_deltas, start=1)]
    df = _pulls(rows)

    d = scheduler.decide(df, sigma_e=0.05, delta=0.1, t_max=9)  # 3 arms x 3 rounds = 9

    assert d["eliminated"] == {}
    assert set(d["survivors"]) == {"A", "B", "C"}
    assert d["done"] is True
    assert d["next_round"] is None
    assert {r[0] for r in d["ranking"]} == {"A", "B", "C"}
    assert len(d["ranking"]) == 3
    # identical means -> everyone is inside the leader's noise floor.
    leader = d["ranking"][0][0]
    assert set(d["tied_with_leader"]) == {"A", "B", "C"} - {leader}


# --- (c) sigma floor applied when n_k == 1 (sample std undefined). --------

def test_sigma_floor_applied_at_n_equals_1():
    df = _pulls([("A", 1, 0.20), ("B", 1, 0.05)])
    sigma_e = 0.10

    d = scheduler.decide(df, sigma_e=sigma_e, delta=0.1, t_max=16)

    hw = _z(0.1) * sigma_e / np.sqrt(1)
    ranking = {r[0]: r for r in d["ranking"]}
    assert ranking["A"][1] == pytest.approx(0.20)
    assert ranking["A"][2] == pytest.approx(0.20 - hw)
    assert ranking["A"][3] == pytest.approx(0.20 + hw)
    assert ranking["B"][2] == pytest.approx(0.05 - hw)
    assert ranking["B"][3] == pytest.approx(0.05 + hw)

    # Without the floor (sigma_hat=0 at n=1), B's ucb (0.05) would fall
    # below A's lcb (0.20) and B would be wrongly cut on a single pull. With
    # the sigma_e floor applied the CIs are wide enough to overlap.
    assert d["eliminated"] == {}
    assert set(d["survivors"]) == {"A", "B"}
    assert d["done"] is False
    assert d["next_round"] == 2


# --- (d) elimination never removes the current leader. --------------------

def test_leader_is_never_eliminated_even_when_a_tied_arm_has_tighter_ci():
    # A and B are tied at mean 0.15; A's own CI is wide (std=0.1) while B's
    # is razor-tight (std=0), so the max lcb across the cohort is set by B,
    # NOT by A -- even though A is picked as leader (max mean, first on a
    # tie, since groupby iterates "A" before "B"). The explicit guard must
    # keep A safe no matter whose lcb set the elimination bar.
    df = _pulls([
        ("A", 1, 0.05), ("A", 2, 0.15), ("A", 3, 0.25),
        ("B", 1, 0.15), ("B", 2, 0.15), ("B", 3, 0.15),
    ])
    d = scheduler.decide(df, sigma_e=0.02, delta=0.1, t_max=16)

    leader = d["ranking"][0][0]
    assert leader == "A"
    assert leader in d["survivors"]
    assert leader not in d["eliminated"]
    assert d["eliminated"] == {}          # B's own ucb clears the bar too


# --- the "null" arm is a noise measurement, not a competing arm. -----------

def test_null_arm_excluded_from_survivors_elimination_ranking_and_budget():
    # A and B are close (not decisively separated), so this exercises the
    # budget path specifically rather than "1 survivor"/"decisive" done.
    rows = [("A", 1, 0.12), ("A", 2, 0.13), ("B", 1, 0.08), ("B", 2, 0.09)]
    # 20 null rows: if wrongly counted toward the t_max budget these alone
    # would exceed it, even though the real arms only have 4 ok pulls.
    rows += [("null", j, -10.0) for j in range(1, 21)]
    df = _pulls(rows)

    d = scheduler.decide(df, sigma_e=0.05, delta=0.1, t_max=16)

    assert "null" not in d["survivors"]
    assert "null" not in d["eliminated"]
    assert {r[0] for r in d["ranking"]} == {"A", "B"}
    assert set(d["survivors"]) == {"A", "B"}
    # Budget must be computed over the 4 real-arm ok pulls only, not 24 --
    # otherwise this would wrongly be done=True at t_max=16.
    assert d["done"] is False
    assert d["next_round"] == 3


# --- status != "ok" rows (e.g. "smoke", "failed") must be ignored. ---------

def test_non_ok_status_rows_are_ignored():
    rows = [("A", 1, 0.30), ("A", 2, 0.31), ("B", 1, 0.05), ("B", 2, 0.06)]
    d_clean = scheduler.decide(_pulls(rows), sigma_e=0.01, delta=0.1, t_max=16)

    # Extreme-valued smoke/failed rows that would flip the leader and blow
    # the budget if they were ever counted.
    noisy_rows = list(rows) + [("A", 1, -100.0, "smoke"), ("B", 1, 100.0, "failed")] * 10
    d_noisy = scheduler.decide(_pulls(noisy_rows), sigma_e=0.01, delta=0.1, t_max=16)

    assert d_noisy["ranking"] == d_clean["ranking"]
    assert d_noisy["survivors"] == d_clean["survivors"]
    assert d_noisy["eliminated"] == d_clean["eliminated"]
    assert d_noisy["done"] == d_clean["done"]
    assert d_noisy["next_round"] == d_clean["next_round"]
