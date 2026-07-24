"""Successive-elimination scheduler for bandit_v1 (Task 14, pure module).

Design: weakregion/BANDIT_V1_DESIGN.md section 4 ("Scheduler -- successive
elimination") + section 9 (invariants: "measure sigma_e before interpreting
any ranking; differences inside it are ties"). This module IS the scheduler:
a single pure function, `decide`, that reads the pulls ledger (or any
pulls_df sharing its columns) and returns a Decision -- who survives, who is
cut and when, whether the race is over, and the current ranking. It has no
I/O, no RNG, no clock, and no persisted state: called twice with the same
pulls_df it returns the same answer both times, so `run_race.py` (Task 12+,
not part of this task -- the pull orchestrator it depends on does not exist
yet) can simply call it fresh after every round with whatever the ledger
currently holds.

Row filtering (first thing `decide()` does, every call):
  - `status != "ok"` rows (e.g. "smoke", "failed") never count -- they
    measure whether a pull's plumbing worked, not an arm's value.
  - `arm == "null"` rows never count either. The 2 null pulls (recipe R on
    D0 alone, no extra demos -- design doc section 1 item 6 / section 8) are
    a NOISE MEASUREMENT, not a competing arm: excluded from survivors,
    elimination, and ranking, AND from the `t_max` pull budget. The design
    doc's own budget language is explicit about the last part: "T = 12-16
    pulls (plus `pi_0`, 2 null pulls, ...)" -- null pulls are additional to
    T, not counted against it.

Per-arm statistics (every remaining arm, over ALL of its own ok rows in the
given `pulls_df` -- pulls accumulate across rounds, so an arm's mean/CI
sharpens every round it keeps surviving):
    mean      = mean(delta)
    sigma_hat = max(sample_std(delta, ddof=1) if n >= 2 else 0, sigma_e)
                (n == 1: sample std is undefined, so the noise floor alone
                governs -- this is the "sigma floor" case.)
    half_w    = z * sigma_hat / sqrt(n),   z = norm.ppf(1 - delta / 2)
                (delta=0.1 -> z ~= 1.6449, matching the brief's stated 1.645)
    lcb, ucb  = mean -+ half_w

Cohort / elimination bookkeeping -- how "eliminated: {arm: round}" stays a
stable, monotonic record across repeated *stateless* calls: the "cohort" for
THIS call is every non-null arm whose most recent ok pull is at the overall
max `round_j` -- i.e. every arm still actually being pulled. Any other arm
(one whose last ok pull is at an EARLIER round) was already cut by a PRIOR
call to `decide()`, because `run_race.py` only pulls arms `decide()`
returned as survivors -- it is reported as eliminated at ITS OWN last round,
which is exactly the round whose data justified cutting it, and is stable:
it never changes on a later call, since that arm never accrues new rows.
Cohort arms alone are eligible to be this call's leader or to be newly
eliminated.

Elimination rule and the leader guard: `max_lcb` = the max lcb across ALL
cohort arms (this can come from any arm, not necessarily the one with the
top mean -- an arm with a slightly lower mean but a much tighter CI can have
a higher lcb). `leader` = the cohort arm with the max MEAN. A non-leader
cohort arm is newly eliminated iff its ucb < max_lcb, timestamped at the
overall max round. The leader itself is explicitly exempted from this test
before it is ever evaluated -- never inferred from the arithmetic -- per the
brief's "guard explicitly; degenerate all-equal case": with exactly (or
near-exactly, in float) tied means, a naive implementation could compute
`max_lcb` from a DIFFERENT tied arm and mis-flag the nominal leader.

`ranking` always lists every arm that has ever appeared (cohort or already
cut), each with its own (frozen, if cut) mean/CI, sorted by mean descending
-- the full leaderboard design section 7.1 wants, not just the live cohort.

`tied_with_leader` is the subset of SURVIVING cohort arms (excluding the
leader itself) additionally inside the noise floor of the leader's mean
(`abs(mean - leader_mean) <= sigma_e`) -- used by the exploit phase's
proportional split (design section 5) when the race ends in a statistical
tie rather than a lone winner.

`done` is True iff: exactly one survivor remains; OR `max_lcb` clears every
surviving rival's ucb (decisive separation -- note this can only actually
fire when the max-lcb-contributing arm coincides with the mean-leader,
since any OTHER cohort arm's own ucb is never below its own lcb, so it can
never be excluded via a threshold equal to its own lcb); OR the arm (non-
null) pull budget -- the count of ok, non-null rows -- has reached `t_max`.
In the budget-exhaustion case there may still be >1 survivor and/or ties --
`ranking` and `tied_with_leader` are reported exactly the same way
regardless of which condition ended the race.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config


def decide(pulls_df: pd.DataFrame, sigma_e: float,
           delta: float = config.DELTA_CONF,
           t_max: int = config.T_MAX_PULLS) -> dict:
    ok = pulls_df[(pulls_df["status"] == "ok") & (pulls_df["arm"] != "null")]

    if ok.empty:
        return {"survivors": [], "eliminated": {}, "next_round": 1,
                "done": False, "ranking": [], "tied_with_leader": []}

    total_ok_pulls = len(ok)
    max_round = int(ok["round_j"].max())
    z = float(norm.ppf(1 - delta / 2))

    stats = {}
    last_round = {}
    for arm, g in ok.groupby("arm"):
        n = len(g)
        mean = float(g["delta"].mean())
        std = float(g["delta"].std(ddof=1)) if n >= 2 else 0.0
        if np.isnan(std):
            std = 0.0
        sigma_hat = max(std, sigma_e)
        half_w = z * sigma_hat / np.sqrt(n)
        stats[arm] = {"mean": mean, "lcb": mean - half_w, "ucb": mean + half_w, "n": n}
        last_round[arm] = int(g["round_j"].max())

    # Cohort = arms still actually being pulled (last row at the overall max
    # round). Arms whose last row is earlier were already cut by a prior
    # (stateless) call and keep their original elimination round forever.
    cohort = [a for a in stats if last_round[a] == max_round]
    eliminated = {a: last_round[a] for a in stats if a not in cohort}

    leader = max(cohort, key=lambda a: stats[a]["mean"])
    max_lcb = max(stats[a]["lcb"] for a in cohort)

    survivors = []
    for a in cohort:
        if a == leader:
            survivors.append(a)                 # guard: never eliminate the leader
            continue
        if stats[a]["ucb"] < max_lcb:
            eliminated[a] = max_round
        else:
            survivors.append(a)

    tied_with_leader = [
        a for a in survivors
        if a != leader and abs(stats[a]["mean"] - stats[leader]["mean"]) <= sigma_e
    ]

    rival_ucbs = [stats[a]["ucb"] for a in survivors if a != leader]
    decisive = not rival_ucbs or max_lcb > max(rival_ucbs)

    done = len(survivors) == 1 or decisive or total_ok_pulls >= t_max
    next_round = None if done else max_round + 1

    ranking = sorted(
        ((a, s["mean"], s["lcb"], s["ucb"], s["n"]) for a, s in stats.items()),
        key=lambda row: row[1],
        reverse=True,
    )

    return {
        "survivors": survivors,
        "eliminated": eliminated,
        "next_round": next_round,
        "done": done,
        "ranking": ranking,
        "tied_with_leader": tied_with_leader,
    }
