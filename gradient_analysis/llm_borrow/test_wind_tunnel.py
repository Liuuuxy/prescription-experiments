"""Unit tests for the allocator wind tunnel — ALL synthetic, ZERO GPU.

Run BEFORE any GPU work:
  cd /data/xinyua11/robocasa/gradient_analysis/llm_borrow
  /data/xinyua11/conda/envs/robocasa/bin/python -m pytest test_wind_tunnel.py -q

What is verified here (in the order the task demands):
  A. Fairness accounting  — every operation is metered, in one unit; cache hits are
     still charged; 20 cheap probes are NOT free; nobody exceeds budget.
  B. Oracle optimality    — the oracle's regret is exactly 0 and it weakly dominates
     every other allocator on every instance (so it is a valid ceiling).
  C. The wind tunnel can EXPRESS our five measured constraints (seed-pairing,
     proxy mis-ranking, loss-progress/outcome decoupling) — otherwise a borrowed
     method could "win" here for reasons that cannot happen on the real stack.

Nothing here touches the bandit ledger, the real config, or a GPU.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wind_tunnel import (  # noqa: E402
    ALLOCATOR_FACTORIES, BudgetExhausted, FULL_PULL_UNITS, FULL_STEPS, Jest,
    JestThenConfirm, LearningProgress, Meter, Oracle, OracleToken, RandomAlloc,
    RegMix, SuccessiveHalving, SyntheticBackend, SyntheticSpec, WindTunnel,
    fte, fwd_units, regret_of, run_race, units_of, value_of,
)

# ------------------------------------------------------------------ fixtures


def toy_spec(**kw):
    """A 5-arm synthetic instance with KNOWN values (arm 'good' is the answer)."""
    d = dict(
        values={"null": -0.005, "good": 0.038, "random": 0.009,
                "bad": -0.004, "meh": 0.003},
        sigma_eps=0.004, sigma_seed=0.006,
        # low-fidelity proxies rank 'bad' first  (measured constraint 3)
        proxy_bias={"bad": 0.060, "good": -0.030},
        # loss progress says 'bad'    (measured constraint F4 / 1)
        lp_reward={"null": 0.10, "good": 0.30, "random": 0.20, "bad": 0.55, "meh": 0.25},
        jest_score={"null": 0.0, "good": 0.9, "random": 0.4, "bad": 0.1, "meh": 0.3},
    )
    d.update(kw)
    return SyntheticSpec(**d)


def env_for(spec, budget=6.0, token=None, name="t", seed_pool=None):
    return WindTunnel(SyntheticBackend(spec), budget_fte=budget, name=name,
                      oracle_token=token, seed_pool=seed_pool)


def all_allocators(token):
    return [f(token) for f in ALLOCATOR_FACTORIES.values()]


# ------------------------------------------------------- A. fairness accounting


def test_full_pull_costs_exactly_one_fte():
    env = env_for(toy_spec(), budget=3.0)
    assert env.pull_cost_fte(FULL_STEPS) == pytest.approx(1.0)
    env.pull("good", seed=0)
    assert env.spent_fte == pytest.approx(1.0)
    assert env.remaining_fte == pytest.approx(2.0)


def test_units_and_fte_are_inverse():
    for x in (0.0, 0.137, 1.0, 12.5):
        assert fte(units_of(x)) == pytest.approx(x)
    assert fwd_units(128 * 3) == pytest.approx(1.0)   # 384 forwards == 1 train step


def test_meter_rejects_overspend_and_stays_consistent():
    m = Meter(1.0)
    m.charge(FULL_PULL_UNITS * 0.6, "pull")
    before = m.spent_units
    with pytest.raises(BudgetExhausted):
        m.charge(FULL_PULL_UNITS * 0.6, "pull")
    assert m.spent_units == before                 # refused charge left NO trace
    assert len(m.charges) == 1
    m.charge(FULL_PULL_UNITS * 0.4, "pull")        # exact fill is allowed
    assert m.spent_fte == pytest.approx(1.0)


def test_ledger_sums_to_spend_and_splits_by_kind():
    env = env_for(toy_spec(), budget=5.0)
    env.pull("good", seed=0)
    env.pull("bad", seed=0, steps=250)
    s = env.open_session(seed=1)
    s.burst("good", 100)
    ref = env.reference(steps=500, seed=2)
    env.learnability(ref, n_per_arm=256)
    m = env.meter
    assert sum(c.units for c in m.charges) == pytest.approx(m.spent_units)
    assert sum(m.by_kind().values()) == pytest.approx(m.spent_fte)
    assert set(m.by_kind()) == {"pull", "burst", "loss_probe", "reference", "score"}


def test_cache_hit_is_still_charged():
    """A repeated (arm, seed, steps) is the SAME fine-tune: free in wall-clock,
    but it must still cost budget, or an allocator could farm the cache."""
    env = env_for(toy_spec(), budget=5.0)
    a = env.pull("good", seed=0)
    b = env.pull("good", seed=0)
    assert a.y == b.y and b.cached and not a.cached
    assert env.spent_fte == pytest.approx(2.0)


def test_cache_replay_noise_models_measured_nondeterminism():
    """MEASURED on the box: the identical pull re-run gives 3.70/2.85/3.35pp.
    With replay_noise_sd on, a repeat must behave like the fresh draw it is --
    and still be charged."""
    b = SyntheticBackend(toy_spec())
    b.replay_noise_sd = 0.0043
    env = WindTunnel(b, budget_fte=5.0)
    a = env.pull("good", seed=0)
    c = env.pull("good", seed=0)
    d = env.pull("good", seed=0)
    assert a.y != c.y != d.y
    assert abs(c.y - a.y) < 0.05
    assert env.spent_fte == pytest.approx(3.0)


def test_twenty_cheap_probes_are_not_free():
    """The headline fairness rule: a method that needs 20 cheap probes pays."""
    env = env_for(toy_spec(), budget=20.0)
    for i in range(20):
        env.pull("random", seed=i, steps=250)
    exp = 20 * (250 + fwd_units(30000)) / FULL_PULL_UNITS
    assert env.spent_fte == pytest.approx(exp)
    assert env.spent_fte > 3.0                      # ~3.2 real pulls, not "free"


@pytest.mark.parametrize("op", ["pull", "probe", "burst", "loss", "reference",
                                "score", "session_eval"])
def test_every_env_op_charges(op):
    env = env_for(toy_spec(), budget=6.0)
    before = env.spent_units
    if op == "pull":
        env.pull("good", seed=0)
    elif op == "probe":
        env.pull("good", seed=0, steps=100)
    elif op == "burst":
        env.open_session(seed=0).burst("good", 50)
    elif op == "loss":
        env.open_session(seed=0).loss(n=512)
    elif op == "reference":
        env.reference(steps=200, seed=0)
    elif op == "score":
        env.learnability(env.reference(steps=200, seed=0))
    elif op == "session_eval":
        env.open_session(seed=0).evaluate()
    assert env.spent_units > before, f"{op} was FREE — accounting hole"


def test_op_refused_when_it_would_overrun():
    env = env_for(toy_spec(), budget=0.5)
    with pytest.raises(BudgetExhausted):
        env.pull("good", seed=0)                     # 1.0 FTE > 0.5 remaining
    assert env.spent_fte == 0.0
    assert env.backend.n_compute == 0                # and no compute was done


def test_no_allocator_exceeds_its_budget():
    tok = OracleToken()
    for inst in range(6):
        spec = toy_spec(seed=inst)
        for alloc in all_allocators(tok):
            env = env_for(spec, budget=6.0, token=tok, name=alloc.name)
            alloc.run(env, np.random.default_rng(inst))
            assert env.spent_fte <= 6.0 + 1e-9, f"{alloc.name} overspent"


def test_every_allocator_returns_a_valid_simplex_decision():
    tok = OracleToken()
    spec = toy_spec()
    for alloc in all_allocators(tok):
        env = env_for(spec, budget=6.0, token=tok, name=alloc.name)
        d = alloc.run(env, np.random.default_rng(0))
        assert set(d.weights) <= set(spec.values), alloc.name
        assert min(d.weights.values()) >= -1e-12, alloc.name
        assert sum(d.weights.values()) == pytest.approx(1.0), alloc.name


def test_allocators_are_deterministic_given_seed():
    tok = OracleToken()
    spec = toy_spec()
    for alloc in all_allocators(tok):
        outs = []
        for _ in range(2):
            env = env_for(spec, budget=6.0, token=tok, name=alloc.name)
            d = alloc.run(env, np.random.default_rng(3))
            outs.append((tuple(sorted(d.weights.items())), round(env.spent_fte, 9)))
        assert outs[0] == outs[1], alloc.name


# --------------------------------------------------------- B. oracle optimality


def test_truth_is_gated_by_token():
    tok = OracleToken()
    env = env_for(toy_spec(), budget=1.0, token=tok)
    with pytest.raises(PermissionError):
        env.truth(None)
    with pytest.raises(PermissionError):
        env.truth(OracleToken())                    # a forged token is not the token
    assert env.truth(tok)["good"] == pytest.approx(0.038)
    assert env.spent_fte == 0.0                     # the ceiling spends nothing


def test_oracle_regret_is_zero_and_weakly_dominates_everyone():
    tok = OracleToken()
    rng = np.random.default_rng(0)
    for inst in range(12):
        vals = {f"a{i}": float(v) for i, v in enumerate(rng.normal(0, 0.02, 5))}
        spec = toy_spec(values=vals, proxy_bias=None, lp_reward=None, jest_score=None)
        truth = spec.values
        env = env_for(spec, budget=6.0, token=tok, name="oracle")
        d = Oracle(tok).run(env, np.random.default_rng(inst))
        assert regret_of(d, truth) == pytest.approx(0.0)
        assert value_of(d, truth) == pytest.approx(max(truth.values()))
        for alloc in all_allocators(tok):
            e2 = env_for(spec, budget=6.0, token=tok, name=alloc.name)
            d2 = alloc.run(e2, np.random.default_rng(inst))
            assert value_of(d, truth) >= value_of(d2, truth) - 1e-12, alloc.name


def test_random_floor_is_worse_than_a_competent_allocator_when_signal_is_clean():
    """The harness must be able to express SKILL, or regret is meaningless."""
    spec = toy_spec(sigma_eps=0.001, sigma_seed=0.001)
    sh, rd = [], []
    for i in range(24):
        e1 = env_for(spec, budget=6.0, name="sh")
        sh.append(regret_of(SuccessiveHalving().run(e1, np.random.default_rng(i)), spec.values))
        e2 = env_for(spec, budget=6.0, name="rand")
        rd.append(regret_of(RandomAlloc().run(e2, np.random.default_rng(i)), spec.values))
    assert np.mean(sh) < np.mean(rd)
    assert np.mean(sh) < 0.002


# ------------------------------------- C. the five measured constraints are expressible


def test_seed_effect_is_shared_across_arms_within_a_seed():
    """Constraint 4: the SEED steers the run. Paired-seed differences must have
    lower variance than unpaired ones, or the wind tunnel cannot reward pairing."""
    spec = toy_spec(sigma_eps=0.002, sigma_seed=0.02)
    env = env_for(spec, budget=400.0)
    paired, unpaired = [], []
    for s in range(40):
        a = env.pull("good", seed=s).y
        b = env.pull("random", seed=s).y
        c = env.pull("random", seed=100 + s).y
        paired.append(a - b)
        unpaired.append(a - c)
    assert np.std(paired) < 0.6 * np.std(unpaired)
    assert abs(np.mean(paired) - (spec.values["good"] - spec.values["random"])) < 0.01


def test_proxy_misranking_is_expressible():
    """Constraint 3: cheap short-training proxies MIS-RANK. At 250 steps the
    planted bias must flip the ranking; at full fidelity it must not."""
    spec = toy_spec(sigma_eps=0.0, sigma_seed=0.0)
    env = env_for(spec, budget=200.0)
    low = {a: env.pull(a, seed=0, steps=250).y for a in spec.values}
    full = {a: env.pull(a, seed=0, steps=FULL_STEPS).y for a in spec.values}
    assert max(low, key=low.get) == "bad"
    assert max(full, key=full.get) == "good"


def test_learning_progress_reward_can_be_decoupled_from_outcome():
    """Constraint F4: loss-based progress need not rank arms like closed-loop value."""
    spec = toy_spec(sigma_eps=0.0, sigma_seed=0.0)
    env = env_for(spec, budget=200.0)
    s = env.open_session(seed=0)
    prog = {}
    for a in spec.values:
        s2 = env.open_session(seed=0)
        b = s2.burst(a, 200)
        prog[a] = b["progress"]
    assert max(prog, key=prog.get) == "bad"          # loss says 'bad'
    assert max(spec.values, key=spec.values.get) == "good"   # truth says 'good'
    del s


def test_run_race_reports_spend_and_normalized_regret():
    tok = OracleToken()
    spec = toy_spec()
    res = run_race(SyntheticBackend(spec), budget_fte=6.0, reps=3,
                   token=tok, verbose=False)
    assert res["oracle"]["mean_regret"] == pytest.approx(0.0)
    for name, r in res.items():
        assert r["mean_spent_fte"] <= 6.0 + 1e-9
        assert r["budget_fte"] == 6.0
        assert set(r["spend_by_kind"]) <= {"pull", "burst", "loss_probe",
                                           "reference", "score", "eval"}
    assert res["random"]["norm_regret"] == pytest.approx(1.0)
