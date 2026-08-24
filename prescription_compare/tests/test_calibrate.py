import numpy as np
import pytest

from prescription_compare.calibrate import (
    robocasa_env, anchor_allocations, fit_fidelity, GROUPS, ROBOCASA_OFFLINE_ONLINE,
)
from prescription_compare.allocators.oracle import Oracle
from prescription_compare.allocators.pfail import PFail


def test_fit_fidelity_is_negative_for_anticorrelated_metrics():
    offline = [0.5, 0.6, 0.7, 0.8]        # higher offline "quality"
    deployed = [0.40, 0.35, 0.30, 0.25]   # ... lower deployed success
    assert fit_fidelity(offline, deployed) == pytest.approx(-1.0, abs=1e-9)


def test_fit_fidelity_is_positive_for_correlated_metrics():
    offline = [0.5, 0.6, 0.7, 0.8]
    deployed = [0.25, 0.30, 0.35, 0.40]
    assert fit_fidelity(offline, deployed) == pytest.approx(1.0, abs=1e-9)


def test_fit_fidelity_is_zero_when_deployed_is_flat():
    # ties must be handled (average ranks) or a flat column spuriously correlates
    offline = [0.5, 0.6, 0.7, 0.8]
    deployed = [0.3, 0.3, 0.3, 0.3]
    assert fit_fidelity(offline, deployed) == pytest.approx(0.0, abs=1e-9)


def test_robocasa_fidelity_is_negative_from_measured_anticorrelation():
    # grounded in the real arms: the best offline selector (whiten) deployed worst
    env = robocasa_env()
    fitted = fit_fidelity([m for _, m, _ in ROBOCASA_OFFLINE_ONLINE],
                          [d for _, _, d in ROBOCASA_OFFLINE_ONLINE])
    assert env.feature_fidelity < 0
    assert env.feature_fidelity == pytest.approx(fitted)


def test_robocasa_env_has_the_documented_group_structure():
    env = robocasa_env()
    by_name = {r.name: r for r in env.regions}
    assert set(by_name) == set(GROUPS)
    # the unfixable-hard tail is (near-)uncollectable, matching pi0 0/14, GR00T 1/9
    assert by_name["unfixable_hard"].collectability < 0.2
    # the grasp-cluster is retention-toxic (the value arm's forgetting)
    assert by_name["retention_toxic"].ret_risk > 0
    # the non-targeted majority carries the most eval weight
    assert by_name["easy_majority"].weight == max(r.weight for r in env.regions)


def test_oracle_prefers_the_hard_but_collectable_group():
    env = robocasa_env()
    names = [r.name for r in env.regions]
    alloc = Oracle(step=2.0).allocate(env, demo_budget=200, measure_budget=0,
                                      rng=np.random.default_rng(0)).allocation
    picks = dict(zip(names, alloc))
    assert picks["hard_collectable"] == max(picks.values())
    assert picks["hard_collectable"] > picks["unfixable_hard"]      # collectability wall
    assert picks["hard_collectable"] > picks["retention_toxic"]     # retention wall


def test_dumping_budget_into_toxic_group_loses_vs_baseline():
    # reproduces the value/coverage arms losing through retention
    env = robocasa_env()
    names = [r.name for r in env.regions]
    baseline = env.net_success(np.zeros(env.n_regions))
    toxic = np.zeros(env.n_regions)
    toxic[names.index("retention_toxic")] = 200
    assert env.net_success(toxic) < baseline


def test_pfail_is_retention_safe_on_robocasa_env():
    # the P(fail) heuristic (core) does not regress the majority -> net >= baseline
    env = robocasa_env()
    baseline = env.net_success(np.zeros(env.n_regions))
    alloc = PFail().allocate(env, 200, 0, np.random.default_rng(0)).allocation
    assert env.net_success(alloc) >= baseline - 1e-9


def test_anchor_allocations_emits_two_named_picks():
    env = robocasa_env()
    out = anchor_allocations(env, demo_budget=200, measure_budget=20_000,
                             rng=np.random.default_rng(0))
    assert set(out) >= {"predictor_pick", "bandit_pick", "region_names"}
    assert len(out["region_names"]) == env.n_regions
    assert out["predictor_pick"].sum() <= 200 + 1e-6
    assert out["bandit_pick"].sum() <= 200 + 1e-6
