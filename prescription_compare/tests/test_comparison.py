"""The headline mechanism, locked as a test: which of predict vs explore wins is a
function of feature fidelity and measurement budget -- not a fixed verdict."""
import numpy as np

from prescription_compare.env import PrescriptionEnv, Region
from prescription_compare.allocators.predictor import PPP
from prescription_compare.allocators.bandit_sh import SuccessiveHalvingBandit
from prescription_compare.allocators.oracle import Oracle
from prescription_compare.evaluate import evaluate


def _misleading_env(fidelity):
    return PrescriptionEnv(
        [
            Region(base=0.6, headroom=0.5, tau=20.0),   # value-dense, low failure
            Region(base=0.2, headroom=0.05, tau=20.0),  # failure-dense, low value
        ],
        n_eval=200, feature_fidelity=fidelity, probe_noise0=0.2,
    )


def _mean_regret(alloc_factory, env, demo_budget, measure_budget, seeds):
    oracle_alloc = Oracle(step=1.0).allocate(env, demo_budget, 0, np.random.default_rng(0)).allocation
    oracle_value = env.net_success(oracle_alloc)
    regrets = []
    for s in seeds:
        rng = np.random.default_rng(s)
        res = evaluate(alloc_factory(), env, demo_budget, measure_budget, rng,
                       oracle_value=oracle_value)
        regrets.append(res.regret)
    return float(np.mean(regrets))


def test_predict_wins_when_features_good_and_measurement_scarce():
    env = _misleading_env(fidelity=1.0)
    seeds = range(20)
    pred = _mean_regret(PPP, env, demo_budget=100, measure_budget=400, seeds=seeds)
    band = _mean_regret(SuccessiveHalvingBandit, env, demo_budget=100, measure_budget=400, seeds=seeds)
    assert pred < band


def test_explore_wins_when_features_useless_and_measurement_ample():
    env = _misleading_env(fidelity=0.0)
    seeds = range(20)
    pred = _mean_regret(PPP, env, demo_budget=100, measure_budget=40_000, seeds=seeds)
    band = _mean_regret(SuccessiveHalvingBandit, env, demo_budget=100, measure_budget=40_000, seeds=seeds)
    assert band < pred
