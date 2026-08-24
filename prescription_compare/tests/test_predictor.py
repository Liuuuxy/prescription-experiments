import numpy as np
import pytest

from prescription_compare.env import PrescriptionEnv, Region
from prescription_compare.allocators.predictor import PPP
from prescription_compare.allocators.pfail import PFail
from prescription_compare.allocators.oracle import Oracle


def _misleading_env(fidelity):
    # region 0: LOW P(fail) but HIGH value; region 1: HIGH P(fail) but LOW value.
    # P(fail) mis-ranks; good features should not.
    return PrescriptionEnv(
        [
            Region(base=0.6, headroom=0.5, tau=20.0),   # value-dense, low failure
            Region(base=0.2, headroom=0.05, tau=20.0),  # failure-dense, low value
        ],
        feature_fidelity=fidelity,
        probe_noise0=0.2,
    )


def test_predictor_charges_measurement_for_probes():
    env = _misleading_env(1.0)
    res = PPP(probe_frac=1.0).allocate(env, demo_budget=100, measure_budget=500,
                                       rng=np.random.default_rng(0))
    assert res.measure_used == pytest.approx(500.0)
    assert res.measure_used <= 500.0 + 1e-6


def test_predictor_with_good_features_beats_pfail():
    env = _misleading_env(fidelity=1.0)
    rng = np.random.default_rng(0)
    ppp = PPP().allocate(env, 100, measure_budget=100_000, rng=rng).allocation
    pf = PFail().allocate(env, 100, 0, rng).allocation
    # good features send budget to the value-dense region 0; P(fail) sends it to region 1
    assert ppp[0] > ppp[1]
    assert pf[1] > pf[0]
    assert env.net_success(ppp) > env.net_success(pf)


def test_predictor_with_useless_features_degenerates_to_pfail():
    # anti-circularity: at zero fidelity the proxy IS P(fail); the predictor must NOT
    # magically beat the heuristic.
    env = _misleading_env(fidelity=0.0)
    rng = np.random.default_rng(0)
    ppp = PPP().allocate(env, 100, measure_budget=100_000, rng=rng).allocation
    assert ppp[1] > ppp[0]  # follows P(fail), same wrong ranking


def test_predictor_refuses_to_spend_when_all_value_predicted_negative():
    # every region retention-toxic (true marginal < 0); good features -> allocate nothing
    env = PrescriptionEnv(
        [Region(base=0.5, headroom=0.05, tau=10.0, ret_risk=0.05) for _ in range(3)],
        feature_fidelity=1.0, probe_noise0=0.01,
    )
    alloc = PPP().allocate(env, 100, measure_budget=100_000,
                           rng=np.random.default_rng(0)).allocation
    assert alloc.sum() == pytest.approx(0.0)
