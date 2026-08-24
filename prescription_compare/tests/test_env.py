import math

import numpy as np
import pytest

from prescription_compare.env import PrescriptionEnv, Region


def test_zero_allocation_returns_weighted_base():
    env = PrescriptionEnv([Region(base=0.3, headroom=0.4, tau=50.0)])
    assert env.net_success([0]) == pytest.approx(0.3)


def test_single_region_saturating_curve_at_tau():
    # allocating tau usable demos reaches (1 - 1/e) of headroom
    env = PrescriptionEnv([Region(base=0.3, headroom=0.4, tau=50.0)])
    expected = 0.3 + 0.4 * (1 - math.exp(-1))
    assert env.net_success([50]) == pytest.approx(expected)


def test_headroom_saturates_at_base_plus_headroom():
    env = PrescriptionEnv([Region(base=0.3, headroom=0.4, tau=50.0)])
    assert env.net_success([10_000_000]) == pytest.approx(0.7, abs=1e-6)


def test_weighted_mean_over_two_regions():
    env = PrescriptionEnv([
        Region(base=0.2, headroom=0.5, tau=10.0, weight=3.0),
        Region(base=0.2, headroom=0.5, tau=10.0, weight=1.0),
    ])
    s0 = 0.2 + 0.5 * (1 - math.exp(-1))   # region 0 got tau=10 demos
    gross = (3.0 * s0 + 1.0 * 0.2) / 4.0
    assert env.net_success([10, 0]) == pytest.approx(gross)


def test_collectability_reduces_usable_demos():
    # 100 requested * 0.5 collectability = 50 usable = tau -> (1-1/e) of headroom
    env = PrescriptionEnv([Region(base=0.3, headroom=0.4, tau=50.0, collectability=0.5)])
    expected = 0.3 + 0.4 * (1 - math.exp(-1))
    assert env.net_success([100]) == pytest.approx(expected)


def test_retention_penalty_subtracts_from_net():
    env = PrescriptionEnv([Region(base=0.5, headroom=0.2, tau=10.0, ret_risk=0.01)])
    gain = 0.5 + 0.2 * (1 - math.exp(-1))
    net = gain - 0.01 * 10  # 10 usable demos * 0.01 ret_risk
    assert env.net_success([10]) == pytest.approx(net)


def test_net_success_clipped_to_unit_interval():
    # huge retention risk cannot drive net below 0
    env = PrescriptionEnv([Region(base=0.5, headroom=0.2, tau=10.0, ret_risk=1.0)])
    assert env.net_success([100]) == pytest.approx(0.0)


def test_usable_deterministic_is_requested_times_collectability():
    env = PrescriptionEnv([Region(base=0.3, headroom=0.4, tau=50.0, collectability=0.4)])
    assert env.usable([100]) == pytest.approx([40.0])


def test_stochastic_eval_noise_is_unbiased():
    # with an rng, net_success returns Binomial(n_eval, p)/n_eval; mean over draws ~ p
    env = PrescriptionEnv([Region(base=0.4, headroom=0.0, tau=1.0)], n_eval=1000)
    rng = np.random.default_rng(0)
    draws = np.array([env.net_success([0], rng=rng) for _ in range(3000)])
    assert draws.mean() == pytest.approx(0.4, abs=0.005)
    assert draws.std() > 0  # genuinely noisy


def test_stochastic_collectability_is_binomial():
    # with an rng, usable demos are Binomial(k, collectability); mean ~ k*c
    env = PrescriptionEnv([Region(base=0.3, headroom=0.4, tau=50.0, collectability=0.5)])
    rng = np.random.default_rng(1)
    usable = np.array([env.usable([100], rng=rng)[0] for _ in range(2000)])
    assert usable.mean() == pytest.approx(50.0, abs=1.0)
    assert set(np.unique(usable)) != {50.0}  # not the deterministic value


def test_marginal_value_is_noise_free_delta():
    env = PrescriptionEnv([Region(base=0.3, headroom=0.4, tau=50.0)])
    expected = 0.4 * (1 - math.exp(-1))  # net(50) - net(0)
    assert env.marginal_value(0, 50) == pytest.approx(expected)


def test_probe_full_fidelity_tracks_true_marginal_value():
    # fidelity=1: probe is an unbiased estimate of the true marginal value; decoy ignored
    env = PrescriptionEnv([Region(base=0.3, headroom=0.4, tau=50.0)],
                          feature_fidelity=1.0, decoy=[999.0])
    rng = np.random.default_rng(0)
    est = np.mean([env.probe(0, 50, cost=100.0, rng=rng) for _ in range(3000)])
    assert est == pytest.approx(env.marginal_value(0, 50), abs=0.01)


def test_probe_zero_fidelity_tracks_the_decoy_not_the_truth():
    # fidelity=0: probe tracks the (mis-ranking) decoy, e.g. P(fail) -- the real-world failure mode
    env = PrescriptionEnv([Region(base=0.3, headroom=0.4, tau=50.0)],
                          feature_fidelity=0.0, decoy=[0.123])
    rng = np.random.default_rng(0)
    est = np.mean([env.probe(0, 50, cost=100.0, rng=rng) for _ in range(3000)])
    assert est == pytest.approx(0.123, abs=0.01)


def test_probe_negative_fidelity_anti_tracks_true_value():
    # fidelity=-1: the proxy tracks the NEGATIVE of true value (actively misleading),
    # the real offline<->online anti-correlation. decoy is non-zero to distinguish
    # (1-|fid|) weighting from (1-fid).
    env = PrescriptionEnv([Region(base=0.3, headroom=0.4, tau=50.0)],
                          feature_fidelity=-1.0, decoy=[0.5])
    rng = np.random.default_rng(0)
    est = np.mean([env.probe(0, 50, cost=100.0, rng=rng) for _ in range(3000)])
    assert est == pytest.approx(-env.marginal_value(0, 50), abs=0.01)


def test_probe_noise_shrinks_with_cost():
    env = PrescriptionEnv([Region(base=0.3, headroom=0.4, tau=50.0)],
                          feature_fidelity=1.0, decoy=[0.0])
    rng = np.random.default_rng(0)
    cheap = np.std([env.probe(0, 50, cost=1.0, rng=rng) for _ in range(2000)])
    dear = np.std([env.probe(0, 50, cost=100.0, rng=rng) for _ in range(2000)])
    assert dear < cheap
