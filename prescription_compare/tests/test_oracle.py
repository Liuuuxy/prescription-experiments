import numpy as np
import pytest

from prescription_compare.env import PrescriptionEnv, Region
from prescription_compare.allocators.oracle import Oracle


def _random_alloc(rng, n_regions, budget):
    w = rng.dirichlet(np.ones(n_regions))
    return w * budget


def test_oracle_never_exceeds_budget():
    env = PrescriptionEnv([
        Region(base=0.2, headroom=0.5, tau=20.0),
        Region(base=0.4, headroom=0.3, tau=20.0),
    ])
    res = Oracle().allocate(env, demo_budget=100, measure_budget=0, rng=np.random.default_rng(0))
    assert res.allocation.sum() <= 100 + 1e-6


def test_oracle_spends_no_measurement():
    env = PrescriptionEnv([Region(base=0.2, headroom=0.5, tau=20.0)])
    res = Oracle().allocate(env, demo_budget=100, measure_budget=1e9, rng=np.random.default_rng(0))
    assert res.measure_used == 0.0


def test_oracle_beats_every_random_allocation():
    env = PrescriptionEnv([
        Region(base=0.2, headroom=0.5, tau=20.0, ret_risk=0.0),
        Region(base=0.4, headroom=0.3, tau=40.0, ret_risk=0.002),
        Region(base=0.1, headroom=0.6, tau=15.0, ret_risk=0.0),
    ])
    rng = np.random.default_rng(0)
    oracle_alloc = Oracle(step=1.0).allocate(env, 120, 0, rng).allocation
    oracle_val = env.net_success(oracle_alloc)
    rand_vals = [env.net_success(_random_alloc(rng, 3, 120)) for _ in range(300)]
    assert oracle_val >= max(rand_vals) - 1e-9


def test_oracle_avoids_high_retention_region():
    env = PrescriptionEnv([
        Region(base=0.2, headroom=0.5, tau=20.0, ret_risk=0.0),     # good
        Region(base=0.2, headroom=0.1, tau=20.0, ret_risk=0.05),    # retention-toxic
    ])
    alloc = Oracle(step=1.0).allocate(env, 100, 0, np.random.default_rng(0)).allocation
    assert alloc[0] > alloc[1]
    assert alloc[1] == pytest.approx(0.0)


def test_oracle_under_spends_when_all_marginals_negative():
    # a region whose retention cost exceeds its value from the first demo -> collect nothing
    env = PrescriptionEnv([Region(base=0.5, headroom=0.1, tau=10.0, ret_risk=0.05)])
    alloc = Oracle(step=1.0).allocate(env, 100, 0, np.random.default_rng(0)).allocation
    assert alloc.sum() == pytest.approx(0.0)
