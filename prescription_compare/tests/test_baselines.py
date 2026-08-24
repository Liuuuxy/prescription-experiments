import numpy as np
import pytest

from prescription_compare.env import PrescriptionEnv, Region
from prescription_compare.allocators.random_alloc import RandomAlloc
from prescription_compare.allocators.pfail import PFail


def _env(bases):
    return PrescriptionEnv([Region(base=b, headroom=0.4, tau=20.0) for b in bases])


def test_random_spends_full_budget_without_measurement():
    env = _env([0.2, 0.4, 0.5])
    res = RandomAlloc().allocate(env, demo_budget=90, measure_budget=1e9,
                                 rng=np.random.default_rng(0))
    assert res.allocation.sum() == pytest.approx(90.0)
    assert res.measure_used == 0.0
    assert (res.allocation >= 0).all()


def test_random_varies_with_seed():
    env = _env([0.2, 0.4, 0.5])
    a0 = RandomAlloc().allocate(env, 90, 1e9, np.random.default_rng(0)).allocation
    a1 = RandomAlloc().allocate(env, 90, 1e9, np.random.default_rng(1)).allocation
    assert not np.allclose(a0, a1)


def test_pfail_puts_more_budget_on_higher_failure_region():
    env = _env([0.1, 0.8])  # region 0 fails 90%, region 1 fails 20%
    res = PFail().allocate(env, demo_budget=100, measure_budget=0,
                           rng=np.random.default_rng(0))
    assert res.allocation[0] > res.allocation[1]
    assert res.allocation.sum() == pytest.approx(100.0)
    assert res.measure_used == 0.0  # weak-region diagnosis is a shared, free prior


def test_pfail_allocates_equally_when_failure_is_equal():
    env = _env([0.3, 0.3])
    res = PFail().allocate(env, 100, 0, np.random.default_rng(0))
    assert res.allocation[0] == pytest.approx(res.allocation[1])
