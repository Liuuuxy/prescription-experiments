import numpy as np
import pytest

from prescription_compare.env import PrescriptionEnv, Region
from prescription_compare.allocators.bandit_sh import SuccessiveHalvingBandit
from prescription_compare.allocators.random_alloc import RandomAlloc


def test_bandit_respects_measurement_budget():
    env = PrescriptionEnv([Region(base=b, headroom=0.5, tau=20.0) for b in (0.2, 0.4, 0.6)],
                          n_eval=280)
    res = SuccessiveHalvingBandit().allocate(env, demo_budget=120, measure_budget=10_000,
                                             rng=np.random.default_rng(0))
    assert res.measure_used <= 10_000 + 1e-6
    assert res.measure_used > 0


def test_bandit_finds_the_high_value_region_in_a_clean_env():
    # region 2 has by far the most headroom -> the bandit should send it the most budget
    env = PrescriptionEnv([
        Region(base=0.4, headroom=0.05, tau=20.0),
        Region(base=0.4, headroom=0.10, tau=20.0),
        Region(base=0.4, headroom=0.60, tau=20.0),
    ], n_eval=200)
    alloc = SuccessiveHalvingBandit().allocate(
        env, 150, measure_budget=60_000, rng=np.random.default_rng(0)).allocation
    assert int(np.argmax(alloc)) == 2


def test_bandit_beats_random_given_enough_measurement():
    env = PrescriptionEnv([
        Region(base=0.3, headroom=0.05, tau=20.0),
        Region(base=0.3, headroom=0.55, tau=20.0),
    ], n_eval=200)
    rng = np.random.default_rng(0)
    b_vals, r_vals = [], []
    for _ in range(10):
        b = SuccessiveHalvingBandit().allocate(env, 100, 40_000, rng).allocation
        r = RandomAlloc().allocate(env, 100, 0, rng).allocation
        b_vals.append(env.net_success(b))
        r_vals.append(env.net_success(r))
    assert np.mean(b_vals) > np.mean(r_vals)
