import numpy as np
import pytest

from prescription_compare.env import PrescriptionEnv, Region
from prescription_compare.allocators.base import Allocator, AllocationResult
from prescription_compare.allocators.oracle import Oracle
from prescription_compare.evaluate import evaluate, BudgetError


class _Fixed(Allocator):
    """Test allocator that returns a preset allocation and measurement cost."""
    name = "fixed"

    def __init__(self, allocation, measure_used=0.0):
        self._a = np.asarray(allocation, float)
        self._m = measure_used

    def allocate(self, env, demo_budget, measure_budget, rng):
        return AllocationResult(allocation=self._a.copy(), measure_used=self._m)


def _env():
    return PrescriptionEnv([
        Region(base=0.2, headroom=0.5, tau=20.0),
        Region(base=0.4, headroom=0.3, tau=20.0),
    ])


def test_evaluate_returns_noise_free_deployed_value():
    env = _env()
    alloc = [20.0, 0.0]
    res = evaluate(_Fixed(alloc), env, demo_budget=100, measure_budget=100,
                   rng=np.random.default_rng(0))
    assert res.value == pytest.approx(env.net_success(alloc))


def test_evaluate_clips_over_budget_allocation_before_scoring():
    env = _env()
    # requests 1000 total but budget is 100 -> scored at the clipped (sum=100) allocation
    res = evaluate(_Fixed([750.0, 250.0]), env, demo_budget=100, measure_budget=100,
                   rng=np.random.default_rng(0))
    assert res.allocation.sum() == pytest.approx(100.0)
    assert res.value == pytest.approx(env.net_success([75.0, 25.0]))


def test_evaluate_raises_when_measurement_over_budget():
    env = _env()
    with pytest.raises(BudgetError):
        evaluate(_Fixed([10.0, 10.0], measure_used=501.0), env,
                 demo_budget=100, measure_budget=500, rng=np.random.default_rng(0))


def test_evaluate_passes_measurement_cost_through():
    env = _env()
    res = evaluate(_Fixed([10.0, 10.0], measure_used=123.0), env,
                   demo_budget=100, measure_budget=500, rng=np.random.default_rng(0))
    assert res.measure_used == 123.0


def test_oracle_has_zero_regret():
    env = _env()
    res = evaluate(Oracle(step=1.0), env, demo_budget=100, measure_budget=0,
                   rng=np.random.default_rng(0))
    assert res.regret == pytest.approx(0.0, abs=1e-9)


def test_regret_is_nonnegative_for_a_suboptimal_allocation():
    env = _env()
    res = evaluate(_Fixed([0.0, 100.0]), env, demo_budget=100, measure_budget=0,
                   rng=np.random.default_rng(0))
    assert res.regret >= -1e-9
