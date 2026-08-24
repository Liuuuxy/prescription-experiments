import numpy as np
import pytest

from prescription_compare.allocators.base import Allocator, AllocationResult, clip_to_budget


def test_allocation_result_holds_allocation_and_measurement_cost():
    res = AllocationResult(allocation=np.array([10.0, 5.0]), measure_used=42.0)
    assert res.allocation.tolist() == [10.0, 5.0]
    assert res.measure_used == 42.0


def test_clip_to_budget_scales_down_over_budget_allocation():
    # total 200 requested but budget 100 -> scaled to sum 100, proportions kept
    clipped = clip_to_budget(np.array([150.0, 50.0]), demo_budget=100.0)
    assert clipped.sum() == pytest.approx(100.0)
    assert clipped[0] / clipped[1] == pytest.approx(3.0)


def test_clip_to_budget_leaves_under_budget_allocation_unchanged():
    a = np.array([30.0, 20.0])
    assert clip_to_budget(a, demo_budget=100.0).tolist() == [30.0, 20.0]


def test_base_allocator_is_abstract():
    with pytest.raises(NotImplementedError):
        Allocator().allocate(env=None, demo_budget=100, measure_budget=100,
                             rng=np.random.default_rng(0))
