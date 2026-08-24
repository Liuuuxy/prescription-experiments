"""Oracle allocator: knows the hidden truth, spends no measurement, upper-bounds regret."""
from __future__ import annotations

import numpy as np

from .base import Allocator, AllocationResult, greedy_allocate


class Oracle(Allocator):
    name = "oracle"

    def __init__(self, step=None):
        self.step = step

    def allocate(self, env, demo_budget, measure_budget, rng) -> AllocationResult:
        step = self.step or max(1.0, demo_budget / 100.0)

        def marginal(r, alloc):
            before = env.net_success(alloc)
            a2 = alloc.copy()
            a2[r] += step
            return env.net_success(a2) - before

        alloc = greedy_allocate(marginal, env.n_regions, demo_budget, step)
        return AllocationResult(allocation=alloc, measure_used=0.0)
