"""Random baseline: spend the whole budget on a random spread across regions, blind."""
from __future__ import annotations

import numpy as np

from .base import Allocator, AllocationResult


class RandomAlloc(Allocator):
    name = "random"

    def allocate(self, env, demo_budget, measure_budget, rng) -> AllocationResult:
        w = rng.dirichlet(np.ones(env.n_regions))
        return AllocationResult(allocation=w * demo_budget, measure_used=0.0)
