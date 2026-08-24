"""P(fail) heuristic: concentrate the budget on the top-fraction highest-failure regions.

This is the real "core" arm -- top-K depth on the worst regions, NOT a proportional
spread across all of them. Concentrating on the top failures is exactly what keeps it
retention-safe (it skips the mid-failure, retention-toxic regions the value/coverage
arms drifted into). Weak-region diagnosis is a shared prior, so it costs no measurement.
"""
from __future__ import annotations

import math

import numpy as np

from .base import Allocator, AllocationResult


class PFail(Allocator):
    name = "pfail"

    def __init__(self, top_frac: float = 0.5):
        self.top_frac = float(top_frac)

    def allocate(self, env, demo_budget, measure_budget, rng) -> AllocationResult:
        pfail = 1.0 - np.array([r.base for r in env.regions])
        n = len(pfail)
        k = max(1, math.ceil(self.top_frac * n))
        threshold = np.sort(pfail)[::-1][k - 1]      # k-th largest
        selected = pfail >= threshold - 1e-12        # include ties at the cutoff

        weights = np.where(selected, pfail, 0.0)
        total = weights.sum()
        if total == 0:                               # all-equal, zero-failure edge case
            weights = selected.astype(float)
            total = weights.sum()
        alloc = weights / total * demo_budget
        return AllocationResult(allocation=alloc, measure_used=0.0)
