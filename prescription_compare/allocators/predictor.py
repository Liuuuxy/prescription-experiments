"""PPP: the predictor. Estimate each region's value from cheap (biased) proxy probes,
then allocate greedily. Wins when the proxy tracks value; degenerates to the P(fail)
decoy when it does not -- and pays for the probes it takes.
"""
from __future__ import annotations

import numpy as np

from .base import Allocator, AllocationResult, proportional_positive


class PPP(Allocator):
    name = "predictor"

    def __init__(self, probe_frac: float = 1.0):
        self.probe_frac = float(probe_frac)

    def allocate(self, env, demo_budget, measure_budget, rng) -> AllocationResult:
        n = env.n_regions
        chunk = demo_budget / n if n else 0.0
        total_probe = self.probe_frac * measure_budget
        cost_each = total_probe / n if n else 0.0

        mu = np.array([
            env.probe(r, chunk, max(cost_each, 1e-9), rng) for r in range(n)
        ])
        alloc = proportional_positive(mu, demo_budget)
        return AllocationResult(allocation=alloc, measure_used=cost_each * n)
