"""Successive-halving best-arm identification over regions.

Arms = regions. A pull evaluates the noisy net-success of putting a representative
chunk into one region (cost = n_eval rollouts). Pulls see the TRUE net-success
(the proxy fidelity cannot fool them), so the bandit trades measurement cost for an
unbiased estimate -- the opposite bias/cost tradeoff from the predictor. Successive
halving concentrates pulls on promising regions, then the same greedy rule allocates.
"""
from __future__ import annotations

import numpy as np

from .base import Allocator, AllocationResult, proportional_positive


class SuccessiveHalvingBandit(Allocator):
    name = "bandit"

    def allocate(self, env, demo_budget, measure_budget, rng) -> AllocationResult:
        n = env.n_regions
        chunk = demo_budget / n if n else 0.0
        pull_cost = float(env.n_eval)
        zero = np.zeros(n)
        measure_used = 0.0

        if n == 0 or measure_budget < pull_cost:
            return AllocationResult(allocation=np.zeros(n), measure_used=0.0)

        # a stabilized shared baseline (a few averaged no-op evals)
        base_pulls = max(1, int(0.1 * measure_budget / pull_cost))
        base_obs = []
        for _ in range(base_pulls):
            if measure_budget - measure_used < pull_cost:
                break
            base_obs.append(env.net_success(zero, rng))
            measure_used += pull_cost
        baseline = float(np.mean(base_obs))

        obs = {r: [] for r in range(n)}
        survivors = list(range(n))
        rounds = max(1, int(np.ceil(np.log2(n)))) if n > 1 else 1
        per_round = (measure_budget - measure_used) / rounds

        while len(survivors) > 1 and (measure_budget - measure_used) >= pull_cost:
            pulls_each = max(1, int((per_round / pull_cost) / len(survivors)))
            for r in list(survivors):
                a = zero.copy()
                a[r] = chunk
                for _ in range(pulls_each):
                    if measure_budget - measure_used < pull_cost:
                        break
                    obs[r].append(env.net_success(a, rng) - baseline)
                    measure_used += pull_cost
            means = {r: (np.mean(obs[r]) if obs[r] else -np.inf) for r in survivors}
            survivors = sorted(survivors, key=lambda r: means[r], reverse=True)
            survivors = survivors[: max(1, len(survivors) // 2)]

        mu = np.array([np.mean(obs[r]) if obs[r] else 0.0 for r in range(n)])
        alloc = proportional_positive(mu, demo_budget)
        return AllocationResult(allocation=alloc, measure_used=measure_used)
