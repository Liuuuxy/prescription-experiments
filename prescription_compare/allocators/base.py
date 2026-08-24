"""Allocator interface and the shared budget helper.

Every allocator returns an AllocationResult: how many demos to request per region
(the deliverable) and how much measurement it spent deciding (charged to the fair
budget). Fairness enforcement lives in evaluate.py; this module only defines the
contract and a budget-clipping helper.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AllocationResult:
    allocation: np.ndarray   # requested demos per region (the committed policy)
    measure_used: float      # measurement rollouts spent deciding (charged to budget)


def clip_to_budget(allocation, demo_budget: float) -> np.ndarray:
    """Scale an allocation down proportionally if it exceeds the demo budget."""
    a = np.asarray(allocation, dtype=float)
    total = a.sum()
    if total <= demo_budget or total == 0:
        return a
    return a * (demo_budget / total)


def greedy_allocate(marginal_fn, n_regions: int, demo_budget: float, step: float) -> np.ndarray:
    """Fill the budget one `step` at a time into the region with the highest current
    marginal gain, stopping once the best marginal is non-positive.

    Optimal for separable, concave-per-region objectives (which PrescriptionEnv is:
    concave saturating gains minus a linear retention cost => monotone-decreasing
    marginals). Stopping early is correct -- over-collecting into a retention-toxic
    region only hurts (the inverted-U).
    """
    alloc = np.zeros(n_regions)
    spent = 0.0
    while spent + step <= demo_budget + 1e-9:
        gains = np.array([marginal_fn(r, alloc) for r in range(n_regions)])
        best = int(np.argmax(gains))
        if gains[best] <= 0:
            break
        alloc[best] += step
        spent += step
    return alloc


def proportional_positive(values, demo_budget: float) -> np.ndarray:
    """Allocate the demo budget proportional to positive estimated value.

    If every region is predicted non-positive, allocate nothing -- refusing to
    spend into predicted-harmful regions (the retention-aware, inverted-U-safe move).
    Shared by predictor and bandit so only their *estimates* differ, not the rule.
    """
    p = np.clip(np.asarray(values, dtype=float), 0.0, None)
    s = p.sum()
    if s == 0:
        return np.zeros_like(p)
    return p / s * demo_budget


class Allocator:
    name = "base"

    def allocate(self, env, demo_budget, measure_budget, rng) -> AllocationResult:
        raise NotImplementedError
