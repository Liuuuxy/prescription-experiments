"""Fair evaluation of an allocator on a PrescriptionEnv.

Fairness rules, enforced here so they are verified not assumed:
  * the committed allocation is clipped to the demo budget (no over-collecting);
  * spending more than the measurement budget is a hard error (a fairness
    violation cannot be silently clipped -- the run is invalid);
  * the score is the NOISE-FREE deployed value of the committed allocation;
  * regret is measured against the Oracle at the same demo budget.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .allocators.base import clip_to_budget
from .allocators.oracle import Oracle


class BudgetError(Exception):
    """Raised when an allocator spends more measurement than its budget allows."""


@dataclass
class EvalResult:
    value: float
    regret: float
    measure_used: float
    allocation: np.ndarray


def evaluate(allocator, env, demo_budget, measure_budget, rng, oracle_value=None) -> EvalResult:
    res = allocator.allocate(env, demo_budget, measure_budget, rng)

    if res.measure_used > measure_budget + 1e-6:
        raise BudgetError(
            f"{allocator.name} spent {res.measure_used:.3f} measurement "
            f"> budget {measure_budget:.3f}"
        )

    alloc = clip_to_budget(res.allocation, demo_budget)
    value = env.net_success(alloc)

    if oracle_value is None:
        oracle_alloc = Oracle().allocate(env, demo_budget, 0, rng).allocation
        oracle_value = env.net_success(oracle_alloc)
    regret = oracle_value - value

    return EvalResult(value=value, regret=regret, measure_used=res.measure_used, allocation=alloc)
