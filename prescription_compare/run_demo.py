"""Run the full predict-vs-explore comparison: the regime sweep + the RoboCasa anchor
picks. Usage: PYTHONPATH=/data/xinyua11/robocasa python -m prescription_compare.run_demo
"""
from __future__ import annotations

import numpy as np

from .experiment import standard_regimes, run_sweep, summarize
from .allocators.predictor import PPP
from .allocators.bandit_sh import SuccessiveHalvingBandit
from .allocators.oracle import Oracle
from .allocators.random_alloc import RandomAlloc
from .allocators.pfail import PFail
from .calibrate import robocasa_env, anchor_allocations
from .plots import plot_regret_vs_budget

ALLOCATORS = {
    "random": RandomAlloc,
    "pfail": PFail,
    "predictor": PPP,
    "bandit": SuccessiveHalvingBandit,
    "oracle": Oracle,
}
BUDGETS = [200, 1000, 5000, 20_000, 80_000]


def main():
    records = run_sweep(standard_regimes(), ALLOCATORS, BUDGETS, n_seeds=40)
    print(summarize(records))

    plot_path = plot_regret_vs_budget(records, "weakregion/prescription_compare_regret.png")
    print(f"\nsaved regret plot -> {plot_path}")

    env = robocasa_env()
    out = anchor_allocations(env, demo_budget=200, measure_budget=20_000,
                             rng=np.random.default_rng(0))
    print("\nRoboCasa-twin anchor allocations (demos per region group):")
    for name, p, b in zip(out["region_names"], out["predictor_pick"], out["bandit_pick"]):
        print(f"  {name:<18} predictor={p:6.1f}   bandit={b:6.1f}")


if __name__ == "__main__":
    main()
