"""Regime sweep: run every allocator across regimes x measurement budgets x seeds,
and aggregate mean regret with confidence intervals. This is the deliverable output --
a map of *when* predict beats explore, not a single winner.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import PrescriptionEnv, Region
from .evaluate import evaluate
from .allocators.oracle import Oracle


@dataclass
class Regime:
    name: str
    env: object
    demo_budget: float


def _template_regions():
    """A spread of regions: value-dense/low-failure, failure-dense/low-value, and a
    retention-toxic one -- so P(fail) mis-ranks and retention can bite."""
    return [
        Region(0.60, 0.50, 30.0, name="value_dense"),
        Region(0.20, 0.05, 20.0, name="failure_dense"),
        Region(0.45, 0.30, 30.0, name="mid"),
        Region(0.30, 0.02, 20.0, ret_risk=0.001, name="toxic"),
        Region(0.55, 0.20, 30.0, name="good2"),
    ]


def standard_regimes():
    """The canonical regime map: synthetic fidelity sweep + the RoboCasa twin."""
    from .calibrate import robocasa_env
    return [
        Regime("predictable", PrescriptionEnv(_template_regions(), n_eval=200, feature_fidelity=1.0), 120),
        Regime("partial", PrescriptionEnv(_template_regions(), n_eval=200, feature_fidelity=0.5), 120),
        Regime("unpredictable", PrescriptionEnv(_template_regions(), n_eval=200, feature_fidelity=0.0), 120),
        Regime("robocasa", robocasa_env(), 200),
    ]


def run_sweep(regimes, allocators, measure_budgets, n_seeds: int, base_seed: int = 0):
    """allocators: {name: factory()}; returns a list of aggregated records."""
    records = []
    for regime in regimes:
        oracle_alloc = Oracle(step=1.0).allocate(
            regime.env, regime.demo_budget, 0, np.random.default_rng(base_seed)).allocation
        oracle_value = regime.env.net_success(oracle_alloc)

        for mb in measure_budgets:
            for name, factory in allocators.items():
                regrets, values, measures = [], [], []
                for s in range(n_seeds):
                    rng = np.random.default_rng(base_seed + 1 + s)
                    res = evaluate(factory(), regime.env, regime.demo_budget, mb, rng,
                                   oracle_value=oracle_value)
                    regrets.append(res.regret)
                    values.append(res.value)
                    measures.append(res.measure_used)
                regrets = np.array(regrets)
                mean = float(regrets.mean())
                half = 1.96 * float(regrets.std(ddof=1)) / np.sqrt(n_seeds) if n_seeds > 1 else 0.0
                records.append({
                    "regime": regime.name,
                    "allocator": name,
                    "measure_budget": mb,
                    "mean_regret": mean,
                    "ci_low": mean - half,
                    "ci_high": mean + half,
                    "mean_value": float(np.mean(values)),
                    "mean_measure_used": float(np.mean(measures)),
                    "n_seeds": n_seeds,
                })
    return records


def summarize(records) -> str:
    """Render records as a fixed-width text table, most-to-least budget."""
    header = f"{'regime':<16}{'allocator':<12}{'M-budget':>10}{'mean_regret':>13}{'95% CI':>20}"
    lines = [header, "-" * len(header)]
    for r in sorted(records, key=lambda x: (x["regime"], x["measure_budget"], x["allocator"])):
        ci = f"[{r['ci_low']:+.4f},{r['ci_high']:+.4f}]"
        lines.append(
            f"{r['regime']:<16}{r['allocator']:<12}{r['measure_budget']:>10.0f}"
            f"{r['mean_regret']:>13.4f}{ci:>20}"
        )
    return "\n".join(lines)
