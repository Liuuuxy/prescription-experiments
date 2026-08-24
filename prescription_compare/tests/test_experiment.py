import numpy as np

from prescription_compare.experiment import Regime, run_sweep, summarize
from prescription_compare.env import PrescriptionEnv, Region
from prescription_compare.allocators.predictor import PPP
from prescription_compare.allocators.bandit_sh import SuccessiveHalvingBandit
from prescription_compare.allocators.oracle import Oracle


def _regimes():
    regions = [Region(base=0.6, headroom=0.5, tau=20.0),
               Region(base=0.2, headroom=0.05, tau=20.0)]
    return [
        Regime("predictable", PrescriptionEnv(regions, n_eval=200, feature_fidelity=1.0), demo_budget=100),
        Regime("unpredictable", PrescriptionEnv(regions, n_eval=200, feature_fidelity=0.0), demo_budget=100),
    ]


def _allocators():
    return {"predictor": PPP, "bandit": SuccessiveHalvingBandit, "oracle": Oracle}


def test_sweep_returns_a_record_per_combination():
    recs = run_sweep(_regimes(), _allocators(), measure_budgets=[400, 40_000], n_seeds=5)
    combos = {(r["regime"], r["allocator"], r["measure_budget"]) for r in recs}
    assert len(combos) == 2 * 3 * 2  # regimes x allocators x budgets


def test_sweep_reports_confidence_intervals():
    recs = run_sweep(_regimes(), _allocators(), measure_budgets=[40_000], n_seeds=8)
    for r in recs:
        assert r["ci_low"] <= r["mean_regret"] <= r["ci_high"]


def test_oracle_regret_is_zero_in_every_regime():
    recs = run_sweep(_regimes(), _allocators(), measure_budgets=[40_000], n_seeds=5)
    for r in recs:
        if r["allocator"] == "oracle":
            assert abs(r["mean_regret"]) < 1e-6


def test_sweep_reproduces_the_regime_dependent_winner():
    recs = run_sweep(_regimes(), {"predictor": PPP, "bandit": SuccessiveHalvingBandit},
                     measure_budgets=[400], n_seeds=20)
    pred_pred = next(r for r in recs if r["regime"] == "predictable" and r["allocator"] == "predictor")
    band_pred = next(r for r in recs if r["regime"] == "predictable" and r["allocator"] == "bandit")
    assert pred_pred["mean_regret"] < band_pred["mean_regret"]

    recs2 = run_sweep(_regimes(), {"predictor": PPP, "bandit": SuccessiveHalvingBandit},
                      measure_budgets=[40_000], n_seeds=20)
    pred_unp = next(r for r in recs2 if r["regime"] == "unpredictable" and r["allocator"] == "predictor")
    band_unp = next(r for r in recs2 if r["regime"] == "unpredictable" and r["allocator"] == "bandit")
    assert band_unp["mean_regret"] < pred_unp["mean_regret"]


def test_summarize_produces_text_with_regime_and_allocator_names():
    recs = run_sweep(_regimes(), _allocators(), measure_budgets=[40_000], n_seeds=3)
    text = summarize(recs)
    assert "predictable" in text and "predictor" in text and "bandit" in text
