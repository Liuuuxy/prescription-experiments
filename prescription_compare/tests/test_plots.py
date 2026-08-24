import os

from prescription_compare.experiment import standard_regimes, run_sweep
from prescription_compare.allocators.predictor import PPP
from prescription_compare.allocators.bandit_sh import SuccessiveHalvingBandit
from prescription_compare.allocators.oracle import Oracle
from prescription_compare.plots import plot_regret_vs_budget


def test_plot_writes_a_nonempty_file(tmp_path):
    recs = run_sweep(standard_regimes()[:2],
                     {"predictor": PPP, "bandit": SuccessiveHalvingBandit, "oracle": Oracle},
                     measure_budgets=[400, 4000], n_seeds=3)
    out = tmp_path / "regret.png"
    path = plot_regret_vs_budget(recs, str(out))
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0
