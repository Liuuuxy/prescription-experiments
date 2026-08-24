"""Plots: mean regret vs measurement budget, one panel per regime, one line per allocator."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_regret_vs_budget(records, path: str) -> str:
    regimes = sorted({r["regime"] for r in records})
    fig, axes = plt.subplots(1, len(regimes), figsize=(4.0 * len(regimes), 3.3), squeeze=False)
    for ax, regime in zip(axes[0], regimes):
        allocs = sorted({r["allocator"] for r in records if r["regime"] == regime})
        for a in allocs:
            rows = sorted(
                (r for r in records if r["regime"] == regime and r["allocator"] == a),
                key=lambda x: x["measure_budget"],
            )
            xs = [r["measure_budget"] for r in rows]
            ys = [r["mean_regret"] for r in rows]
            lo = [r["mean_regret"] - r["ci_low"] for r in rows]  # half-widths for error bars
            ax.errorbar(xs, ys, yerr=lo, marker="o", capsize=2, label=a)
        ax.set_title(regime)
        ax.set_xscale("log")
        ax.set_xlabel("measurement budget (rollouts)")
        ax.set_ylabel("mean regret vs oracle")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
