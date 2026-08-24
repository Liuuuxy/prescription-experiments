"""DataDecide-style decision-accuracy diagnostic on the bandit_v1 ledger.

Borrowed instrument: Magnusson et al., DataDecide (arXiv 2504.11393, ICML 2025).
They show pairwise "decision accuracy" (fraction of arm pairs whose winner the
proxy gets right) is driven by SPREAD (sd of arm means) vs NOISE (sd across
seeds). Predictable settings have spread >> noise. This script computes both on
our own pulls and simulates how many pulls/arm we would need.

Read-only w.r.t. the ledger. Writes decision_accuracy_diagnostic.json here.
Run: /data/xinyua11/conda/envs/robocasa/bin/python <this file>
"""
import itertools, json, os, sys

import numpy as np

sys.path.insert(0, "/data/xinyua11/robocasa")
from bandit_v1 import ledger  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GPU_H_PER_PULL = 7.0


def main():
    ok = ledger.read("pulls")
    ok = ok[ok.status == "ok"]
    sel = sorted(a for a in ok.arm.unique() if a != "null")
    mu = ok[ok.arm.isin(sel)].groupby("arm")["delta"].mean()

    ss, df = 0.0, 0
    for _, d in ok.groupby("arm")["delta"]:
        x = d.values
        if len(x) > 1:
            ss += ((x - x.mean()) ** 2).sum()
            df += len(x) - 1
    noise = float(np.sqrt(ss / df))
    spread = float(mu.std(ddof=1))

    rng = np.random.default_rng(0)
    truth = mu.values
    pairs = list(itertools.combinations(range(len(truth)), 2))
    curve = {}
    for n in (1, 2, 3, 5, 10, 20, 40, 80):
        acc = []
        for _ in range(4000):
            est = truth + rng.normal(0, noise / np.sqrt(n), len(truth))
            acc.append(np.mean([(est[i] > est[j]) == (truth[i] > truth[j])
                                for i, j in pairs]))
        curve[n] = {"decision_accuracy": float(np.mean(acc)),
                    "gpu_h": n * len(truth) * GPU_H_PER_PULL}

    out = {
        "n_ok_pulls": int(len(ok)),
        "arm_means_pp": {k: round(v * 100, 3) for k, v in mu.items()},
        "spread_pp": round(spread * 100, 3),
        "noise_pp": round(noise * 100, 3),
        "spread_over_noise": round(spread / noise, 3),
        "decision_accuracy_vs_pulls_per_arm": curve,
        "note": ("Optimistic: treats observed arm means as ground truth. "
                 "DataDecide reports ~0.80 decision accuracy at 0.01% of target "
                 "compute on near-deterministic LLM benchmarks, where "
                 "spread/noise >> 1."),
    }
    p = os.path.join(HERE, "decision_accuracy_diagnostic.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
