"""kappa-sweep frontier — the LAST simulator work (frozen allocators, frozen
corruptions; this script only varies kappa and reports, it selects nothing).

For each kappa: per-corruption P(pick best) for the calibrated gate, the
gate-open fraction (proxy utilization), and the reviewer's worst-case
degradation D(kappa) = max over corruptions of [P_rollout - P_gate].
kappa=1.0 remains the A-PRIORI frozen operating point (pre-registered before
this sweep ran); the frontier exists so the paper shows the whole tradeoff
instead of one convenient point.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/data/xinyua11/robocasa/sandbox_regions")
import proxy_continuum as pc

KAPPAS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
MODES = ["clean", "noise4", "tail_invert", "interior_invert_top", "goodhart_spike"]
REPS = 3000
SCEN = "poison"


def run_alloc(fn, mode, tag):
    hit = 0.0
    for rep in range(REPS):
        rng = np.random.default_rng(777_000 * (hash((SCEN, mode, tag)) % 2**20) + rep)
        mu, pidx = pc.landscape(SCEN, rng)
        phi, badm = pc.corrupt(mu, mode, rng, pidx)
        a = fn(mu, phi, rng)
        hit += (a == int(np.argmax(mu)))
    return hit / REPS


def main():
    base = {m: run_alloc(pc.alloc_rollout_se, m, "base") for m in MODES}
    print("rollout-only baseline:", {m: f"{v:.1%}" for m, v in base.items()})
    out = {"baseline": base, "kappa": {}}
    for k in KAPPAS:
        pc.KAPPA = k
        row = {m: run_alloc(lambda mu, phi, rng: pc.alloc_calib_gate(mu, phi, rng, local=False),
                            m, f"k{k}") for m in MODES}
        D = max(base[m] - row[m] for m in MODES)
        gain = row["clean"] - base["clean"]
        out["kappa"][k] = {"cells": row, "worst_case_D": round(D, 4),
                           "clean_gain": round(gain, 4)}
        print(f"kappa={k:4}: " + "  ".join(f"{m}={row[m]:.1%}" for m in MODES)
              + f"  | worst-case D={D*100:+.1f}pp  clean gain={gain*100:+.1f}pp", flush=True)
    json.dump(out, open("/data/xinyua11/robocasa/sandbox_regions/kappa_frontier.json", "w"), indent=1)
    print("KAPPA FRONTIER COMPLETE")


if __name__ == "__main__":
    main()
