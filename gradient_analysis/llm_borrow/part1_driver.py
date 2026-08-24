"""PART 1 driver: every allocator x every budget x N replicates, in the NORMAL regime.

Resumable and incremental: one JSON cell per (mode, budget, allocator), written after
each cell, so a run that is stopped early still reports everything it finished.

  python part1_driver.py --mode synthetic_robot --budgets 1,2,3,4,6,8,12,16,24,32,48 --reps 500
  CUDA_VISIBLE_DEVICES=0 python part1_driver.py --mode cifar --budgets 1.5,3,6,12,24 --reps 5 \
      --allocators successive_halving,random,oracle,jest,jest_then_confirm,regmix
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from race_lib import atomic_json, run_cell, slim  # noqa: E402
from wind_tunnel import (ALLOCATOR_FACTORIES, HERE, SPEC_CIFAR_LIKE,  # noqa: E402
                         SPEC_ROBOT_LIKE, CifarBackend, SyntheticBackend)


def make_backend(mode: str, cache: str | None, verbose: bool = False):
    if mode == "cifar":
        return CifarBackend(cache_path=cache or f"{HERE}/wind_tunnel_cifar_cache.json",
                            verbose=verbose), [0, 1, 2, 3, 4, 5, 6, 7]
    spec = SPEC_CIFAR_LIKE if mode == "synthetic" else SPEC_ROBOT_LIKE
    return SyntheticBackend(spec), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="synthetic_robot",
                    choices=["synthetic", "synthetic_robot", "cifar"])
    ap.add_argument("--budgets", default="1,2,3,4,6,8,12,16,24,32,48")
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--allocators", default=",".join(ALLOCATOR_FACTORIES))
    ap.add_argument("--seed-pool", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--time-budget-s", type=float, default=1e9)
    a = ap.parse_args()

    budgets = [float(x) for x in a.budgets.split(",")]
    allocs = a.allocators.split(",")
    backend, pool = make_backend(a.mode, a.cache)
    if a.seed_pool:
        pool = [int(x) for x in a.seed_pool.split(",")]
    out_path = a.out or f"{HERE}/part1_{a.mode}.json"
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    out.setdefault("mode", a.mode)
    out.setdefault("reps", a.reps)
    out.setdefault("seed_pool", pool)
    out.setdefault("cells", {})
    truth = backend.true_values()
    out["truth"] = truth
    print(f"[p1] mode={a.mode} truth(pp)=" +
          " ".join(f"{k}={v*100:+.2f}" for k, v in sorted(truth.items())), flush=True)

    t0 = time.time()
    for b in budgets:
        for name in allocs:
            key = f"{b}|{name}"
            if key in out["cells"] and out["cells"][key].get("reps") == a.reps:
                print(f"[p1] skip {key} (done)", flush=True)
                continue
            if time.time() - t0 > a.time_budget_s:
                print(f"[p1] time budget reached; stopping before {key}", flush=True)
                atomic_json(out_path, out)
                return
            c = run_cell(backend, name, b, a.reps, seed_pool=pool, truth=truth)
            out["cells"][key] = slim(c)
            atomic_json(out_path, out)
            print(f"[p1] {key:34s} regret {c['mean_regret']*100:7.3f} "
                  f"+/-{c['se_regret']*100:5.3f}pp  P(corr) {c['p_correct']:.2f}  "
                  f"spent {c['mean_spent_fte']:.2f}  {c['wall_s']:.0f}s  "
                  f"picks {c['pick_counts']}", flush=True)
    atomic_json(out_path, out)
    print(f"[p1] wrote {out_path} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
