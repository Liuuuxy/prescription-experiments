"""Calibrate the wind tunnel's CIFAR backend: (1) how reproducible is a pull at a
FIXED seed, (2) do cheap short-training proxies rank the arms like full pulls?

(1) matters because the harness memoizes pulls by (mixture, seed, steps) and the
armrace results are imported as cache entries. The first import check showed a
0.85pp gap between the imported value and a from-scratch recompute of the SAME
fine-tune -- so "same seed" does not mean "same result" here, and the size of
that gap is the wind tunnel's real floor.

(2) is measured constraint 3 (cheap proxies MIS-RANK) evaluated in the sandbox.

Run (robocasa env, ~10 min shared-GPU):
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python wind_tunnel_calibration.py
Writes wind_tunnel_calibration.json here. Restores the cache it perturbs.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wind_tunnel as wt  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "wind_tunnel_calibration.json")


def repeats(b, arm, seed, k, steps=wt.FULL_STEPS):
    """Re-run the SAME pull k times from scratch (cache bypassed each time)."""
    key = b._key({arm: 1.0}, seed, steps)
    saved = b.cache.get(key)
    ys = []
    for _ in range(k):
        b.cache.pop(key, None)
        y, _, _ = b.run_pull({arm: 1.0}, seed, steps)
        ys.append(y)
    if saved is not None:                       # restore the declared-truth entry
        b.cache[key] = saved
        b._persist()
    return ys


def main():
    b = wt.CifarBackend()
    b._setup()
    truth = b.true_values()
    out = {"truth_pp": {k: round(v * 100, 3) for k, v in truth.items()}}

    rep = {}
    for arm, seed, k in (("rare", 0, 2), ("null", 0, 2), ("random", 0, 2)):
        ys = repeats(b, arm, seed, k)
        imported = b.cache.get(b._key({arm: 1.0}, seed, wt.FULL_STEPS))
        allv = ([imported] if imported is not None else []) + ys
        rep[f"{arm}_s{seed}"] = {
            "armrace_value_pp": round((imported or float("nan")) * 100, 3),
            "recomputes_pp": [round(y * 100, 3) for y in ys],
            "sd_across_identical_runs_pp": round(float(np.std(allv, ddof=1)) * 100, 3),
            "range_pp": round((max(allv) - min(allv)) * 100, 3)}
        print(f"[cal] {arm} s{seed}: {rep[f'{arm}_s{seed}']}", flush=True)
    out["same_seed_reproducibility"] = rep
    sds = [v["sd_across_identical_runs_pp"] for v in rep.values()]
    out["same_seed_sd_pp_mean"] = round(float(np.mean(sds)), 3)

    out["proxy_fidelity"] = wt.check_proxy_fidelity(b, steps=(250, 500), seeds=(0, 1))
    print("[cal] proxy fidelity:", json.dumps(out["proxy_fidelity"], indent=1), flush=True)

    out["cifar_compute"] = {"n_real_finetunes": b.n_compute,
                            "compute_wall_s": round(b.compute_wall_s, 1)}
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print(f"[cal] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
