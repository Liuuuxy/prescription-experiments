"""Mixture-ratio sweep (owner hypothesis, 2026-08-04): is there a band mixture
that beats random? Evidence for yes already: random (+3.11) beat the best pure
vertex (hard +2.81) -> superadditive mixing -> interior optimum exists.

Arms: ratio ladders over (hard/med/easy) incl. the owner's 70/25/5, plus a
continuous 'headroom' arm (per-class weight ~ 1 - base_acc). 6 seeds each,
same base ckpt / recipe / 70-30 old-new mix as region_sandbox.py.
"""
import json, os, sys, time
import numpy as np, pandas as pd
import torch

sys.path.insert(0, "/data/xinyua11/robocasa/sandbox_regions")
import region_sandbox as rs

OUT = "/data/xinyua11/robocasa/sandbox_regions"
B, N_SEEDS = rs.B, 6
MIXES = {                      # (hard, med, easy) fractions of B
    "mix_85_13_2": (0.85, 0.13, 0.02),
    "mix_70_25_5": (0.70, 0.25, 0.05),   # the owner's suggestion
    "mix_55_35_10": (0.55, 0.35, 0.10),
    "mix_40_50_10": (0.40, 0.50, 0.10),
    "headroom": None,                     # per-class ~ (1 - acc0)
}

def main():
    import torchvision
    rep = json.load(open(f"{OUT}/report.json"))
    acc0 = np.array(rep["base_acc"])
    tr = torchvision.datasets.CIFAR100(rs.DATA, train=True, download=False)
    te = torchvision.datasets.CIFAR100(rs.DATA, train=False, download=False,
                                       transform=rs.PLAIN)
    targets = np.array(tr.targets)
    rng = np.random.default_rng(rs.SEED)
    counts = {c: rs.BASE_COUNTS[c % len(rs.BASE_COUNTS)] for c in range(100)}
    base_idx, pool_idx = [], []
    for c in range(100):
        idxs = rng.permutation(np.where(targets == c)[0])
        base_idx += list(idxs[:counts[c]]); pool_idx += list(idxs[counts[c]:])
    pool = np.array(pool_idx)
    bands = rs.band_of(acc0)
    pool_band = np.array([bands[targets[i]] for i in pool])
    pool_frac = {b: float((pool_band == b).mean()) for b in rs.BANDS}
    rs.log(f"pool band fractions (== random arm's implicit mix): {pool_frac}")
    old_fixed = rng.choice(base_idx, rs.OLD_MIX, replace=False)

    rows = []
    for arm, mix in MIXES.items():
        for s in range(N_SEEDS):
            r = np.random.default_rng(3000 + 61 * s + hash(arm) % 997)
            if mix is None:  # headroom: per-class weight ~ (1 - acc0)
                w = np.array([1 - acc0[targets[i]] for i in pool]); w /= w.sum()
                add = r.choice(pool, B, replace=False, p=w)
            else:
                add = []
                for frac, b in zip(mix, ["hard", "med", "easy"]):
                    src = pool[pool_band == b]
                    k = min(int(round(B * frac)), len(src))
                    add += list(r.choice(src, k, replace=False))
                add = np.array(add)
            m = rs.make_model(); m.load_state_dict(torch.load(f"{OUT}/base_ckpt.pt"))
            t0 = time.time()
            rs.train(m, rs.Sub(tr, list(old_fixed) + list(add), rs.AUG),
                     epochs=3, lr=0.01, seed=4000 + s)
            acc = rs.per_class_acc(m, te)
            d = {b: float(np.mean([acc[c] - acc0[c] for c in range(100)
                                   if bands[c] == b])) for b in rs.BANDS}
            rows.append({"arm": arm, "seed": s, "n_add": len(add),
                         "d_overall": float(acc.mean() - acc0.mean()),
                         **{f"d_{b}": d[b] for b in rs.BANDS},
                         "acc_json": json.dumps([round(float(a), 4) for a in acc])})
            rs.log(f"ARM DONE {arm} seed={s} ({(time.time()-t0)/60:.1f}min) "
                   f"d_overall={rows[-1]['d_overall']:+.4f} d_hard={d['hard']:+.4f} "
                   f"d_med={d['med']:+.4f} d_easy={d['easy']:+.4f}")
            pd.DataFrame(rows).to_parquet(f"{OUT}/mixture_results.parquet")
    rs.log("MIXTURE SWEEP COMPLETE")

if __name__ == "__main__":
    main()
