"""CIFAR analogue of the clean object test (owner, 2026-08-14).

Robot finding to test: adding demos of the objects the policy currently FAILS
at made it worse overall and no better on those objects (paired -4.67).
Analogue: add images of the classes the base model is WORST at vs images of
the classes it is BEST at, budget-matched, everything else equal.

Arms (B images each, 6 seeds, fixed batches):
  hard_cls : from the 6 lowest-accuracy classes of the base model
  easy_cls : from the 6 highest-accuracy classes
  rand_ctrl: uniform from the pool
B = min(availability) so the three arms are exactly budget-matched.
Reports overall + per-band + on-target deltas.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/data/xinyua11/robocasa/cifar_experiments/sandbox_regions")
import region_sandbox as rs

OUT = "/data/xinyua11/robocasa/cifar_experiments/sandbox_regions"
N_SEEDS = 6


def main():
    import torchvision
    rep = json.load(open(f"{OUT}/report.json"))
    acc0 = np.array(rep["base_acc"])
    order = np.argsort(acc0)
    hard_cls = list(order[:6]); easy_cls = list(order[-6:])
    print(f"[objcifar] hard classes {hard_cls} acc={acc0[hard_cls].round(2)}", flush=True)
    print(f"[objcifar] easy classes {easy_cls} acc={acc0[easy_cls].round(2)}", flush=True)

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
    pool_cls = targets[pool]
    n_hard = int(np.isin(pool_cls, hard_cls).sum())
    n_easy = int(np.isin(pool_cls, easy_cls).sum())
    B = min(n_hard, n_easy)
    print(f"[objcifar] availability hard={n_hard} easy={n_easy} -> B={B}", flush=True)
    old_fixed = rng.choice(base_idx, rs.OLD_MIX, replace=False)
    bands = rs.band_of(acc0)
    band_classes = {b: [c for c in range(100) if bands[c] == b] for b in rs.BANDS}

    batches = {
        "hard_cls": rng.choice(pool[np.isin(pool_cls, hard_cls)], B, replace=False),
        "easy_cls": rng.choice(pool[np.isin(pool_cls, easy_cls)], B, replace=False),
        "rand_ctrl": rng.choice(pool, B, replace=False),
    }
    rows = []
    for arm, add in batches.items():
        for s in range(N_SEEDS):
            m = rs.make_model(); m.load_state_dict(torch.load(f"{OUT}/base_ckpt.pt"))
            t0 = time.time()
            rs.train(m, rs.Sub(tr, list(old_fixed) + list(add), rs.AUG),
                     epochs=3, lr=0.01, seed=7000 + s)
            acc = rs.per_class_acc(m, te)
            d = {b: float(np.mean([acc[c] - acc0[c] for c in band_classes[b]]))
                 for b in rs.BANDS}
            rows.append({"arm": arm, "seed": s,
                         "d_overall": float(acc.mean() - acc0.mean()),
                         "d_hard_cls": float(np.mean([acc[c] - acc0[c] for c in hard_cls])),
                         "d_easy_cls": float(np.mean([acc[c] - acc0[c] for c in easy_cls])),
                         **{f"d_{b}": d[b] for b in rs.BANDS}})
            print(f"[objcifar] {arm} s={s} ({(time.time()-t0)/60:.1f}m) "
                  f"overall={rows[-1]['d_overall']*100:+.2f} "
                  f"on_hard_cls={rows[-1]['d_hard_cls']*100:+.2f} "
                  f"on_easy_cls={rows[-1]['d_easy_cls']*100:+.2f}", flush=True)
            pd.DataFrame(rows).to_parquet(f"{OUT}/objtest_cifar.parquet")
    df = pd.DataFrame(rows)
    print("\n[objcifar] SUMMARY (mean +- sd over 6 seeds, pp):", flush=True)
    for arm in batches:
        g = df[df.arm == arm]
        print(f"  {arm:10s} overall {g.d_overall.mean()*100:+.2f}+-{g.d_overall.std()*100:.2f}  "
              f"on-hard-classes {g.d_hard_cls.mean()*100:+.2f}  "
              f"on-easy-classes {g.d_easy_cls.mean()*100:+.2f}", flush=True)
    print("[objcifar] COMPLETE", flush=True)


if __name__ == "__main__":
    main()
