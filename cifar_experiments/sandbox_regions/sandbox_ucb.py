"""UCB with different rewards on gradient-cluster arms (owner, 2026-08-05).

The actual adaptive bandit, run where pulls are cheap enough to afford real
dynamics: CIFAR-100 sandbox, arms = k-means clusters of the pool's per-sample
gradients (exact rank-1 factors from pool_grads.npz) + a random control.
Each pull: draw B images from the arm, fine-tune the fixed base ckpt (2 epochs,
70/30 old-new mix), measure everything. FIVE SEPARATE UCB1 LOOPS, each driven
by a different reward:

  raw      : delta overall accuracy (raw class mean)
  balanced : delta mean-of-band-means (retention-aware)
  hard     : delta hard-band accuracy (pure target)
  loss     : -delta CE on hard-band test probe (the cheap surrogate the robot
             would need -- validated here against the accuracy-driven loops)
  comp     : hard gain + min(0, med delta) + min(0, easy delta)  (gain minus
             collateral damage)

Every pull logs ALL reward variants regardless of which drives the loop.
Outputs: sandbox_regions/ucb_results.parquet (one row per pull),
ucb_summary.json (allocations + winners per loop).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, "/data/xinyua11/robocasa/cifar_experiments/sandbox_regions")
import region_sandbox as rs

OUT = "/data/xinyua11/robocasa/cifar_experiments/sandbox_regions"
K = 8
B = 2000
T_HORIZON = 40
FT_EPOCHS = 2
UCB_SCALE = 0.005          # reward scale for the exploration bonus (0.5pp)
REWARDS = ["raw", "balanced", "hard", "loss", "comp"]


def log(*a):
    print(f"[ucb {time.strftime('%H:%M:%S')}]", *a, flush=True)


@torch.no_grad()
def eval_acc_and_ce(model, test):
    from torch.utils.data import DataLoader
    model.eval()
    dl = DataLoader(test, batch_size=512, num_workers=0)
    hit = np.zeros(100); tot = np.zeros(100); ce = np.zeros(100)
    for x, y in dl:
        out = model(x)
        p = out.argmax(1)
        c = F.cross_entropy(out, y, reduction="none")
        for cls in range(100):
            m = y == cls
            tot[cls] += int(m.sum()); hit[cls] += int((p[m] == cls).sum())
            ce[cls] += float(c[m].sum())
    return hit / np.maximum(tot, 1), ce / np.maximum(tot, 1)


def main():
    import torchvision
    rep = json.load(open(f"{OUT}/report.json"))
    acc0 = np.array(rep["base_acc"])
    bands = rs.band_of(acc0)
    band_classes = {b: [c for c in range(100) if bands[c] == b] for b in rs.BANDS}

    g = np.load(f"{OUT}/pool_grads.npz", allow_pickle=True)
    feats = g["feats"].astype(np.float32); errs = g["errs"].astype(np.float32)
    pool = g["pool_idx"].astype(int); band_arr = g["band"]
    X = np.concatenate([feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9),
                        errs / (np.linalg.norm(errs, axis=1, keepdims=True) + 1e-9)], 1)
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=K, n_init=10, random_state=7).fit(X)
    lab = km.labels_
    arms = {f"gc{c}": pool[lab == c] for c in range(K)}
    arms["random"] = pool
    comp = {a: {bb: round(float((band_arr[np.isin(pool, ids)] == bb).mean()), 2)
                for bb in rs.BANDS} for a, ids in arms.items()}
    log(f"arms: { {a: len(v) for a, v in arms.items()} }")
    log(f"band composition: {comp}")

    tr = torchvision.datasets.CIFAR100(rs.DATA, train=True, download=False)
    te = torchvision.datasets.CIFAR100(rs.DATA, train=False, download=False,
                                       transform=rs.PLAIN)
    targets = np.array(tr.targets)
    rng = np.random.default_rng(rs.SEED)
    counts = {c: rs.BASE_COUNTS[c % len(rs.BASE_COUNTS)] for c in range(100)}
    base_idx = []
    for c in range(100):
        idxs = rng.permutation(np.where(targets == c)[0])
        base_idx += list(idxs[:counts[c]])
    old_fixed = rng.choice(base_idx, rs.OLD_MIX, replace=False)

    base_model = rs.make_model()
    base_model.load_state_dict(torch.load(f"{OUT}/base_ckpt.pt"))
    _, ce0 = eval_acc_and_ce(base_model, te)
    ce0_hard = float(np.mean([ce0[c] for c in band_classes["hard"]]))

    def pull(arm, seed):
        r = np.random.default_rng(seed)
        ids = arms[arm]
        add = r.choice(ids, min(B, len(ids)), replace=False)
        m = rs.make_model(); m.load_state_dict(torch.load(f"{OUT}/base_ckpt.pt"))
        rs.train(m, rs.Sub(tr, list(old_fixed) + list(add), rs.AUG),
                 epochs=FT_EPOCHS, lr=0.01, seed=seed)
        acc, ce = eval_acc_and_ce(m, te)
        d = {b: float(np.mean([acc[c] - acc0[c] for c in band_classes[b]]))
             for b in rs.BANDS}
        ce_hard = float(np.mean([ce[c] for c in band_classes["hard"]]))
        return {"raw": float(acc.mean() - acc0.mean()),
                "balanced": float(np.mean(list(d.values()))),
                "hard": d["hard"],
                "loss": ce0_hard - ce_hard,
                "comp": d["hard"] + min(0.0, d["med"]) + min(0.0, d["easy"]),
                **{f"d_{b}": d[b] for b in rs.BANDS}}

    arm_names = list(arms)
    all_rows = []
    summary = {}
    seed_counter = 50000
    for drive in REWARDS:
        means = {a: 0.0 for a in arm_names}; ns = {a: 0 for a in arm_names}
        rewards_seen = []
        for t in range(1, T_HORIZON + 1):
            if t <= len(arm_names):
                a = arm_names[t - 1]           # one init pull per arm
            else:
                ucb = {a: means[a] + UCB_SCALE * np.sqrt(2 * np.log(t) / ns[a])
                       for a in arm_names}
                a = max(ucb, key=ucb.get)
            seed_counter += 1
            t0 = time.time()
            res = pull(a, seed_counter)
            rwd = res[drive]
            ns[a] += 1; means[a] += (rwd - means[a]) / ns[a]
            rewards_seen.append(rwd)
            all_rows.append({"drive": drive, "t": t, "arm": a, "reward": rwd, **res})
            log(f"[{drive}] t={t} arm={a} reward={rwd*100:+.2f} "
                f"(raw {res['raw']*100:+.2f} bal {res['balanced']*100:+.2f} "
                f"hard {res['hard']*100:+.2f}) {time.time()-t0:.0f}s")
            pd.DataFrame(all_rows).to_parquet(f"{OUT}/ucb_results.parquet")
        best = max(means, key=lambda a: means[a] if ns[a] > 0 else -9)
        summary[drive] = {"allocation": dict(ns), "means_pp": {a: round(m * 100, 2)
                          for a, m in means.items()}, "best_arm": best}
        log(f"LOOP {drive} DONE: best={best} alloc={dict(ns)}")
        json.dump({"summary": summary, "cluster_composition": comp},
                  open(f"{OUT}/ucb_summary.json", "w"))
    log("UCB EXPLORATION COMPLETE")


if __name__ == "__main__":
    main()
