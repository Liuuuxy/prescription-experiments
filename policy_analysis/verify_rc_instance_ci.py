#!/usr/bin/env python3
"""
Adversarial follow-up: is the top-10/kNN "+0.006 win" inside noise?
Bootstrap CI on the AUC delta (top-10 vs category-mean), paired over candidates.
Also: is the m=8 top-3=0.6162 just one lucky seed? Compare its percentile vs the
10-seed distribution. And test whether a RANDOM unit target gets AUC>0.5 (null).
"""
import json
import numpy as np

RC = "/data/xinyua11/robocasa/weakregion/rcless"
HARD_JSON = "/data/xinyua11/robocasa/weakregion/targeted_rebalanced.json"


def unit(x, ax=-1):
    n = np.linalg.norm(x, axis=ax, keepdims=True)
    return x / (n + 1e-12)


def auc(s, hot):
    s = np.asarray(s, float); hot = np.asarray(hot, bool)
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s)); r[o] = np.arange(len(s))
    nh = hot.sum()
    return (r[hot].sum() - nh*(nh-1)/2.0) / (nh*(~hot).sum())


def main():
    HARD = set(json.load(open(HARD_JSON))["new_targeted"])
    cand = unit(np.load(f"{RC}/cand_sketch.npy").astype(np.float64))
    dval = unit(np.load(f"{RC}/dval_sketch.npy").astype(np.float64))
    cand_cats = json.load(open(f"{RC}/cand_meta.json"))["categories"]
    dval_cats = json.load(open(f"{RC}/dval_meta.json"))["categories"]
    hot = np.array([c in HARD for c in cand_cats])
    dhard = np.array([c in HARD for c in dval_cats])
    dh = dval[dhard]

    g = unit(dh.mean(0))
    s_base = cand @ g
    C = cand @ dh.T
    Cs = -np.sort(-C, axis=1)
    s_top10 = Cs[:, :10].mean(1)
    s_top5 = Cs[:, :5].mean(1)

    a_base = auc(s_base, hot); a_t10 = auc(s_top10, hot)
    print(f"base AUC={a_base:.4f}  top10 AUC={a_t10:.4f}  delta={a_t10-a_base:+.4f}")

    # Paired bootstrap over candidates: resample candidates, recompute both AUCs
    rng = np.random.default_rng(0)
    idx_hot = np.where(hot)[0]; idx_cold = np.where(~hot)[0]
    deltas = []; t10s = []; bases = []
    B = 2000
    for _ in range(B):
        bh = rng.choice(idx_hot, size=len(idx_hot), replace=True)
        bc = rng.choice(idx_cold, size=len(idx_cold), replace=True)
        ii = np.concatenate([bh, bc])
        h = np.concatenate([np.ones(len(bh), bool), np.zeros(len(bc), bool)])
        ab = auc(s_base[ii], h); at = auc(s_top10[ii], h)
        bases.append(ab); t10s.append(at); deltas.append(at-ab)
    deltas = np.array(deltas)
    print(f"bootstrap delta(top10-base): mean={deltas.mean():+.4f} "
          f"95%CI=[{np.percentile(deltas,2.5):+.4f},{np.percentile(deltas,97.5):+.4f}] "
          f"P(delta>0)={(deltas>0).mean():.3f}")
    print(f"bootstrap base AUC 95%CI=[{np.percentile(bases,2.5):.4f},{np.percentile(bases,97.5):.4f}]")

    # m=8 top-3 over MANY seeds to see where 0.6162 sits
    def kmeans(X, m, seed):
        r = np.random.default_rng(seed)
        c = unit(X[r.choice(len(X), m, replace=False)].copy())
        for _ in range(100):
            a = (X @ c.T).argmax(1)
            nc = np.zeros_like(c)
            for k in range(m):
                msk = a == k
                nc[k] = X[msk].mean(0) if msk.any() else X[r.integers(len(X))]
            nc = unit(nc)
            if np.allclose(nc, c): break
            c = nc
        return c
    a8 = []
    for sd in range(50):
        cent = kmeans(dh, 8, sd)
        Cc = cand @ cent.T
        a8.append(auc((-np.sort(-Cc, 1))[:, :3].mean(1), hot))
    a8 = np.array(a8)
    print(f"\nm=8 top-3 over 50 seeds: mean={a8.mean():.4f}+/-{a8.std():.4f} "
          f"min={a8.min():.4f} max={a8.max():.4f}")
    print(f"  fraction of seeds >= baseline 0.6067: {(a8>=a_base).mean():.2f}")
    print(f"  fraction of seeds >= the reported 0.6162: {(a8>=0.6162).mean():.2f}")
    print(f"  seed=0 value: {a8[0]:.4f}")

    # NULL control: random unit targets — what AUC do they get?
    rng2 = np.random.default_rng(7)
    null = []
    for _ in range(200):
        v = unit(rng2.standard_normal(cand.shape[1]))
        null.append(auc(cand @ v, hot))
    null = np.array(null)
    print(f"\nrandom-direction null AUC: mean={null.mean():.4f}+/-{null.std():.4f} "
          f"max={null.max():.4f}  (base {a_base:.4f} is {(a_base-null.mean())/null.std():.1f} sigma above null)")


if __name__ == "__main__":
    main()
