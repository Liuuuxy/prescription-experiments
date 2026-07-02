#!/usr/bin/env python3
"""
TASK B -- per-instance / finer-grained targeting on robocasa rcless/ sketches.

Question: does finer-grained targeting (per-demo max, top-K mean, k-means
clusters, k-NN density, single nearest demo) beat the category-mean baseline
AUC(hard>easy) = 0.605, or does finer granularity not help (confirming the
signal is genuinely weak, not just washed out by averaging)?

We score the 9036 candidate sketches against targets built from the dval
demos (313 rows). HARD = the 10 failure categories. Each candidate has a
binary "hot" label = its category is HARD. AUC(hard>easy) measures how well
the score ranks hard candidates above easy ones. We also report the fraction
of the top-200 scored candidates that are hard ("top-200 targeted%").

Targets:
  (i)   category-mean cos   (BASELINE, expect 0.605)
  (ii)  per-demo MAX        max over hard dval demos of cos(cand, demo)
  (iii) per-demo top-K mean (K=3,5,10)
  (iv)  k-means clusters (m=4,8): max cos to centroid, and top-3-mean cos
  (v)   k-NN density: mean cos to the candidate's nearest 10 hard dval demos
  (vi)  single nearest hard dval demo (== per-demo MAX; reported for clarity)

Each target family is computed twice: using only HARD dval demos, and using
ALL dval demos, to test whether the hard-restriction matters.
"""
import json
import numpy as np

RC = "/data/xinyua11/robocasa/weakregion/rcless"
HARD_JSON = "/data/xinyua11/robocasa/weakregion/targeted_rebalanced.json"
TOPN = 200


def unit(x, ax=-1):
    n = np.linalg.norm(x, axis=ax, keepdims=True)
    return x / (n + 1e-12)


def auc(s, hot):
    """AUC that score s ranks hot (True) items above cold (False) items."""
    o = np.argsort(s)
    r = np.empty(len(s))
    r[o] = np.arange(len(s))
    nh = hot.sum()
    return (r[hot].sum() - nh * (nh - 1) / 2) / (nh * (~hot).sum())


def topn_frac(s, hot, n=TOPN):
    idx = np.argsort(-s)[:n]
    return hot[idx].mean()


def kmeans(X, m, iters=100, seed=0):
    """Tiny cosine-friendly k-means on unit rows. Returns unit centroids."""
    rng = np.random.default_rng(seed)
    c = X[rng.choice(len(X), size=m, replace=False)].copy()
    c = unit(c)
    for _ in range(iters):
        sim = X @ c.T              # (n,m) cosine since rows are unit
        assign = sim.argmax(1)
        newc = np.zeros_like(c)
        for k in range(m):
            mask = assign == k
            if mask.any():
                newc[k] = X[mask].mean(0)
            else:                  # reseed empty cluster
                newc[k] = X[rng.integers(len(X))]
        newc = unit(newc)
        if np.allclose(newc, c):
            c = newc
            break
        c = newc
    return c


def load():
    HARD = set(json.load(open(HARD_JSON))["new_targeted"])
    out = {}
    for tag in ["dval", "cand"]:
        sk = np.load(f"{RC}/{tag}_sketch.npy").astype(np.float64)
        cats = json.load(open(f"{RC}/{tag}_meta.json"))["categories"]
        ishard = np.array([c in HARD for c in cats])
        out[tag] = (unit(sk), ishard)            # sketches are unit-mean; renorm to be safe
    return out


def per_demo_scores(cand_u, demos_u):
    """cos(cand, demo) matrix (n_cand, n_demo); rows unit so cos = dot."""
    return cand_u @ demos_u.T


def run_family(cand_u, demos_u, hot, label):
    """Compute all targets for a given dval subset (hard-only or all)."""
    rows = []
    # cosine matrix once
    C = per_demo_scores(cand_u, demos_u)        # (n_cand, n_demo)

    # (i) category-mean cos  -- mean of demo directions, renormalized
    g = unit(demos_u.mean(0))
    s = cand_u @ g
    rows.append((f"[{label}] (i) category-mean cos", s))

    # (ii) per-demo MAX  == (vi) single nearest demo
    s = C.max(1)
    rows.append((f"[{label}] (ii)/(vi) per-demo MAX (1-NN)", s))

    # (iii) per-demo top-K mean
    Csort = -np.sort(-C, axis=1)                # descending per row
    for K in (3, 5, 10):
        if K <= Csort.shape[1]:
            s = Csort[:, :K].mean(1)
            rows.append((f"[{label}] (iii) top-{K} mean", s))

    # (iv) k-means clusters
    for m in (4, 8):
        if m <= len(demos_u):
            cent = kmeans(demos_u, m, seed=0)
            Cc = cand_u @ cent.T                 # (n_cand, m)
            s = Cc.max(1)
            rows.append((f"[{label}] (iv) kmeans m={m} max-cent", s))
            k3 = min(3, m)
            s = (-np.sort(-Cc, axis=1))[:, :k3].mean(1)
            rows.append((f"[{label}] (iv) kmeans m={m} top-{k3} cent", s))

    # (v) k-NN density: mean cos to nearest 10 demos
    knn = 10
    if knn <= Csort.shape[1]:
        s = Csort[:, :knn].mean(1)
        rows.append((f"[{label}] (v) kNN-10 mean", s))

    return [(name, auc(s, hot), topn_frac(s, hot)) for name, s in rows]


def main():
    d = load()
    cand_u, cand_hot = d["cand"]
    dval_u, dval_hard = d["dval"]

    print(f"n_cand={len(cand_u)}  hard={cand_hot.sum()}  easy={(~cand_hot).sum()}")
    print(f"n_dval={len(dval_u)}  hard={dval_hard.sum()}  all={len(dval_u)}")
    print()

    results = []
    # HARD-only dval target
    results += run_family(cand_u, dval_u[dval_hard], cand_hot, "HARD-dval")
    # ALL dval target (contrast)
    results += run_family(cand_u, dval_u, cand_hot, "ALL-dval")

    print(f"{'target':<42}{'AUC':>8}{'top200%':>10}")
    print("-" * 60)
    base = None
    for name, a, tf in results:
        if "(i) category-mean" in name and "HARD" in name:
            base = a
        print(f"{name:<42}{a:>8.4f}{tf*100:>9.1f}%")
    print("-" * 60)
    print(f"baseline (HARD category-mean) AUC = {base:.4f}")

    best = max(results, key=lambda r: r[1])
    print(f"best target = {best[0]}  AUC={best[1]:.4f}  top200={best[2]*100:.1f}%")
    print(f"delta vs baseline = {best[1]-base:+.4f}")


if __name__ == "__main__":
    main()
