#!/usr/bin/env python3
"""
INDEPENDENT adversarial re-verification of TASK per_instance.
Written from scratch — does NOT import rc_instance.py.
Checks:
  - baseline category-mean cos AUC reproduces ~0.605
  - per-demo max / top-K mean / kNN / kmeans numbers
  - AUC direction (hard>easy) and tie handling
  - kmeans seed robustness
  - sanity: flipped-direction AUC, mean-as-control
"""
import json
import numpy as np

RC = "/data/xinyua11/robocasa/weakregion/rcless"
HARD_JSON = "/data/xinyua11/robocasa/weakregion/targeted_rebalanced.json"


def unit(x, ax=-1):
    n = np.linalg.norm(x, axis=ax, keepdims=True)
    return x / (n + 1e-12)


def auc_ranksum(s, hot):
    """AUC(score ranks hot above cold), midrank ties via scipy-free averaging."""
    s = np.asarray(s, float)
    hot = np.asarray(hot, bool)
    # average ranks to handle ties properly
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    sorted_s = s[order]
    i = 0
    n = len(s)
    # assign 0-based ranks first
    base = np.empty(n)
    base[order] = np.arange(n)
    # now average over tie groups
    ranks[order] = np.arange(n, dtype=float)
    j = 0
    while j < n:
        k = j
        while k + 1 < n and sorted_s[k + 1] == sorted_s[j]:
            k += 1
        if k > j:
            avg = (j + k) / 2.0
            ranks[order[j:k + 1]] = avg
        j = k + 1
    nh = hot.sum()
    ne = (~hot).sum()
    # use 1-based-ish rank-sum; with 0-based ranks subtract nh*(nh-1)/2
    return (ranks[hot].sum() - nh * (nh - 1) / 2.0) / (nh * ne)


def auc_brute(s, hot):
    """Brute-force pairwise AUC with 0.5 for ties — ground truth, O(nh*ne)."""
    s = np.asarray(s, float)
    hot = np.asarray(hot, bool)
    sh = s[hot]
    se = s[~hot]
    # count pairs sh>se + 0.5*ties
    sh = np.sort(sh)
    se = np.sort(se)
    # for each easy value, count hard greater / equal
    gt = 0.0
    for v in se:
        gt += (sh > v).sum() + 0.5 * (sh == v).sum()
    return gt / (len(sh) * len(se))


def topn_frac(s, hot, n=200):
    idx = np.argsort(-s)[:n]
    return np.asarray(hot, bool)[idx].mean()


def kmeans(X, m, iters=100, seed=0):
    rng = np.random.default_rng(seed)
    c = unit(X[rng.choice(len(X), size=m, replace=False)].copy())
    for _ in range(iters):
        sim = X @ c.T
        assign = sim.argmax(1)
        newc = np.zeros_like(c)
        for k in range(m):
            mask = assign == k
            newc[k] = X[mask].mean(0) if mask.any() else X[rng.integers(len(X))]
        newc = unit(newc)
        if np.allclose(newc, c):
            break
        c = newc
    return c


def main():
    HARD = set(json.load(open(HARD_JSON))["new_targeted"])
    print("HARD cats:", sorted(HARD))

    cand = np.load(f"{RC}/cand_sketch.npy").astype(np.float64)
    dval = np.load(f"{RC}/dval_sketch.npy").astype(np.float64)
    cand_cats = json.load(open(f"{RC}/cand_meta.json"))["categories"]
    dval_cats = json.load(open(f"{RC}/dval_meta.json"))["categories"]

    cand_u = unit(cand)
    dval_u = unit(dval)
    cand_hot = np.array([c in HARD for c in cand_cats])
    dval_hard = np.array([c in HARD for c in dval_cats])

    print(f"n_cand={len(cand_u)} hard={cand_hot.sum()} easy={(~cand_hot).sum()}")
    print(f"n_dval={len(dval_u)} hard={dval_hard.sum()} all={len(dval_u)}")

    # cross-check the two AUC implementations on a random score
    rng = np.random.default_rng(1)
    rs = rng.standard_normal(len(cand_u))
    print(f"\n[AUC self-check] ranksum={auc_ranksum(rs,cand_hot):.6f} brute={auc_brute(rs,cand_hot):.6f} (expect ~0.5)")

    dh = dval_u[dval_hard]
    print(f"\n=== HARD-dval targets (n_hard_demos={len(dh)}) ===")

    def report(name, s):
        a = auc_ranksum(s, cand_hot)
        ab = auc_brute(s, cand_hot)
        tf = topn_frac(s, cand_hot)
        flag = "" if abs(a - ab) < 1e-6 else "  <-- ties differ!"
        print(f"  {name:<28} AUC={a:.4f} (brute {ab:.4f}) top200={tf*100:.1f}%{flag}")
        return a

    # (i) category-mean
    g = unit(dh.mean(0))
    a_base = report("(i) category-mean cos", cand_u @ g)

    C = cand_u @ dh.T  # (n_cand, n_hard_demos)
    # (ii) max / 1-NN
    report("(ii) per-demo MAX (1-NN)", C.max(1))
    Cs = -np.sort(-C, axis=1)
    for K in (3, 5, 10):
        report(f"(iii) top-{K} mean", Cs[:, :K].mean(1))
    # (iv) kmeans
    for m in (4, 8):
        cent = kmeans(dh, m, seed=0)
        Cc = cand_u @ cent.T
        report(f"(iv) kmeans m={m} max", Cc.max(1))
        k3 = min(3, m)
        report(f"(iv) kmeans m={m} top{k3}", (-np.sort(-Cc, axis=1))[:, :k3].mean(1))
    # (v) kNN-10
    report("(v) kNN-10 mean", Cs[:, :10].mean(1))

    # flipped-direction sanity
    af = auc_ranksum(-(cand_u @ g), cand_hot)
    print(f"\n[direction check] flipped category-mean AUC={af:.4f} (should be 1-base={1-a_base:.4f})")

    # === ALL-dval contrast ===
    print(f"\n=== ALL-dval targets (n={len(dval_u)}) ===")
    g_all = unit(dval_u.mean(0))
    report("(i) category-mean cos", cand_u @ g_all)
    Ca = cand_u @ dval_u.T
    report("(ii) per-demo MAX", Ca.max(1))
    Cas = -np.sort(-Ca, axis=1)
    report("(iii) top-3 mean", Cas[:, :3].mean(1))
    report("(iii) top-10 mean", Cas[:, :10].mean(1))
    cent = kmeans(dval_u, 4, seed=0)
    Cc = cand_u @ cent.T
    report("(iv) kmeans m=4 top3", (-np.sort(-Cc, axis=1))[:, :3].mean(1))
    report("(v) kNN-10 mean", Cas[:, :10].mean(1))

    # === kmeans seed robustness ===
    print(f"\n=== kmeans seed robustness (10 seeds, HARD-dval) ===")
    for m in (4, 8):
        aucs = []
        for sd in range(10):
            cent = kmeans(dh, m, seed=sd)
            Cc = cand_u @ cent.T
            s = (-np.sort(-Cc, axis=1))[:, :3].mean(1)
            aucs.append(auc_ranksum(s, cand_hot))
        aucs = np.array(aucs)
        print(f"  m={m} top-3: mean={aucs.mean():.4f} +/- {aucs.std():.4f} "
              f"min={aucs.min():.4f} max={aucs.max():.4f}")

    print(f"\nBASELINE (HARD category-mean) = {a_base:.4f}")


if __name__ == "__main__":
    main()
