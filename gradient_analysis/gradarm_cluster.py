"""Gradient-space arm definition for the "well-defined arms" test (2026-08-01).

Clusters ALL 9,485 W-pool demos by their LoRA-gradient sketches (pi0base/19999,
2048-d JL, merged from sketches_pi0base_19999/ + sketches_pi0base_19999_pool/),
then picks the two most-separated clusters as arms gradarm_a / gradarm_b.

Preprocessing (validated by Q0, gradient_analysis/Q0_REPORT.md):
  unit-normalize -> remove top-10 SVD modes (the ~0.20-cos generic pick-place
  common mode; deeper whitening collapsed AUC in Q0's ablation) -> renormalize.

Diagnostics (cluster_study.py conventions): bootstrap-ARI stability,
triviality ARI vs condition-region labels (did gradient space just rediscover
tall/mid/easy?), composition tables (region / category / height / traj_len /
grad-norm per cluster), silhouette.

Output: gradient_analysis/gradarm_assignment.json + gradarm_report.json
CPU-only; run in the robocasa env.
"""
import json
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "/data/xinyua11/robocasa")

GA = "/data/xinyua11/robocasa/gradient_analysis"
DIRS = [f"{GA}/sketches_pi0base_19999", f"{GA}/sketches_pi0base_19999_pool"]
LISTS = f"{GA}/demo_lists.json"
KS = [4, 6, 8]
K_PRIMARY = 6
TOP_MODES = 10
MIN_ARM = 1200          # arm needs >=1200 demos so 3 draws of B=200 stay diverse
SEED = 20260801


def load_merged():
    lists = json.load(open(LISTS))
    region = {int(e): r for e, r in lists["region_assignment"].items()}
    rows, seen = {}, set()
    for d in DIRS:
        meta = json.load(open(f"{d}/episodes.json"))
        S = np.load(f"{d}/sketches.npy")
        N = np.load(f"{d}/norms.npy")
        assert meta["proj_seed"] == 12345 and meta["ckpt_step"] == 19999, d
        for i, e in enumerate(meta["episodes"]):
            e = int(e)
            if e in region and e not in seen:
                rows[e] = (S[i], float(N[i]))
                seen.add(e)
    eps = sorted(rows)
    X = np.stack([rows[e][0] for e in eps]).astype(np.float64)
    norms = np.array([rows[e][1] for e in eps])
    reg = np.array([region[e] for e in eps])
    return eps, X, norms, reg


def whiten(X, top_modes=TOP_MODES):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    mu = Xn.mean(0, keepdims=True)
    Xc = Xn - mu
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    Xw = Xc - (Xc @ Vt[:top_modes].T) @ Vt[:top_modes]
    Xw = Xw / (np.linalg.norm(Xw, axis=1, keepdims=True) + 1e-12)
    ev = (s**2) / (s**2).sum()
    return Xw, ev[:top_modes].sum()


def main():
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    eps, X, norms, reg = load_merged()
    print(f"[cluster] merged sketches: {len(eps)} pool demos "
          f"(regions: {dict(Counter(reg))})", flush=True)
    assert len(eps) == 9485, f"expected full pool, got {len(eps)}"

    Xw, ev_removed = whiten(X)
    print(f"[cluster] whitened: removed top-{TOP_MODES} modes "
          f"({ev_removed:.1%} of variance)", flush=True)

    import pandas as pd
    from bandit_v1 import pool as bpool
    pdf = bpool.build_pool_table(write=False).set_index("episode_index")
    pdf = pdf.loc[eps]

    rng = np.random.default_rng(SEED)
    report = {"n": len(eps), "top_modes_removed": TOP_MODES,
              "variance_removed": float(ev_removed), "ks": {}}
    labels_by_k = {}
    for k in KS:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xw)
        lab = km.labels_
        labels_by_k[k] = (lab, km.cluster_centers_)
        sub = rng.choice(len(eps), 3000, replace=False)
        sil = float(silhouette_score(Xw[sub], lab[sub]))
        ari_region = float(adjusted_rand_score(reg, lab))
        boots = []
        for b in range(5):
            idx = rng.choice(len(eps), int(0.8 * len(eps)), replace=False)
            kb = KMeans(n_clusters=k, n_init=5, random_state=SEED + 1 + b).fit(Xw[idx])
            boots.append(float(adjusted_rand_score(lab[idx], kb.labels_)))
        sizes = dict(Counter(int(x) for x in lab))
        comp = {}
        for c in range(k):
            m = lab == c
            comp[c] = {
                "n": int(m.sum()),
                "region_share": {r: round(float((reg[m] == r).mean()), 3)
                                 for r in np.unique(reg)},
                "top_categories": Counter(pdf["category"][m]).most_common(3),
                "mean_h": round(float(pdf["h"][m].mean()), 4),
                "mean_traj_len": round(float(pdf["traj_len"][m].mean()), 1),
                "mean_grad_norm": round(float(norms[m].mean()), 3),
            }
        report["ks"][k] = {"silhouette": sil, "ari_vs_region": ari_region,
                           "bootstrap_ari": boots,
                           "bootstrap_ari_mean": float(np.mean(boots)),
                           "sizes": sizes, "composition": comp}
        print(f"[cluster] k={k}: sil={sil:.3f} ARI-vs-region={ari_region:.3f} "
              f"bootARI={np.mean(boots):.3f} sizes={sizes}", flush=True)

    # arm pick: primary k, two most-separated clusters with >= MIN_ARM members
    def pick(k):
        lab, cents = labels_by_k[k]
        sizes = Counter(int(x) for x in lab)
        big = [c for c in range(k) if sizes[c] >= MIN_ARM]
        best, bd = None, -1
        for i in range(len(big)):
            for j in range(i + 1, len(big)):
                d = float(np.linalg.norm(cents[big[i]] - cents[big[j]]))
                if d > bd:
                    bd, best = d, (big[i], big[j])
        return best, bd

    k_used = K_PRIMARY
    best, bd = pick(k_used)
    if best is None:
        k_used = 4
        best, bd = pick(k_used)
    assert best is not None, "no cluster pair with both sizes >= MIN_ARM"
    lab, _ = labels_by_k[k_used]
    ca, cb = best
    ids_a = [int(eps[i]) for i in np.where(lab == ca)[0]]
    ids_b = [int(eps[i]) for i in np.where(lab == cb)[0]]
    report["arm_pick"] = {"k_used": k_used, "clusters": [int(ca), int(cb)],
                          "centroid_dist": bd,
                          "n_a": len(ids_a), "n_b": len(ids_b)}
    print(f"[cluster] ARMS from k={k_used}: gradarm_a=cluster{ca} (n={len(ids_a)}) "
          f"gradarm_b=cluster{cb} (n={len(ids_b)}) centroid_dist={bd:.3f}", flush=True)

    json.dump({"gradarm_a": ids_a, "gradarm_b": ids_b,
               "k_used": k_used, "clusters": [int(ca), int(cb)],
               "seed": SEED, "top_modes_removed": TOP_MODES,
               "sketch_dirs": DIRS},
              open(f"{GA}/gradarm_assignment.json", "w"))
    json.dump(report, open(f"{GA}/gradarm_report.json", "w"), default=str)
    print(f"[cluster] wrote gradarm_assignment.json + gradarm_report.json", flush=True)


if __name__ == "__main__":
    main()
