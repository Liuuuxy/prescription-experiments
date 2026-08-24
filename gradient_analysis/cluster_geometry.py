"""Do gradient clusters mean anything? CIFAR vs RoboCasa, same recipe, same metrics.

Recipe (identical to the robot gradarm construction): unit-normalize per-demo
gradient sketches -> remove the top-10 shared SVD modes ("whitening") -> re-unit
-> k-means k=6.  Then, per domain:
  GEOMETRY : silhouette (sampled), inter-centroid distances, within-cluster
             dispersion, between/within ratio, nearest-centroid margin (overlap)
  MEANING  : cluster composition vs the labels each domain has --
             CIFAR: 100-class labels + rare(20)/common split (ARI/NMI, rare
             fraction per cluster, top classes)
             RoboCasa: region (tall/mid/easy) ARI + per-cluster region mix,
             per-cluster mean TRUE grad-norm, and enrichment of the 875
             style_hi/style_lo (quality-labelled) demos.

Run: /data/xinyua11/conda/envs/robocasa/bin/python gradient_analysis/cluster_geometry.py
Writes gradient_analysis/cluster_geometry.json
"""
import json
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

sys.path.insert(0, "/data/xinyua11/robocasa")
GA = "/data/xinyua11/robocasa/gradient_analysis"
K = 6
TOPMODES = 10
SEED = 0


def unit(x, ax=-1):
    return x / (np.linalg.norm(x, axis=ax, keepdims=True) + 1e-12)


def whiten_unit(X):
    U = unit(X.astype(np.float64))
    Xc = U - U.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Vt[:TOPMODES]
    W = U - (U @ P.T) @ P
    return unit(W)


def geometry(W, lab_km, centroids):
    d_to_c = np.linalg.norm(W - centroids[lab_km], axis=1)
    within = np.array([d_to_c[lab_km == c].mean() for c in range(K)])
    D = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
    tri = D[np.triu_indices(K, 1)]
    # nearest-centroid margin: (d2 - d1) / d1  -- small = points sit between clusters
    dall = np.linalg.norm(W[:, None, :] - centroids[None, :, :], axis=2)
    dall.sort(axis=1)
    margin = (dall[:, 1] - dall[:, 0]) / (dall[:, 0] + 1e-12)
    rng = np.random.RandomState(0)
    idx = rng.choice(len(W), size=min(2000, len(W)), replace=False)
    sil = float(silhouette_score(W[idx], lab_km[idx]))
    return {
        "silhouette_sampled2000": round(sil, 4),
        "inter_centroid_dist": {"mean": round(float(tri.mean()), 4),
                                "min": round(float(tri.min()), 4),
                                "max": round(float(tri.max()), 4)},
        "within_dispersion_mean": round(float(within.mean()), 4),
        "between_over_within": round(float(tri.mean() / within.mean()), 4),
        "nearest_margin_median": round(float(np.median(margin)), 4),
        "frac_margin_lt_5pct": round(float((margin < 0.05).mean()), 4),
        "cluster_sizes": np.bincount(lab_km, minlength=K).tolist(),
    }


out = {}

# ---------------- CIFAR ----------------
G = "/data/xinyua11/xgradtest/gradlog"
meta = json.load(open(f"{G}/meta.json"))
ti = meta["ckpts"].index(6000)
Xc = np.asarray(np.load(f"{G}/cand_raw.npy", mmap_mode="r")[ti], dtype=np.float64)
yc = np.load(f"{G}/pool_labels.npy")
rare = yc < meta["n_rare"]
Wc = whiten_unit(Xc)
kmc = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(Wc)
lc = kmc.labels_
gc = geometry(Wc, lc, kmc.cluster_centers_)
comp = []
for c in range(K):
    m = lc == c
    cls, cnt = np.unique(yc[m], return_counts=True)
    top = sorted(zip(cnt, cls), reverse=True)[:3]
    comp.append({"n": int(m.sum()),
                 "rare_frac": round(float(rare[m].mean()), 3),
                 "top_classes": [[int(k), int(n)] for n, k in top],
                 "top1_purity": round(float(top[0][0] / m.sum()), 3)})
cifar = {
    "geometry": gc,
    "ARI_vs_100class": round(float(adjusted_rand_score(yc, lc)), 4),
    "NMI_vs_100class": round(float(normalized_mutual_info_score(yc, lc)), 4),
    "ARI_vs_rare_binary": round(float(adjusted_rand_score(rare, lc)), 4),
    "pool_rare_frac": round(float(rare.mean()), 3),
    "clusters": comp,
}
out["cifar"] = cifar

# ---------------- RoboCasa ----------------
eps, Ss, Ns = [], [], []
for d in (f"{GA}/sketches_pi0base_19999", f"{GA}/sketches_pi0base_19999_pool"):
    m = json.load(open(f"{d}/episodes.json"))
    eps += list(m["episodes"])
    Ss.append(np.load(f"{d}/sketches.npy"))
    Ns.append(np.load(f"{d}/norms.npy"))
S = np.vstack(Ss); N = np.concatenate(Ns)
lists = json.load(open(f"{GA}/demo_lists.json"))
regmap = lists["region_assignment"]              # W demos only (9,485)
keep = [i for i, e in enumerate(eps) if str(e) in regmap]
Sr = S[keep]; Nr = N[keep]
er = [eps[i] for i in keep]
yreg = np.array([regmap[str(e)] for e in er])
Wr = whiten_unit(Sr)
kmr = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(Wr)
lr = kmr.labels_
gr = geometry(Wr, lr, kmr.cluster_centers_)
jobs = json.load(open(f"{GA}/theory_jobs.json"))
hi = set(e for p in ("style_hi_j3", "style_hi_j4", "style_hi_j5") for e in jobs[p]["demo_ids"])
lo = set(e for p in ("style_lo_j3", "style_lo_j4", "style_lo_j5") for e in jobs[p]["demo_ids"])
compr = []
regions = ["tall_vessel_grasp_fail", "mid_band", "easy_band"]
for c in range(K):
    m = lr == c
    mix = {r: round(float((yreg[m] == r).mean()), 3) for r in regions}
    inhi = sum(1 for i in np.where(m)[0] if er[i] in hi)
    inlo = sum(1 for i in np.where(m)[0] if er[i] in lo)
    compr.append({"n": int(m.sum()), "region_mix": mix,
                  "mean_gnorm": round(float(Nr[m].mean()), 3),
                  "n_style_hi": inhi, "n_style_lo": inlo})
robot = {
    "geometry": gr,
    "ARI_vs_region": round(float(adjusted_rand_score(yreg, lr)), 4),
    "NMI_vs_region": round(float(normalized_mutual_info_score(yreg, lr)), 4),
    "pool_region_mix": {r: round(float((yreg == r).mean()), 3) for r in regions},
    "pool_style_counts": {"hi": len(hi), "lo": len(lo)},
    "clusters": compr,
}
out["robocasa"] = robot

json.dump(out, open(f"{GA}/cluster_geometry.json", "w"), indent=1)

print("=== GEOMETRY (same recipe both domains: unit -> whiten top-10 -> k-means k=6) ===")
for dom in ("cifar", "robocasa"):
    g = out[dom]["geometry"]
    print(f"{dom:9s} silhouette {g['silhouette_sampled2000']:+.3f}  "
          f"between/within {g['between_over_within']:.3f}  "
          f"median NN-margin {g['nearest_margin_median']:.3f}  "
          f"frac margin<5% {g['frac_margin_lt_5pct']:.2f}  sizes {g['cluster_sizes']}")
print("\n=== MEANING: CIFAR ===")
print(f"ARI vs 100 classes {cifar['ARI_vs_100class']}  NMI {cifar['NMI_vs_100class']}  "
      f"ARI vs rare/common {cifar['ARI_vs_rare_binary']}  (pool rare frac {cifar['pool_rare_frac']})")
for i, c in enumerate(cifar["clusters"]):
    print(f"  c{i}: n={c['n']:4d} rare_frac={c['rare_frac']:.3f} top1_purity={c['top1_purity']:.3f} "
          f"top_classes={c['top_classes']}")
print("\n=== MEANING: RoboCasa ===")
print(f"ARI vs region {robot['ARI_vs_region']}  NMI {robot['NMI_vs_region']}  "
      f"(pool mix {robot['pool_region_mix']})")
for i, c in enumerate(robot["clusters"]):
    print(f"  c{i}: n={c['n']:4d} region={c['region_mix']} |g|={c['mean_gnorm']:.2f} "
          f"style_hi={c['n_style_hi']} style_lo={c['n_style_lo']}")
