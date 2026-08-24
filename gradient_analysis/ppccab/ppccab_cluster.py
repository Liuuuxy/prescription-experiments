"""Task-2 gradient-cluster arms (pre-registration 2026-08-18).

Merge the two sketch shards, apply gradarm_cluster's exact preprocessing
(unit-norm -> remove top-10 SVD modes -> renorm), k-means k=6 seed 20260801,
freeze gc0..gc5 episode lists + 800 planted candidates (rng 777, from the
gated pool) to gradient_analysis/ppccab/ucb_robot/arms_r3.json. Diagnostics
to cluster_report.json. CPU-only, robocasa env, BANDIT_TASK_PROFILE=ppccab.
"""
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "/data/xinyua11/robocasa")
sys.path.insert(0, "/data/xinyua11/robocasa/gradient_analysis")
assert os.environ.get("BANDIT_TASK_PROFILE") == "ppccab"

from gradarm_cluster import whiten  # noqa: E402  (top-10-mode removal, validated recipe)

GA = "/data/xinyua11/robocasa/gradient_analysis/ppccab"
SHARDS = [f"{GA}/sketches_ppccabbase_9999_shard0of2", f"{GA}/sketches_ppccabbase_9999_shard1of2"]
OUT = f"{GA}/ucb_robot"
K = 6
SEED = 20260801


def main():
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    os.makedirs(OUT, exist_ok=True)
    eps, rows = [], []
    for d in SHARDS:
        meta = json.load(open(f"{d}/episodes.json"))
        S = np.load(f"{d}/sketches.npy")
        assert len(meta["episodes"]) == len(S), d
        assert meta["proj_seed"] == 12345 and meta["ckpt_step"] == 9999
        eps += [int(e) for e in meta["episodes"]]
        rows.append(S)
    X = np.concatenate(rows).astype(np.float64)
    order = np.argsort(eps)
    eps = [eps[i] for i in order]; X = X[order]
    print(f"[cluster] merged {len(eps)} sketches, dim {X.shape[1]}", flush=True)

    Xw, ev_removed = whiten(X)
    km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(Xw)
    sil = float(silhouette_score(Xw, km.labels_, sample_size=3000, random_state=0))
    sizes = Counter(int(l) for l in km.labels_)
    print(f"[cluster] k={K} sizes={dict(sorted(sizes.items()))} sil={sil:.3f} "
          f"top10-mode-energy-removed={ev_removed:.3f}", flush=True)

    arms = {f"gc{c}": sorted(int(eps[i]) for i in np.where(km.labels_ == c)[0])
            for c in range(K)}
    rng777 = np.random.default_rng(777)
    arms["planted_bad"] = sorted(int(x) for x in rng777.choice(eps, 800, replace=False))
    json.dump(arms, open(f"{OUT}/arms_r3.json", "w"))

    # composition diagnostics vs pool features
    from bandit_v1 import pool
    W = pool.build_pool_table(write=False).set_index("episode_index")
    comp = {}
    for c in range(K):
        sub = W.loc[[e for e in arms[f"gc{c}"] if e in W.index]]
        comp[f"gc{c}"] = {"n": len(sub), "mean_h": round(float(sub.h.mean()), 3),
                          "top_cats": dict(sub.category.value_counts().head(5)),
                          "mean_traj_len": round(float(sub.traj_len.mean()), 1)}
        print(f"[cluster] gc{c}: {comp[f'gc{c}']}", flush=True)
    json.dump({"sizes": {f"gc{c}": sizes[c] for c in range(K)}, "silhouette": sil,
               "ev_removed": ev_removed, "seed": SEED, "k": K, "composition": comp},
              open(f"{OUT}/cluster_report.json", "w"), default=int)
    small = [f"gc{c}" for c in range(K) if sizes[c] < 1200]
    if small:
        print(f"[cluster] NOTE: arms under task-1's 1200-demo diversity floor: {small}", flush=True)
    print("PPCCAB CLUSTER COMPLETE", flush=True)


if __name__ == "__main__":
    main()
