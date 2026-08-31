"""Arm-race mirror on CIFAR-100: does the RoboCasa bandit design separate arms
in a domain where the loss PROVABLY encodes the weak region?

Mirrors the bandit_v1 structure exactly, on the xgradtest sandbox:
  base policy  = ResNet18-GN @ step6000 (trained on base set with 20 rare classes
                 at 10% data -- the known "weak region"; mirror of pi_0)
  arm          = a rule for drawing B=200 pool examples:
     null    : nothing added (fine-tune on base only -> the noise floor)
     rare    : 200 rare-class examples (ground-truth helpful; mirror of the tall arm)
     random  : 200 uniform pool (the control arm)
     easy    : 200 common-class examples (mirror of easy_band)
     gradarm_a/b : the two most-separated k-means clusters of the pool's
               whitened unit gradient sketches at step6000 (mirror of the
               gradient-cluster arms; k=6, top-10 modes removed)
  pull         = draw 200 (fresh per pull), fine-tune base+draw from step6000
                 (AdamW lr 3e-4, wd 5e-4, bs 128, 2000 steps), eval on the test
                 set: overall acc, rare-class acc (target stratum), common acc
                 (retention stratum). delta = acc - base ckpt acc.
  rounds       = 4, paired training seeds (1000+r shared across arms in a round).

Per-draw gradient statistics at the base ckpt (from gradlog/cand_raw.npy, the
same JL-sketch machinery as RoboCasa Q0/Q3): mean grad norm, batch coherence
(norm of mean unit sketch), whitened batch separation (top-10 modes removed),
self-similarity (mean pairwise cos), cos to the rare-direction (mean unit rare -
mean unit common, fit on pool examples NOT in the draw).

Outputs xgradtest/armrace/results.json incrementally (resume-safe: completed
pulls are skipped) and prints the final analysis (arm deltas vs null floor,
batch-separation pre-flight, stat-vs-delta correlations).

Run (robocasa env, ~1-1.5h shared-GPU):
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
  /data/xinyua11/conda/envs/robocasa/bin/python /data/xinyua11/xgradtest/xarm_race.py
"""
import json
import os
import sys
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/data/xinyua11/xgradtest")
from xgrad import (ROOT, CKPTS, LOGS, DEV, N_RARE, cifar100, loader,  # noqa: E402
                   make_model, accuracy)

OUT_DIR = f"{ROOT}/armrace"
RES = f"{OUT_DIR}/results.json"
BASE_STEP = 6000
B = 200
ROUNDS = 4
FT_STEPS = 2000
FT_LR = 3e-4
FT_BS = 128
ARMS = ["null", "rare", "random", "easy", "gradarm_a", "gradarm_b"]


def unit(x, ax=-1):
    return x / (np.linalg.norm(x, axis=ax, keepdims=True) + 1e-12)


def load_sketches():
    meta = json.load(open(f"{LOGS}/meta.json"))
    ti = meta["ckpts"].index(BASE_STEP)
    raw = np.asarray(np.load(f"{LOGS}/cand_raw.npy", mmap_mode="r")[ti], dtype=np.float64)
    nrm = np.load(f"{LOGS}/cand_norms.npy")[ti]
    pool = np.array(meta["pool"])
    labels = np.load(f"{LOGS}/pool_labels.npy")
    return raw, nrm, pool, labels


def whiten_basis(U, k=10):
    X = U - U.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:k]


def gradarm_clusters(Uw, seed=0, k=6):
    """k-means on whitened unit sketches; return member masks of the two
    most-separated clusters (mirror of gradient_analysis/gradarm_cluster.py)."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Uw)
    C = km.cluster_centers_
    best, pair = -1.0, (0, 1)
    for i in range(k):
        for j in range(i + 1, k):
            d = float(np.linalg.norm(C[i] - C[j]))
            if d > best:
                best, pair = d, (i, j)
    a, b = pair
    return km.labels_ == a, km.labels_ == b, best


def batch_stats(pool_pos, Uu, Uw, nrm, rare_dir):
    Up = Uu[pool_pos]
    return {
        "mean_gnorm": float(nrm[pool_pos].mean()),
        "coherence": float(np.linalg.norm(Up.mean(0))),
        "wsep": float(np.linalg.norm(unit(Uw[pool_pos]).mean(0))),
        "self_sim": float((Up @ Up.T)[np.triu_indices(len(pool_pos), 1)].mean()),
        "cos_rare_dir": float((Up @ rare_dir).mean()),
    }


def finetune_eval(base_state, tr, te, train_idx, seed, sp):
    torch.manual_seed(seed)
    model = make_model()
    model.load_state_dict(base_state)
    opt = torch.optim.AdamW(model.parameters(), lr=FT_LR, weight_decay=5e-4)
    g = torch.Generator(); g.manual_seed(seed)
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(tr, train_idx),
                                     batch_size=FT_BS, shuffle=True, num_workers=2,
                                     drop_last=True, generator=g)
    step = 0; it = iter(dl)
    model.train()
    while step < FT_STEPS:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(dl); x, y = next(it)
        x, y = x.to(DEV), y.to(DEV)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        step += 1
    rare_classes = set(range(N_RARE))
    common_classes = set(range(N_RARE, 100))
    val = list(range(len(te)))
    return {
        "acc_overall": accuracy(model, te, val),
        "acc_rare": accuracy(model, te, val, classes=rare_classes),
        "acc_common": accuracy(model, te, val, classes=common_classes),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = json.load(open(RES)) if os.path.exists(RES) else {}

    raw, nrm, pool, labels = load_sketches()
    Uu = unit(raw, ax=1)
    P10 = whiten_basis(Uu, 10)
    Uw = Uu - (Uu @ P10.T) @ P10
    is_rare = labels < N_RARE
    ga_mask, gb_mask, sep = gradarm_clusters(unit(Uw), seed=0)
    print(f"[armrace] pool {len(pool)} (rare {int(is_rare.sum())}) | "
          f"gradarm clusters {int(ga_mask.sum())}/{int(gb_mask.sum())} (centroid sep {sep:.3f})", flush=True)

    # rare-direction fit on a fixed half of the pool (draws exclude nothing --
    # scoring uses the OTHER half's direction only when overlap matters; here we
    # fit on even positions and only ever score draws' mean cos: tiny overlap
    # bias is shared equally by all arms)
    fit = np.arange(len(pool)) % 2 == 0
    rare_dir = unit(Uu[fit & is_rare].mean(0) - Uu[fit & ~is_rare].mean(0))

    tr, te = cifar100(True), cifar100(False)
    sp = json.load(open(f"{ROOT}/splits.json"))
    base = sp["base"]
    base_state = torch.load(f"{CKPTS}/step{BASE_STEP}.pt", map_location=DEV)

    # base (pre-fine-tune) reference accs
    if "base_ref" not in results:
        m = make_model(); m.load_state_dict(base_state)
        val = list(range(len(te)))
        results["base_ref"] = {
            "acc_overall": accuracy(m, te, val),
            "acc_rare": accuracy(m, te, val, classes=set(range(N_RARE))),
            "acc_common": accuracy(m, te, val, classes=set(range(N_RARE, 100))),
        }
        json.dump(results, open(RES, "w"), indent=1)
        del m
    ref = results["base_ref"]
    print(f"[armrace] base ref: overall {ref['acc_overall']:.4f} rare {ref['acc_rare']:.4f} "
          f"common {ref['acc_common']:.4f}", flush=True)

    arm_positions = {
        "rare": np.where(is_rare)[0], "random": np.arange(len(pool)),
        "easy": np.where(~is_rare)[0],
        "gradarm_a": np.where(ga_mask)[0], "gradarm_b": np.where(gb_mask)[0],
    }

    for r in range(ROUNDS):
        train_seed = 1000 + r
        for arm in ARMS:
            pid = f"{arm}_r{r}"
            if pid in results:
                print(f"[armrace] {pid}: done, skip", flush=True)
                continue
            t0 = time.time()
            if arm == "null":
                draw_pos = np.array([], dtype=int)
                train_idx = list(base)
            else:
                rng = np.random.RandomState(zlib.crc32(f"{arm}_{r}".encode()) % (2**31))
                draw_pos = rng.choice(arm_positions[arm], size=B, replace=False)
                train_idx = list(base) + [int(pool[p]) for p in draw_pos]
            accs = finetune_eval(base_state, tr, te, train_idx, train_seed, sp)
            row = {"arm": arm, "round": r, "seed": train_seed,
                   "delta_overall": accs["acc_overall"] - ref["acc_overall"],
                   "delta_rare": accs["acc_rare"] - ref["acc_rare"],
                   "delta_common": accs["acc_common"] - ref["acc_common"],
                   **accs}
            if len(draw_pos):
                row["stats"] = batch_stats(draw_pos, Uu, Uw, nrm, rare_dir)
                row["draw_rare_frac"] = float(is_rare[draw_pos].mean())
                row["draw_pos"] = [int(p) for p in draw_pos]
            results[pid] = row
            json.dump(results, open(RES, "w"), indent=1)
            print(f"[armrace] {pid}: d_overall {row['delta_overall']:+.4f} "
                  f"d_rare {row['delta_rare']:+.4f} d_common {row['delta_common']:+.4f} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)

    # ---------------- analysis ----------------
    print("\n================ ANALYSIS ================", flush=True)
    pulls = [v for k, v in results.items() if k != "base_ref"]
    print(f"{'arm':>10s}  {'d_rare (4 rounds)':>28s}  {'mean':>7s} | {'d_common':>8s} | {'d_overall':>9s}")
    for arm in ARMS:
        rows = sorted([p for p in pulls if p["arm"] == arm], key=lambda x: x["round"])
        dr = [p["delta_rare"] for p in rows]
        print(f"{arm:>10s}  {' '.join(f'{d:+.3f}' for d in dr):>28s}  {np.mean(dr):+.4f} | "
              f"{np.mean([p['delta_common'] for p in rows]):+.4f} | "
              f"{np.mean([p['delta_overall'] for p in rows]):+.4f}")

    # batch-separation pre-flight mirror: arm-draw vs random-draw whitened batch-mean distance
    rng = np.random.RandomState(3)
    def wmean(pos):
        return unit(Uw[pos]).mean(0)
    null_d = [np.linalg.norm(wmean(rng.choice(len(pool), B, False)) -
                             wmean(rng.choice(len(pool), B, False))) for _ in range(20)]
    print(f"\nbatch-mean whitened separation: random-vs-random null = {np.mean(null_d):.4f}")
    for arm in ["rare", "easy", "gradarm_a", "gradarm_b"]:
        rng2 = np.random.RandomState(11)
        d = []
        for r in range(ROUNDS):
            pos = results.get(f"{arm}_r{r}", {}).get("draw_pos")
            if pos is None:
                continue
            d.append(np.linalg.norm(wmean(np.array(pos)) - wmean(rng2.choice(len(pool), B, False))))
        if d:
            print(f"  {arm:>10s} vs random: {np.mean(d):.4f}  ({np.mean(d)/np.mean(null_d):.1f}x null)")

    # stat-vs-delta correlations across all non-null pulls
    wp = [p for p in pulls if "stats" in p]
    def spear(a, b):
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])
    print(f"\nstat vs delta correlations over {len(wp)} non-null pulls:")
    for stat in ["mean_gnorm", "coherence", "wsep", "self_sim", "cos_rare_dir"]:
        v = np.array([p["stats"][stat] for p in wp])
        for tgt in ["delta_rare", "delta_overall"]:
            d = np.array([p[tgt] for p in wp])
            print(f"  {stat:>13s} vs {tgt:>13s}: spearman {spear(v, d):+.3f}")
    print("[armrace] DONE", flush=True)


if __name__ == "__main__":
    main()
