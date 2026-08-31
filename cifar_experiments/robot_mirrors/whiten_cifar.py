"""Whitening k-sweep on the CIFAR gradlog (sanity / generalization check for the robocasa whitening
finding). Reuses the exact whitening primitives from influence_offline.py so the math is identical.

Target = unit(gfail[ti]); helpful = pool_labels < n_rare (rare classes). For each checkpoint index ti
we project out the top-k SHARED SVD modes of the centered unit-candidate cloud from BOTH the candidate
grads and gfail, then cosine. We also report the random-direction control (project out k random
orthonormal dirs) and the variance fraction carried by the removed modes.

  python cifar_experiments/robot_mirrors/whiten_cifar.py --ti 2,3 --ks 0,1,2,3,5,8,12,20,30,50,80
"""
import argparse, json, os, sys, numpy as np
sys.path.insert(0, "/data/xinyua11/robocasa/policy_analysis")
from influence_offline import unit, auc, whiten_basis, whitened_score

GR = "/data/xinyua11/xgradtest/gradlog"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=GR)
    ap.add_argument("--ti", default="2,3", help="checkpoint indices into cand_raw[0] axis")
    ap.add_argument("--ks", default="0,1,2,3,5,8,12,20,30,50,80")
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.dir, "meta.json")))
    cand = np.load(os.path.join(a.dir, "cand_raw.npy")).astype(np.float32)   # [T,n,d]
    gfail = np.load(os.path.join(a.dir, "gfail.npy")).astype(np.float32)     # [T,d]
    labels = np.load(os.path.join(a.dir, "pool_labels.npy"))
    n_rare = meta["n_rare"]; ckpts = meta["ckpts"]
    hot = labels < n_rare
    ks = [int(x) for x in a.ks.split(",")]
    print(f"[cifar] cand {cand.shape} | rare(<{n_rare}) {int(hot.sum())}/{len(hot)} | ckpts {ckpts}")

    for ti in [int(x) for x in a.ti.split(",")]:
        Zu = unit(cand[ti])                  # [n,d] unit candidate grads at this ckpt
        tgt = unit(gfail[ti])
        mu, Vt, sv2frac = whiten_basis(Zu)
        base = auc(Zu @ tgt, hot)
        print(f"\n=== ckpt idx {ti} (step {ckpts[ti]}) | baseline(k=0) AUC={base:.4f} ===")
        print(f"{'k':>4s} {'AUC(rare>common)':>17s} {'lift_vs_k0':>11s} {'var%_topk':>10s} {'AUC_randk':>10s}")
        rng = np.random.RandomState(0)
        for k in ks:
            s = whitened_score(Zu, tgt, Vt, k)
            a_w = auc(s, hot)
            var_topk = sv2frac[:k].sum() if k > 0 else 0.0
            if k > 0:
                G = rng.randn(k, Zu.shape[1]).astype(np.float32)
                Qr, _ = np.linalg.qr(G.T); Vr = Qr.T
                a_r = auc(whitened_score(Zu, tgt, Vr, k), hot)
            else:
                a_r = a_w
            print(f"{k:>4d} {a_w:>17.4f} {a_w-base:>+11.4f} {var_topk*100:>9.1f}% {a_r:>10.4f}")


if __name__ == "__main__":
    main()
