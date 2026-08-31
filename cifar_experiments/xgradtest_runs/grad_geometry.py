"""Gradient-geometry analysis: WHY does influence work (CIFAR) or not (robocasa)?

Decomposes the per-candidate gradient cloud into singular modes to quantify:
  1. EFFECTIVE RANK -- is the failure signal a distinct direction (high rank) or a common mode
     (one direction dominates)? This is a cheap "will influence work here?" predictor.
  2. WHICH mode carries the discriminative signal -- AUC of each singular-mode coordinate vs the
     ground-truth helpful set. If it's a TOP mode -> easy to find; a TAIL mode -> buried.
  3. WHITENING sweep -- remove the top-k shared modes (generalized contrast), recompute the
     cos-to-g_fail AUC. Does isolating the discriminative subspace LIFT separation? (a method fix.)
  4. cos(g_val, g_fail) collinearity.

  python grad_geometry.py            # CIFAR gradlog (raw + ground-truth labels)
"""
import json
import numpy as np

LOGS = "/data/xinyua11/xgradtest/gradlog"


def unit(x, ax=-1):
    return x / (np.linalg.norm(x, axis=ax, keepdims=True) + 1e-12)


def auc(score, hot):
    o = np.argsort(score); r = np.empty(len(score)); r[o] = np.arange(len(score))
    nh = hot.sum()
    return (r[hot].sum() - nh * (nh - 1) / 2) / (nh * (~hot).sum())


def eff_rank(S):
    p = S ** 2 / (S ** 2).sum()
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))   # entropy participation ratio


def main():
    m = json.load(open(f"{LOGS}/meta.json"))
    raw = np.load(f"{LOGS}/cand_raw.npy")        # [T,n,d]
    gfail = np.load(f"{LOGS}/gfail.npy"); gval = np.load(f"{LOGS}/gval.npy")
    y = np.load(f"{LOGS}/pool_labels.npy")
    hot = y < m["n_rare"]                          # ground-truth helpful (rare-class)
    ck = m["ckpts"]; n, d = raw.shape[1], raw.shape[2]
    print(f"[geom] CIFAR  pool {n} (helpful {hot.sum()})  dim {d}  ckpts {ck}\n")

    for ti, c in enumerate(ck):
        Z = unit(raw[ti])                          # candidate directions
        gf = unit(gfail[ti])
        U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)   # modes of the cloud (centered)
        var = S ** 2 / (S ** 2).sum()
        er = eff_rank(S)
        base_auc = auc(Z @ gf, hot)

        # which singular mode separates helpful vs common? (AUC of each mode coordinate)
        coords = (Z - Z.mean(0)) @ Vt.T            # [n, d] projections onto modes
        mode_auc = np.array([max(auc(coords[:, j], hot), 1 - auc(coords[:, j], hot)) for j in range(min(40, d))])
        best_mode = int(mode_auc.argmax())

        # whitening: remove top-k shared modes from Z and gf, recompute cos-AUC
        def whiten_auc(k):
            if k == 0:
                return base_auc
            P = Vt[:k]                              # top-k right singular dirs (shared/common modes)
            Zw = Z - (Z @ P.T) @ P
            gw = gf - (gf @ P.T) @ P
            return auc(unit(Zw) @ unit(gw), hot)

        print(f"=== checkpoint {c} ===")
        print(f"  effective rank {er:.1f}/{d}  | var in top-1 {var[0]*100:.0f}%  top-5 {var[:5].sum()*100:.0f}%  top-20 {var[:20].sum()*100:.0f}%")
        print(f"  cos(g_val,g_fail) {float(unit(gval[ti])@gf):+.3f}")
        print(f"  raw cos-to-g_fail AUC {base_auc:.3f}  | discriminative signal lives in MODE #{best_mode} (its AUC {mode_auc[best_mode]:.3f})")
        print(f"  whitening (remove top-k modes) AUC:  " +
              "  ".join(f"k={k}:{whiten_auc(k):.3f}" for k in [0, 1, 3, 10]))
        print()
    print("READS: low effective rank + signal in a TAIL mode + whitening LIFTS AUC => common-mode (influence hard).")
    print("       high effective rank + signal already in cos-to-g_fail + whitening neutral => directionally distinct (influence easy).")


if __name__ == "__main__":
    main()
