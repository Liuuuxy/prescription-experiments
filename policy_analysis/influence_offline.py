"""Offline influence-score menu: load saved gradient SKETCHES once, try many scoring variants in
seconds (no GPU, no re-extraction). For each variant, report the hard-vs-easy separation AUC and the
top-200 targeted fraction, so we can see which scoring choice actually carries signal.

Works on two sketch formats in --dir:
  * unit-mean  : {dval,ret,cand}_sketch.npy  [n, d]   (direction-only; magnitude already dropped)
  * raw per-ckpt (preferred, from `influence_score.py sketch --raw`):
                 {dval,ret,cand}_raw.npy [T, n, d] + {dval,ret,cand}_norms.npy [T, n]
    -> enables MAGNITUDE-AWARE variants (raw dot, projection) and per-checkpoint weighting.

  python policy_analysis/influence_offline.py --dir weakregion/rcless
"""
import argparse
import json
import os
import numpy as np

TARGETED = set(["juice", "spray", "pitcher", "canned_food", "soap_dispenser",
                "tupperware", "cheese_grater", "ice_cube", "cream_cheese_stick", "jar"])


def unit(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12)


def load(d, tag):
    """Return (vecs [n,d] mean-over-ckpt direction, norms [n] or None, raw [T,n,d] or None, meta)."""
    meta = json.load(open(os.path.join(d, f"{tag}_meta.json")))
    raw_p = os.path.join(d, f"{tag}_raw.npy")
    if os.path.exists(raw_p):
        raw = np.load(raw_p).astype(np.float32)               # [T,n,d] raw, per-ckpt
        norms = np.load(os.path.join(d, f"{tag}_norms.npy")).astype(np.float32)  # [T,n]
        vec = unit(raw, axis=2).mean(0)                        # mean of per-ckpt unit dirs
        return vec, norms.mean(0), raw, meta
    return np.load(os.path.join(d, f"{tag}_sketch.npy")).astype(np.float32), None, None, meta


def kmeans(X, k, iters=40, seed=0):
    rng = np.random.RandomState(seed); C = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        a = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        C = np.stack([X[a == j].mean(0) if (a == j).any() else C[j] for j in range(k)])
    return unit(C)


def auc(scores, is_hard):
    h = scores[is_hard]; e = scores[~is_hard]
    if len(h) == 0 or len(e) == 0:
        return float("nan")
    # P(hard > easy) via rank-sum (fast, exact ties=0.5)
    order = np.argsort(scores); ranks = np.empty(len(scores)); ranks[order] = np.arange(len(scores))
    return (ranks[is_hard].sum() - len(h) * (len(h) - 1) / 2) / (len(h) * len(e))


def whiten_basis(Zu):
    """SVD the CENTERED unit-candidate cloud once; return (mean, Vt, sv) where Vt[:k] are the
    top-k SHARED modes (right-singular vectors, by descending singular value). Project these out
    of both candidates and the target to isolate the discriminative subspace.
    sv2_frac[k] = fraction of total variance carried by mode k (for the noise-dim sanity check)."""
    mu = Zu.mean(0)
    Xc = Zu - mu                                   # center; shared mean is itself a mode we keep
    # economy SVD of [n,d]: rows = candidates, Vt rows = directions in feature space
    _, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
    sv2 = sv ** 2
    return mu, Vt, sv2 / (sv2.sum() + 1e-12)


def project_out(X, V):
    """Remove the subspace spanned by rows of V (orthonormal) from each row of X. V: [k,d]."""
    if V is None or len(V) == 0:
        return X
    return X - (X @ V.T) @ V


def whitened_score(Zu, target, Vt, k):
    """Project out top-k shared modes from BOTH candidates and target, then cosine."""
    V = Vt[:k] if k > 0 else None
    Zw = unit(project_out(Zu, V))
    tw = unit(project_out(target[None], V))[0]
    return Zw @ tw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="weakregion/rcless")
    ap.add_argument("--topk_targets", type=int, default=5)
    ap.add_argument("--m", type=int, default=14)
    ap.add_argument("--hard", default=None,
                    help="override hard categories: a json with 'new_targeted' (e.g. weakregion/targeted_rebalanced.json) or a comma list")
    ap.add_argument("--whiten_k", type=int, default=30,
                    help="default k for the whitening variant shown in the menu (project out top-k shared SVD modes)")
    ap.add_argument("--whiten_sweep", default=None,
                    help="comma list of k values; if set, run ONLY the whitening k-sweep (+random-direction control) and exit")
    a = ap.parse_args()
    global TARGETED
    if a.hard:
        if a.hard.endswith(".json"):
            TARGETED = set(json.load(open(a.hard))["new_targeted"])
        else:
            TARGETED = set(c.strip() for c in a.hard.split(","))
        print(f"[off] using hard set: {sorted(TARGETED)}")

    Dv, Dn, Dr, Dm = load(a.dir, "dval")
    Rv, Rn, Rr, Rm = load(a.dir, "ret")
    Zv, Zn, Zr, Zm = load(a.dir, "cand")
    cats = np.array(Zm["categories"])
    is_hard = np.array([c in TARGETED for c in cats])
    dval_hard = np.array([c in TARGETED for c in Dm["categories"]])
    has_raw = Dr is not None
    print(f"[off] cand {Zv.shape} (hard {is_hard.sum()}/{len(is_hard)}) | dval {Dv.shape} | raw={has_raw}")

    gD = unit(Dv.mean(0)); gR = unit(Rv.mean(0))
    gHard = unit(Dv[dval_hard].mean(0)) if dval_hard.any() else gD
    Zu = unit(Zv)

    # ---- WHITENING k-sweep mode (generalized contrast): remove top-k SHARED candidate SVD modes ----
    if a.whiten_sweep is not None:
        ks = [int(x) for x in a.whiten_sweep.split(",")]
        mu, Vt, sv2frac = whiten_basis(Zu)
        base = Zu @ gHard
        print(f"[whiten] cand {Zu.shape} hard {is_hard.sum()}/{len(is_hard)} | target=g_hard | baseline(k=0) AUC={auc(base,is_hard):.3f}")
        print(f"\n{'k':>4s} {'AUC(hard>easy)':>15s} {'lift_vs_k0':>11s} {'top200 tgt%':>12s} {'var%_topk':>10s} {'AUC_randk':>10s}")
        rng = np.random.RandomState(0)
        for k in ks:
            s = whitened_score(Zu, gHard, Vt, k)
            top = np.argsort(-s)[:200]
            a_w = auc(s, is_hard); tfrac = is_hard[top].mean()
            var_topk = sv2frac[:k].sum() if k > 0 else 0.0
            # random-direction control: project out k RANDOM orthonormal dirs (same k, same cosine pipeline)
            if k > 0:
                G = rng.randn(k, Zu.shape[1]).astype(np.float32)
                Qr, _ = np.linalg.qr(G.T)        # [d,k] orthonormal columns
                Vr = Qr.T                         # [k,d]
                a_r = auc(whitened_score(Zu, gHard, Vr, k), is_hard)
            else:
                a_r = a_w
            print(f"{k:>4d} {a_w:>15.3f} {a_w-auc(base,is_hard):>+11.3f} {tfrac*100:>11.1f}% {var_topk*100:>9.1f}% {a_r:>10.3f}")
        return
    C = kmeans(unit(Dv), a.m)                          # gradient-mode centroids (all D_val)
    Chard = kmeans(unit(Dv[dval_hard]), min(a.m, max(2, dval_hard.sum() // 3))) if dval_hard.sum() > 6 else None

    variants = {}
    variants["plain  cos(z, mean Dval)"] = Zu @ gD
    variants["contrast cos(z, hard-ret)"] = Zu @ unit(gHard - gR)
    variants["contrast cos(z, Dval-ret)"] = Zu @ unit(gD - gR)
    variants["max-over-targets"] = (Zu @ unit(Dv).T).max(1)
    variants[f"top{a.topk_targets} targets (mean)"] = np.sort(Zu @ unit(Dv).T, 1)[:, -a.topk_targets:].mean(1)
    variants["coverage top3 (all modes)"] = np.sort(Zu @ C.T, 1)[:, -3:].mean(1)
    if Chard is not None:
        variants["coverage top3 (HARD modes)"] = np.sort(Zu @ Chard.T, 1)[:, -3:].mean(1)
    variants["cos(z, g_hard) only"] = Zu @ gHard
    _mu, _Vt, _ = whiten_basis(Zu)
    variants[f"WHITEN k={a.whiten_k} (g_hard)"] = whitened_score(Zu, gHard, _Vt, a.whiten_k)
    if has_raw:
        gD_raw = Dr.mean(1).mean(0)                    # raw mean gradient (magnitude kept)
        gHard_raw = Dr[:, dval_hard].mean(1).mean(0) if dval_hard.any() else gD_raw
        variants["MAG raw-dot  <z, mean Dval>"] = (Zr.mean(0)) @ unit(gD_raw)         # projection (mag-aware)
        variants["MAG projection on g_hard"] = (Zr.mean(0)) @ unit(gHard_raw)
        variants["grad-norm only (||z||)"] = Zn

    print(f"\n{'variant':34s} {'AUC(hard>easy)':>15s} {'top200 targeted%':>17s}")
    for name, s in variants.items():
        s = np.asarray(s, dtype=np.float64)
        top = np.argsort(-s)[:200]
        tfrac = is_hard[top].mean()
        print(f"  {name:32s} {auc(s, is_hard):>15.3f} {tfrac*100:>16.1f}%")
    print("\nAUC 0.5 = no separation; >0.6 = real signal. (hard=targeted-10 cand, easy=non-targeted)")
    print("targeted% = fraction of the top-200 in the failure categories.")


if __name__ == "__main__":
    main()
