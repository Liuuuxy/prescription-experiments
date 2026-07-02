"""RC-LESS stage 2: Retention-Constrained, Coverage-aware selection from saved gradient sketches.

Consumes weakregion/rcless/{dval,ret,cand}_sketch.npy (+ *_meta.json) produced by
`influence_score.py sketch`. Implements the theory roadmap's recommended selector:

  * CLUSTER the D_val target tugs into m gradient modes (k-means in sketch space), after
    CONTRAST-CENTERING by g_R = mean(retention sketches) so the common-mode "generic pick-place"
    direction is removed (fixes F1 / the user's "don't average heterogeneous targets" point).
  * COVERAGE score: cov(z) = mean of z's top-3 cosines to the m centroids -> rewards demos that
    strongly help SOME specific mode, not the blurry average; greedy facility-location picks a SET
    that covers all modes (anti-redundancy, fixes F5).
  * RETENTION penalty (corrected sign): demote demos TOO PARALLEL to g_R (the shared-grasp/forgetting
    direction) -> score(z) = cov(z) - lambda * max(0, <z,g_R> - rho)  (fixes F3).
  * PER-CATEGORY FLOOR on the failing (targeted-10) categories -> re-implements core's winning
    depth-on-holes mechanism (so RC-LESS is >= core by construction); + per-category CAP (no 14 tongs).

  python policy_analysis/rc_select.py --dir weakregion/rcless --mg_dir <MG> \
      --n_select 200 --m 14 --floor 12 --cap 10 --lam 1.0 --out weakregion/rc_core.json
"""
import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np

TARGETED = ["juice", "spray", "pitcher", "canned_food", "soap_dispenser",
            "tupperware", "cheese_grater", "ice_cube", "cream_cheese_stick", "jar"]


def kmeans(X, k, iters=50, seed=0):
    rng = np.random.RandomState(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        a = d.argmin(1)
        newC = np.stack([X[a == j].mean(0) if (a == j).any() else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return C, a


def unit(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="weakregion/rcless")
    ap.add_argument("--mg_dir", required=True)
    ap.add_argument("--n_select", type=int, default=200)
    ap.add_argument("--m", type=int, default=14, help="number of gradient-mode clusters")
    ap.add_argument("--floor", type=int, default=12, help="min demos per failing (targeted) category")
    ap.add_argument("--cap", type=int, default=10, help="max demos per category (anti-redundancy)")
    ap.add_argument("--lam", type=float, default=1.0, help="retention penalty weight")
    ap.add_argument("--rho", type=float, default=0.0, help="retention parallelism budget")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_center", action="store_true",
                    help="FIX: do NOT contrast-center coverage modes by g_R. g_R is ~78%% the task "
                         "gradient, so centering (and the lam penalty) steer selection AWAY from grasp.")
    ap.add_argument("--out", default="weakregion/rc_core.json")
    a = ap.parse_args()

    D = np.load(os.path.join(a.dir, "dval_sketch.npy"))
    R = np.load(os.path.join(a.dir, "ret_sketch.npy"))
    Z = np.load(os.path.join(a.dir, "cand_sketch.npy"))
    cand_eps = json.load(open(os.path.join(a.dir, "cand_meta.json")))["episodes"]
    cand_cat = json.load(open(os.path.join(a.dir, "cand_meta.json")))["categories"]
    print(f"[rc] dval {D.shape} ret {R.shape} cand {Z.shape} | lam={a.lam} floor={a.floor} no_center={a.no_center}")

    gR = unit(R.mean(0))                       # retention direction (~0.78 collinear with the task grad!)
    gD = unit(D.mean(0))                       # overall task/improvement direction
    if a.no_center:                            # FIXED: cluster raw targets (coverage ~ task-alignment)
        Dc, Zc = D, Z
    else:                                      # ORIGINAL (buggy): center by g_R -> away from the task
        Dc = D - (D @ gR)[:, None] * gR
        Zc = Z - (Z @ gR)[:, None] * gR
    C, _ = kmeans(unit(Dc), a.m, seed=a.seed)  # m gradient-mode centroids
    C = unit(C)

    Cov = unit(Zc) @ C.T                       # [n_cand, m] cosine of each candidate to each mode
    ret_par = Z @ gR                           # alignment with the retention direction
    top3 = np.sort(Cov, axis=1)[:, -3:].mean(1)
    score = top3 - a.lam * np.maximum(0.0, ret_par - a.rho)  # lam=0 disables the (inverted) penalty
    # sanity diagnostics vs the pool
    csel = unit(Z) @ gD
    print(f"[rc] pool: mean cos(demo,g_D)={csel.mean():.3f}  mean <z,g_R>={ret_par.mean():.3f}")

    cat_count = defaultdict(int)
    selected, best = [], np.full(a.m, -1e9)

    def take(i):
        selected.append(i); cat_count[cand_cat[i]] += 1
        np.maximum(best, Cov[i], out=best)

    # 1) per-category FLOOR on the failing categories (depth-on-holes, core's mechanism)
    order = np.argsort(-score)
    for cat in TARGETED:
        got = 0
        for i in order:
            if got >= a.floor or len(selected) >= a.n_select:
                break
            if cand_cat[i] == cat and i not in selected:
                take(i); got += 1

    # 2) fill remaining by greedy FACILITY-LOCATION marginal gain (coverage) - retention penalty,
    #    respecting the per-category cap
    sel = set(selected)
    while len(selected) < a.n_select:
        gain = np.maximum(0.0, Cov - best[None, :]).sum(1) - a.lam * np.maximum(0.0, ret_par - a.rho)
        bestg, bi = -1e18, -1
        for i in np.argsort(-gain):
            if i in sel or cat_count[cand_cat[i]] >= a.cap:
                continue
            bestg, bi = gain[i], i
            break
        if bi < 0:  # cap saturated everywhere -> relax cap
            for i in np.argsort(-gain):
                if i not in sel:
                    bi = i
                    break
        take(bi); sel.add(bi)

    chosen = sorted(int(cand_eps[i]) for i in selected)
    mix = defaultdict(int)
    for i in selected:
        mix[cand_cat[i]] += 1
    sel_idx = np.array(selected)
    tfrac = sum(1 for i in selected if cand_cat[i] in TARGETED) / len(selected)
    sel_cosD = float(csel[sel_idx].mean())
    sel_retp = float(ret_par[sel_idx].mean())
    out = {"method": "rc_less", "n_select": len(chosen), "m": a.m, "floor": a.floor, "cap": a.cap,
           "lam": a.lam, "no_center": a.no_center, "core_episodes": chosen,
           "selected_targeted_frac": tfrac, "sel_cos_gD": sel_cosD, "sel_ret_par": sel_retp,
           "selected_mix": dict(sorted(mix.items(), key=lambda x: -x[1]))}
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"[rc] wrote {a.out}: {len(chosen)} demos, targeted_frac={tfrac:.3f}")
    print(f"[rc] GATE: selected cos(demo,g_D)={sel_cosD:.3f} (pool {csel.mean():.3f}; want >=) | "
          f"<z,g_R>={sel_retp:.3f} (pool {ret_par.mean():.3f}; want >=)")
    print(f"[rc] mix (top 15): {dict(list(out['selected_mix'].items())[:15])}")


if __name__ == "__main__":
    main()
