"""Trust-gated index (c-meter) — SHADOW mode for the task-2 gradient-arm race.

Per the 2026-08-18 decision: rounds 1-3 allocate KILL+UNIFORM exactly as
pre-registered; this module only COMPUTES what a trust-gated UCB would do and
logs it, so the replication stays clean while the mechanism accumulates its
demonstration. Round-4 live allocation is a separate, explicit decision.

Per pull (arm, j):
  Delta       : paired score = delta(arm,j) − delta(random,j)   [pp]
  phi_probe   : loss_balanced − loss_retention from shadow/<pid>_probe9999.json
  phi_grad_*  : pre-training gradient stats of the pull's OWN B=200 demo_ids
                from the frozen sketches (whitened space): mean pairwise cos
                (coherence), mean norm, mean cos to own-arm centroid.
Trust per proxy: Fisher-z shrunk correlation c = tanh( n/(n+N0) * atanh(r) ),
N0=8; gate = |c| >= 0.35 AND n >= 8. Index(a) = paired_mean(a)
+ t90*sigma/sqrt(n_a) + 1[gate]*c*zscore(phi(a)) * sigma.
Writes ucb_robot/cmeter_report.json and prints the round table.
"""
import json
import os
import sys
from math import atanh, tanh

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/xinyua11/robocasa")
assert os.environ.get("BANDIT_TASK_PROFILE") == "ppccab"

from scipy import stats as sps

from bandit_v1 import ledger

GA = "/data/xinyua11/robocasa/gradient_analysis/ppccab"
UCB = f"{GA}/ucb_robot"
SHARDS = [f"{GA}/sketches_ppccabbase_9999_shard0of2", f"{GA}/sketches_ppccabbase_9999_shard1of2"]
GC_ARMS = [f"gc{c}" for c in range(6)]
N0 = 8
C_MIN = 0.35
SIGMA0 = 3.2          # task-1 paired sigma prior, replaced by task-2 estimate when df>=4


def load_sketch_space():
    import sys as _s
    _s.path.insert(0, "/data/xinyua11/robocasa/gradient_analysis")
    from gradarm_cluster import whiten
    eps, rows = [], []
    for d in SHARDS:
        m = json.load(open(f"{d}/episodes.json"))
        eps += [int(e) for e in m["episodes"]]
        rows.append(np.load(f"{d}/sketches.npy"))
    X = np.concatenate(rows).astype(np.float64)
    order = np.argsort(eps)
    eps = [eps[i] for i in order]
    Xw, _ = whiten(X[order])
    return {e: i for i, e in enumerate(eps)}, Xw


def grad_stats(demo_ids, idx, Xw, centroid):
    ii = [idx[e] for e in demo_ids if e in idx]
    V = Xw[ii]
    G = V @ V.T
    n = len(V)
    coh = float((G.sum() - np.trace(G)) / max(1, n * (n - 1)))
    to_c = float((V @ centroid).mean())
    return {"phi_grad_coherence": coh, "phi_grad_centroid_cos": to_c}


def shrunk_corr(x, y):
    n = len(x)
    if n < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0, 0.0, n
    r = float(np.corrcoef(x, y)[0, 1])
    r = max(min(r, 0.999), -0.999)
    c = tanh(n / (n + N0) * atanh(r))
    return c, r, n


def calibrated_validity(phi, delta, kappa=0.75, sigma_rollout=SIGMA0):
    """PRIMARY trust machinery (2026-08-18 review amendment): fit
    f̂(φ)≈E[Δ|φ] on the calibration pairs (all uniform-allocation pulls, so
    selection-free) and certify a jackknife-conformal residual bound
    B = q90 over leave-one-out |Δ_i − f̂_{-i}(φ_i)|. The proxy earns entry
    iff B < kappa·σ_rollout — a proxy prediction must be tighter than a real
    pull's own noise to be worth borrowing — and, per-arm, only inside the
    calibration support [min φ, max φ] (tail/OOD guard: a poison-like arm
    with extreme φ gets w=0 even when the global fit looks good).
    Returns dict(fit, B, gate, support)."""
    n = len(phi)
    if n < 4 or np.std(phi) < 1e-12:
        return {"n": n, "gate": False, "reason": "n<4 or degenerate phi"}
    phi = np.asarray(phi, float); delta = np.asarray(delta, float)
    # PARAMETRIC latent-proxy-error bound (2026-08-18 review-2 audit): NOT a
    # conformal quantile. Under independence, Var(resid) = s_proxy^2 +
    # sigma_meas^2, so subtract the KNOWN rollout-noise variance from the
    # dof-corrected residual variance; B = z90 * s_proxy. Assumes Gaussian-ish
    # residuals; no distribution-free coverage claimed. The sim
    # (proxy_continuum.py) caught the original LOO-quantile version rejecting
    # PERFECT proxies (residuals dominated by the calibration pull's own noise).
    b1, b0 = np.polyfit(phi, delta, 1)
    rss = float(np.sum((delta - (b1 * phi + b0)) ** 2))
    s_tot2 = rss / (n - 2)
    s_p2 = max(0.0, s_tot2 - sigma_rollout ** 2)
    B = 1.28 * float(np.sqrt(s_p2))
    gate = bool(B < kappa * sigma_rollout)
    return {"n": n, "slope": round(float(b1), 3), "intercept": round(float(b0), 3),
            "latent_B90": round(B, 2), "kappa_sigma": round(kappa * sigma_rollout, 2),
            "s_total": round(float(np.sqrt(s_tot2)), 2),
            "gate": gate, "support": [round(float(phi.min()), 4), round(float(phi.max()), 4)]}


def main():
    pulls = ledger.read("pulls")
    pulls = pulls[pulls.status == "smoke"]
    rows = []
    arms_def = json.load(open(f"{UCB}/arms_r3.json"))
    idx, Xw = load_sketch_space()
    cents = {a: np.mean(Xw[[idx[e] for e in arms_def[a] if e in idx]], axis=0)
             for a in GC_ARMS if a in arms_def}
    for a in cents:
        cents[a] /= (np.linalg.norm(cents[a]) + 1e-12)

    race = pulls[pulls.pull_id.str.contains(r"_j14\d$", na=False)]
    for j, grp in race.groupby("round_j"):
        ctrl = grp[grp.arm == "random"]
        d_ctrl = float(ctrl.delta.iloc[0]) * 100 if len(ctrl) else None
        for _, r in grp.iterrows():
            if r.arm == "random":
                continue
            row = {"arm": r.arm, "j": int(j), "delta_pp": float(r.delta) * 100,
                   "paired_pp": None if d_ctrl is None else float(r.delta) * 100 - d_ctrl}
            pj = f"{UCB}/shadow/{r.pull_id}_probe9999.json"
            if os.path.exists(pj):
                row["phi_probe"] = json.load(open(pj))["reward"]
            if r.arm in cents:
                row.update(grad_stats([int(x) for x in r.demo_ids], idx, Xw, cents[r.arm]))
            rows.append(row)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    have = df.dropna(subset=["paired_pp"])
    sig = SIGMA0
    per_arm = have.groupby("arm").paired_pp.agg(["mean", "count", "std"])
    if (per_arm["count"] >= 2).sum() >= 2:
        pooled = have.groupby("arm").paired_pp.transform("mean")
        resid = have.paired_pp - pooled
        dfree = len(have) - have.arm.nunique()
        if dfree >= 4:
            sig = float(np.sqrt((resid ** 2).sum() / dfree))

    report = {"n_pulls": len(df), "sigma_used": sig, "proxies": {}, "arms": {},
              "note": "all pairs are uniform-allocation (selection-free calibration stream); "
                      "PRIMARY gate = calibrated validity, corr gate = ablation only"}
    for proxy in ["phi_probe", "phi_grad_coherence", "phi_grad_centroid_cos"]:
        sub = have.dropna(subset=[proxy]) if proxy in have else pd.DataFrame()
        if len(sub) >= 3:
            c, r, n = shrunk_corr(sub[proxy].values, sub.paired_pp.values)
            cal = calibrated_validity(sub[proxy].values, sub.paired_pp.values,
                                      sigma_rollout=sig)
            report["proxies"][proxy] = {"ablation_corr": {"raw_r": round(r, 3),
                                        "shrunk_c": round(c, 3),
                                        "gate": bool(abs(c) >= C_MIN and n >= N0)},
                                        "calibrated": cal, "n": n,
                                        "gate": cal.get("gate", False)}
    t90 = {1: 3.08, 2: 1.89, 3: 1.64, 4: 1.53}
    for a, g in per_arm.iterrows():
        n = int(g["count"])
        tt = t90.get(n, 1.44)
        ent = {"paired_mean": round(float(g["mean"]), 2), "n": n,
               "lcb": round(float(g["mean"]) - tt * sig / np.sqrt(n), 2),
               "ucb_bound": round(float(g["mean"]) + tt * sig / np.sqrt(n), 2)}
        bonus, guards = 0.0, []
        for proxy, pr in report["proxies"].items():
            if not pr["gate"]:
                continue
            cal = pr["calibrated"]
            pa = have[(have.arm == a)].dropna(subset=[proxy])
            if not len(pa):
                continue
            pv = float(pa[proxy].mean())
            lo, hi = cal["support"]
            if not (lo <= pv <= hi):          # tail/OOD guard: outside support -> no borrow
                guards.append(f"{proxy}:OOD")
                continue
            pred = cal["slope"] * pv + cal["intercept"]
            w = max(0.0, 1.0 - (cal["latent_B90"] / (cal["kappa_sigma"] + 1e-9)) ** 2)
            bonus += w * (pred - float(have.paired_pp.mean()))
        ent["shadow_index"] = round(ent["paired_mean"] + tt * sig / np.sqrt(n) + bonus, 2)
        ent["proxy_bonus"] = round(bonus, 2)
        if guards:
            ent["ood_guards"] = guards
        report["arms"][a] = ent
    json.dump(report, open(f"{UCB}/cmeter_report.json", "w"), indent=1)
    print("\n[cmeter] sigma=%.2f  proxies=%s" % (sig, report["proxies"]))
    print("[cmeter] arms:", json.dumps(report["arms"], indent=1))
    print("CMETER DONE")


if __name__ == "__main__":
    main()
