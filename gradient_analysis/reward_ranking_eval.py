"""Reward-candidate ranking evaluation (owner's gate, 2026-08-05/06).

Question: which candidate reward RANKS arms the way ground-truth rollout
delta does? Candidates scored per ARM from existing artifacts only:

  probe19999 : composite (loss_balanced - loss_retention) at 20k ckpts, per-arm mean
  probe10000 : same at 10k ckpts (parquet may be partial; report coverage)
  inf_target : cos( mean_grad(arm), mean_grad(gate_in target set) )   [LESS-style]
  inf_d0     : cos( mean_grad(arm), mean_grad(D0 sample) )            [interference]
  inf_div    : 1 - mean pairwise cos within arm                        [diversity]
  inf_style  : cos( mean_grad(arm), style_hi_axis )  (circular for style arms; noted)

Whitened sketch space (top-10 modes removed, Q0-validated). Ground truth =
per-arm mean rollout delta from the ledger (18 pulls, 8 arms).
Pre-registered thresholds (owner): DROP probe loss if Spearman<0.5 on the
6-arm core OR style_lo not last OR an influence variant beats it by >=0.2.
KEEP if >=0.8 and >= all influence variants.
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/xinyua11/robocasa")
sys.path.insert(0, "/data/xinyua11/robocasa/gradient_analysis")

GA = "/data/xinyua11/robocasa/gradient_analysis"
CORE6 = ["style_hi", "mid_band", "tall_vessel_grasp_fail", "random", "easy_band", "style_lo"]


def main():
    from scipy.stats import spearmanr
    from gradarm_cluster import load_merged
    from bandit_v1 import ledger

    # ground truth per arm
    p = ledger.read("pulls")
    ok = p[p.status.isin(["ok", "smoke"]) & ~p.arm.isin(["null"])]
    truth = (ok.groupby("arm")["delta"].mean() * 100).to_dict()
    truth["gc2"] = truth.pop("gradarm_a")
    truth["gc3"] = truth.pop("gradarm_b")
    print("ground truth (pp):", {k: round(v, 2) for k, v in sorted(truth.items(), key=lambda kv: -kv[1])})

    # whitened sketches for pool + transform for D0 rows
    eps, X, norms, reg = load_merged()
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    mu = Xn.mean(0, keepdims=True)
    Xc = Xn - mu
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    top = Vt[:10]
    def wh(M):
        Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
        Mc = Mn - mu
        Mw = Mc - (Mc @ top.T) @ top
        return Mw / (np.linalg.norm(Mw, axis=1, keepdims=True) + 1e-12)
    Xw = wh(X)
    idx = {e: i for i, e in enumerate(eps)}

    # D0 sample sketches from the Q0 dir (excluded from pool merge)
    q0 = json.load(open(f"{GA}/sketches_pi0base_19999/episodes.json"))
    q0_s = np.load(f"{GA}/sketches_pi0base_19999/sketches.npy")
    d0_rows = [i for i, e in enumerate(q0["episodes"]) if "d0_sample" in q0["tags"][str(e)]]
    D0w = wh(q0_s[d0_rows].astype(np.float64))
    lists = json.load(open(f"{GA}/demo_lists.json"))
    tgt_ids = [idx[int(e)] for e in lists["gate_in_region"] if int(e) in idx]

    arms = json.load(open(f"{GA}/ucb_robot/arms.json"))
    arms = {k: v for k, v in arms.items() if k != "random"}
    pool_all = list(range(len(eps)))
    arm_rows = {a: [idx[int(e)] for e in ids if int(e) in idx] for a, ids in arms.items()}
    arm_rows["random"] = pool_all

    tgt_dir = Xw[tgt_ids].mean(0)
    d0_dir = D0w.mean(0)
    style_axis = Xw[arm_rows["style_hi"]].mean(0) - Xw[arm_rows["style_lo"]].mean(0)
    rng = np.random.default_rng(0)

    scores = {}
    for a, rows in arm_rows.items():
        M = Xw[rows]
        m = M.mean(0)
        sub = M[rng.choice(len(M), min(400, len(M)), replace=False)]
        G = sub @ sub.T
        div = 1 - (G.sum() - np.trace(G)) / (len(sub) * (len(sub) - 1))
        scores[a] = {
            "inf_target": float(m @ tgt_dir / (np.linalg.norm(m) * np.linalg.norm(tgt_dir))),
            "inf_d0": float(m @ d0_dir / (np.linalg.norm(m) * np.linalg.norm(d0_dir))),
            "inf_div": float(div),
            "inf_style": float(m @ style_axis / (np.linalg.norm(m) * np.linalg.norm(style_axis))),
        }

    # probe-loss candidates per arm
    def probe_scores(path):
        try:
            d = pd.read_parquet(path)
        except Exception:
            return {}, 0
        d = d[~d.pull_id.str.startswith(("null", "pi0base"))].copy()
        d["arm"] = (d.pull_id.str.replace(r"_j\d+$", "", regex=True)
                    .str.replace("gradarm_a", "gc2").str.replace("gradarm_b", "gc3"))
        d["comp"] = d.loss_balanced - d.loss_retention
        return d.groupby("arm")["comp"].mean().to_dict(), len(d)

    p20, n20 = probe_scores(f"{GA}/loss_probe_calibration.parquet")
    p10, n10 = probe_scores(f"{GA}/loss_probe_calibration_10000.parquet")
    print(f"probe coverage: 19999 -> {n20} ckpts, 10000 -> {n10} ckpts")

    cands = {k: {a: v[k] for a, v in scores.items()} for k in
             ["inf_target", "inf_d0", "inf_div", "inf_style"]}
    cands["probe19999"] = p20
    if n10 >= 12:
        cands["probe10000"] = p10

    print(f"\n{'candidate':12s} {'rho6':>7s} {'rho_all':>8s} {'lo_last':>8s} {'rand>easy':>10s}")
    results = {}
    for name, sc in cands.items():
        common6 = [a for a in CORE6 if a in sc]
        commonA = [a for a in truth if a in sc]
        if len(common6) < 5:
            print(f"{name:12s}  insufficient coverage {common6}")
            continue
        r6 = spearmanr([sc[a] for a in common6], [truth[a] for a in common6]).statistic
        rA = spearmanr([sc[a] for a in commonA], [truth[a] for a in commonA]).statistic
        lo_last = min(common6, key=lambda a: sc[a]) == "style_lo" if "style_lo" in common6 else None
        re_ok = sc.get("random", np.nan) > sc.get("easy_band", np.nan)
        results[name] = {"rho6": round(float(r6), 3), "rho_all": round(float(rA), 3),
                         "style_lo_last": bool(lo_last), "random_gt_easy": bool(re_ok),
                         "scores": {a: round(float(sc[a]), 4) for a in commonA}}
        print(f"{name:12s} {r6:7.2f} {rA:8.2f} {str(lo_last):>8s} {str(re_ok):>10s}")
    json.dump({"truth": truth, "results": results},
              open(f"{GA}/reward_ranking_results.json", "w"), default=str)
    print("\nwrote reward_ranking_results.json")


if __name__ == "__main__":
    main()
