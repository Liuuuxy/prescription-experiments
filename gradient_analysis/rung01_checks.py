"""Rung 0 + Rung 1 (free ledger/sketch arithmetic) — overnight 2026-08-06.

Rung 0: (a) split-half reliability of the ground-truth ARM ranking;
        (b) exact family-wise permutation null for the candidate rewards
            on the 6 non-style arms (720 permutations, enumerated).
Rung 1: corrected-protocol supervised axes — rotate the supervising pair
        (mid-vs-easy, gc3-vs-gc2), evaluate on arms outside construction.
Writes gradient_analysis/rung01_results.json.
"""
import itertools
import json
import sys

import numpy as np

sys.path.insert(0, "/data/xinyua11/robocasa")
sys.path.insert(0, "/data/xinyua11/robocasa/gradient_analysis")

GA = "/data/xinyua11/robocasa/gradient_analysis"


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def main():
    from bandit_v1 import ledger
    p = ledger.read("pulls")
    ok = p[p.status.isin(["ok", "smoke"]) & ~p.arm.isin(["null"])]
    pulls = {a: list(g.delta * 100) for a, g in ok.groupby("arm")}
    pulls["gc2"] = pulls.pop("gradarm_a"); pulls["gc3"] = pulls.pop("gradarm_b")
    arms = sorted(pulls)
    truth = {a: float(np.mean(v)) for a, v in pulls.items()}

    # ---- rung 0a: split-half reliability of the arm ranking ----
    rng = np.random.default_rng(0)
    rhos = []
    for _ in range(4000):
        h1, h2 = [], []
        for a in arms:
            v = rng.permutation(pulls[a])
            k = max(1, len(v) // 2)
            h1.append(np.mean(v[:k])); h2.append(np.mean(v[k:]))
        rhos.append(spearman(h1, h2))
    rel = float(np.mean(rhos))
    rel_sb = 2 * rel / (1 + rel) if rel > -1 else float("nan")  # Spearman-Brown to full-n
    ceiling = float(np.sqrt(max(rel_sb, 0)))

    # ---- rung 0b: exact family permutation null on 6 non-style arms ----
    rr = json.load(open(f"{GA}/reward_ranking_results.json"))
    non_style = ["mid_band", "gc3", "tall_vessel_grasp_fail", "random", "easy_band", "gc2"]
    fam = {}
    for cand in ["inf_target", "inf_d0", "inf_div", "inf_style", "probe19999"]:
        sc = rr["results"][cand]["scores"]
        if all(a in sc for a in non_style):
            fam[cand] = [sc[a] for a in non_style]
    tvals = [truth[a] for a in non_style]
    obs = {c: spearman(v, tvals) for c, v in fam.items()}
    obs_max = max(abs(x) for x in obs.values())
    null_max = []
    for perm in itertools.permutations(tvals):
        null_max.append(max(abs(spearman(v, list(perm))) for v in fam.values()))
    null_max = np.array(null_max)
    p_fam = float((null_max >= obs_max).mean())
    q95 = float(np.quantile(null_max, 0.95))

    # ---- rung 1: rotated supervised axes ----
    from gradarm_cluster import load_merged
    eps, X, _, _ = load_merged()
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    mu = Xn.mean(0, keepdims=True); Xc = Xn - mu
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    top = Vt[:10]
    Xw = Xc - (Xc @ top.T) @ top
    Xw /= (np.linalg.norm(Xw, axis=1, keepdims=True) + 1e-12)
    idx = {e: i for i, e in enumerate(eps)}
    am = json.load(open(f"{GA}/ucb_robot/arms.json"))
    rows = {a: [idx[int(e)] for e in ids if int(e) in idx] for a, ids in am.items() if a != "random"}
    rows["random"] = list(range(len(eps)))
    means = {a: Xw[r].mean(0) for a, r in rows.items()}
    all8 = ["style_hi", "style_lo", "mid_band", "easy_band", "tall_vessel_grasp_fail",
            "random", "gc2", "gc3"]
    rot = {}
    for hi, lo in [("mid_band", "easy_band"), ("gc3", "gc2"), ("style_hi", "style_lo")]:
        ax = means[hi] - means[lo]
        evalset = [a for a in all8 if a not in (hi, lo)]
        sc = [float(means[a] @ ax / (np.linalg.norm(means[a]) * np.linalg.norm(ax)))
              for a in evalset]
        rot[f"axis_{hi}-{lo}"] = {"eval_arms": evalset,
                                  "rho": spearman(sc, [truth[a] for a in evalset]),
                                  "scores": dict(zip(evalset, [round(s, 4) for s in sc]))}

    out = {"truth": truth,
           "rung0": {"split_half_rho": round(rel, 3), "spearman_brown": round(rel_sb, 3),
                     "observable_ceiling": round(ceiling, 3),
                     "family_obs": {k: round(v, 3) for k, v in obs.items()},
                     "family_obs_max": round(obs_max, 3),
                     "family_null_q95": round(q95, 3), "family_p": round(p_fam, 4)},
           "rung1": rot}
    json.dump(out, open(f"{GA}/rung01_results.json", "w"))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
