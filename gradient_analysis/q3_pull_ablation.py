"""Q3: does ANY pi0-time gradient statistic of a 200-demo draw predict realized delta?

The design's "logged-only selector ablation", run over every completed pull with a
draw (12 race non-null + 4 gradarm = 16 pulls; nulls have no draw). All stats are
computed from the pool-wide 2048-d JL sketches at pi0base/19999 (merged Q0 +
pool_sketches archives) -- nothing here was used to select anything (logged-only).

Per-draw statistics:
  mean_gnorm : mean true LoRA-grad norm of the draw's demos
  coherence  : ||mean of unit sketches||           (batch alignment incl. common mode)
  wsep       : ||mean of whitened unit sketches|| (batch-mean separation after
               removing top-10 global SVD modes -- the gradarm pre-flight statistic)
  self_sim   : mean pairwise cosine within the draw (redundancy)
  cos_region : mean cosine to the Q0 tall-region contrast direction

Correlations vs realized delta: Spearman across all 16; Pearson after demeaning
within paired-seed round (j3/j4/j5 share seeds across arms). n=16 -- exploratory,
|rho| >~ 0.5 needed for p<0.05.

Run: /data/xinyua11/conda/envs/robocasa/bin/python gradient_analysis/q3_pull_ablation.py
"""
import json
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "/data/xinyua11/robocasa")
GA = "/data/xinyua11/robocasa/gradient_analysis"


def load_archive(d):
    m = json.load(open(f"{d}/episodes.json"))
    S = np.load(f"{d}/sketches.npy")
    N = np.load(f"{d}/norms.npy")
    return m["episodes"], S, N


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    e1, S1, N1 = load_archive(f"{GA}/sketches_pi0base_19999")
    e2, S2, N2 = load_archive(f"{GA}/sketches_pi0base_19999_pool")
    eps = list(e1) + list(e2)
    S = np.vstack([S1, S2]); Nrm = np.concatenate([N1, N2])
    idx = {e: i for i, e in enumerate(eps)}
    U = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
    print(f"[q3] merged archive: {len(eps)} demos")

    # global whitening basis from a fixed pool sample (excl. nothing; deterministic)
    rng = np.random.RandomState(7)
    samp = rng.choice(len(eps), size=3000, replace=False)
    X = U[samp] - U[samp].mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    P = Vt[:10]

    def whiten_unit(M):
        W = M - (M @ P.T) @ P
        return W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)

    # region direction from Q0 gate lists
    lists = json.load(open(f"{GA}/demo_lists.json"))
    gi = [idx[e] for e in lists["gate_in_region"]]
    go = [idx[e] for e in lists["gate_out_region"]]
    d_region = U[gi].mean(0) - U[go].mean(0)
    d_region /= np.linalg.norm(d_region) + 1e-12

    from bandit_v1 import ledger
    p = ledger.read("pulls")
    p = p[p.status.isin(["ok", "smoke"])]
    rows = []
    for _, r in p.iterrows():
        ids = r.demo_ids
        if ids is None or (hasattr(ids, "__len__") and len(ids) == 0):
            continue
        ids = [int(x) for x in ids]
        ii = [idx[e] for e in ids if e in idx]
        if len(ii) < 150:
            continue
        Up = U[ii]
        Uw = whiten_unit(Up)
        n_pair = min(len(ii), 200)
        rows.append({
            "pull": str(r.pull_id), "arm": str(r.arm), "round": int(r.round_j),
            "delta": float(r.delta),
            "mean_gnorm": float(Nrm[ii].mean()),
            "coherence": float(np.linalg.norm(Up.mean(0))),
            "wsep": float(np.linalg.norm(Uw.mean(0))),
            "self_sim": float((Up @ Up.T)[np.triu_indices(len(ii), 1)].mean()),
            "cos_region": float((Up @ d_region).mean()),
        })
    rows.sort(key=lambda x: (x["round"], x["pull"]))
    print(f"[q3] {len(rows)} pulls with draws")
    hdr = ["pull", "round", "delta", "mean_gnorm", "coherence", "wsep", "self_sim", "cos_region"]
    print("  " + "  ".join(f"{h:>10s}" for h in hdr))
    for x in rows:
        print(f"  {x['pull']:>26s}  {x['round']:2d}  {x['delta']:+.3f}"
              f"  {x['mean_gnorm']:10.3f}  {x['coherence']:10.4f}  {x['wsep']:10.4f}"
              f"  {x['self_sim']:10.4f}  {x['cos_region']:+10.4f}")

    delta = np.array([x["delta"] for x in rows])
    rounds = np.array([x["round"] for x in rows])
    print("\n[q3] stat vs realized delta (n=%d):" % len(rows))
    for stat in ["mean_gnorm", "coherence", "wsep", "self_sim", "cos_region"]:
        v = np.array([x[stat] for x in rows])
        rho = spearman(v, delta)
        # within-round demeaned Pearson (controls the paired-seed round effect)
        vd, dd = v.copy().astype(float), delta.copy()
        for rd in np.unique(rounds):
            m = rounds == rd
            vd[m] -= vd[m].mean(); dd[m] -= dd[m].mean()
        rp = float(np.corrcoef(vd, dd)[0, 1])
        print(f"  {stat:>10s}: spearman {rho:+.3f}   within-round pearson {rp:+.3f}")

    json.dump(rows, open(f"{GA}/q3_pull_ablation.json", "w"), indent=1)
    print(f"\n[q3] wrote {GA}/q3_pull_ablation.json")


if __name__ == "__main__":
    main()
