"""Theory demo, stage 2: does the batch gradient predict the realized weight update?

Tests the assumption 'similar gradients => similar effect' directly in weight space,
on every retained pull:
  G_i        = sum of the draw's per-demo RAW gradient sketches at pi_0 (+ scaled D0 term)
  dtheta_i   = JL sketch of LoRA(final) - LoRA(pi_0)   (same projection => comparable)

Reports:
  A. |dtheta| per pull vs realized delta (does the poison pull stand out?)
  B. pairwise cos(dtheta_i, dtheta_j) decomposed by relation:
       null-null (same data, diff seed) / same-round cross-arm (diff data, same seed)
       / same-arm cross-round (diff draw, diff seed) / cross-arm cross-round
  C. Mantel-style Spearman: pairwise cos(G) vs pairwise cos(dtheta)
  D. first-order alignment cos(-G_full_i, dtheta_i) at step 19999 and 5000
     (sign check doubles as the leaf-ordering sanity gate: clearly positive at 5000
      = orderings match; ~0 everywhere = suspect flatten-order mismatch, say so)

Run: /data/xinyua11/conda/envs/robocasa/bin/python gradient_analysis/theory_analyze.py
"""
import glob
import itertools
import json
import os

import numpy as np

GA = "/data/xinyua11/robocasa/gradient_analysis"
TW = f"{GA}/theory_weights"
D0_TOTAL = 400  # D0 episodes in every training mix; we have 120 sampled -> scale


def load_demo_archive():
    eps, S = [], []
    for d in (f"{GA}/sketches_pi0base_19999", f"{GA}/sketches_pi0base_19999_pool"):
        m = json.load(open(f"{d}/episodes.json"))
        eps += list(m["episodes"])
        S.append(np.load(f"{d}/sketches.npy"))
    S = np.vstack(S)
    idx = {e: i for i, e in enumerate(eps)}
    d0_ids = [e for d in (f"{GA}/sketches_pi0base_19999",) for e, t in
              json.load(open(f"{d}/episodes.json"))["tags"].items() if "d0_sample" in t]
    d0_rows = [idx[int(e)] for e in d0_ids]
    return idx, S, d0_rows


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    jobs = json.load(open(f"{GA}/theory_jobs.json"))
    idx, S, d0_rows = load_demo_archive()
    g_d0 = S[d0_rows].sum(0) * (D0_TOTAL / len(d0_rows))

    pulls = {}
    for f in glob.glob(f"{TW}/*__19999.npz"):
        pid = os.path.basename(f).split("__")[0]
        z = np.load(f)
        pulls[pid] = {"dtheta": z["sketch"].astype(np.float64), "dnorm": float(z["norm"])}
    print(f"[a] {len(pulls)} finals loaded")

    for pid, rec in pulls.items():
        spec = jobs[pid]
        ids = spec["demo_ids"]
        rows = [idx[e] for e in ids if e in idx]
        rec.update(arm=spec["arm"], round=spec["round"], delta=spec["delta"],
                   coverage=(len(rows) / len(ids)) if ids else None)
        if ids and len(rows) / len(ids) > 0.9:
            G = S[rows].sum(0)
            rec["G"] = G.astype(np.float64)
            rec["G_full"] = (G + g_d0).astype(np.float64)
        elif not ids:
            rec["G_full"] = g_d0.astype(np.float64)  # null pulls: D0-only batch

    # ---------- A. magnitudes ----------
    print("\nA. |dtheta| (LoRA weight-update magnitude) vs realized delta:")
    for pid, r in sorted(pulls.items(), key=lambda kv: kv[1]["delta"]):
        cov = "" if r["coverage"] in (None, 1.0) else f"  (G coverage {r['coverage']:.2f})"
        print(f"  {pid:32s} |dtheta| {r['dnorm']:7.3f}   delta {r['delta']:+.3f}{cov}")

    # ---------- B. pairwise dtheta similarity by relation ----------
    def relation(a, b):
        pa, pb = pulls[a], pulls[b]
        if pa["arm"] == "null" and pb["arm"] == "null":
            return "null-null (same data, diff seed)"
        if pa["round"] == pb["round"] and pa["arm"] != pb["arm"]:
            return "same round (same SEED, diff data)"
        if pa["arm"] == pb["arm"]:
            return "same arm (diff draw, diff seed)"
        return "diff arm, diff round"

    groups = {}
    pids = sorted(pulls)
    for a, b in itertools.combinations(pids, 2):
        if "planted_bad" in a or "planted_bad" in b:
            continue  # reported separately
        groups.setdefault(relation(a, b), []).append(cos(pulls[a]["dtheta"], pulls[b]["dtheta"]))
    print("\nB. pairwise cos(dtheta_i, dtheta_j) by relation (poison excluded):")
    for k, v in sorted(groups.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"  {k:38s} mean {np.mean(v):+.3f}  sd {np.std(v):.3f}  n={len(v)}")
    pb = [p for p in pids if "planted_bad" in p]
    if pb:
        v = [cos(pulls[a]["dtheta"], pulls[b]["dtheta"]) for a in pb for b in pids if b not in pb]
        vp = cos(pulls[pb[0]]["dtheta"], pulls[pb[1]]["dtheta"]) if len(pb) == 2 else None
        print(f"  {'POISON vs all normal pulls':38s} mean {np.mean(v):+.3f}  sd {np.std(v):.3f}")
        if vp is not None:
            print(f"  {'POISON vs POISON':38s} mean {vp:+.3f}")

    # ---------- C. Mantel: cos(G) vs cos(dtheta) ----------
    withG = [p for p in pids if "G" in pulls[p]]
    gs, ds = [], []
    for a, b in itertools.combinations(withG, 2):
        gs.append(cos(pulls[a]["G"], pulls[b]["G"]))
        ds.append(cos(pulls[a]["dtheta"], pulls[b]["dtheta"]))
    print(f"\nC. across {len(withG)} pulls ({len(gs)} pairs): "
          f"Spearman[ cos(G_i,G_j) vs cos(dtheta_i,dtheta_j) ] = {spearman(gs, ds):+.3f}")
    # same excluding poison
    wg = [p for p in withG if "planted_bad" not in p]
    gs2, ds2 = [], []
    for a, b in itertools.combinations(wg, 2):
        gs2.append(cos(pulls[a]["G"], pulls[b]["G"]))
        ds2.append(cos(pulls[a]["dtheta"], pulls[b]["dtheta"]))
    print(f"   excluding poison: {spearman(gs2, ds2):+.3f}  (n={len(gs2)} pairs)")

    # ---------- D. first-order alignment ----------
    print("\nD. cos(-G_full_i, dtheta_i): batch loss-gradient at pi_0 vs realized update")
    early = {}
    for f in glob.glob(f"{TW}/*__5000.npz"):
        early[os.path.basename(f).split("__")[0]] = np.load(f)["sketch"].astype(np.float64)
    for pid in pids:
        r = pulls[pid]
        if "G_full" not in r:
            continue
        c19 = cos(-r["G_full"], r["dtheta"])
        c5 = cos(-r["G_full"], early[pid]) if pid in early else None
        tag = f"  @5000 {c5:+.3f}" if c5 is not None else ""
        print(f"  {pid:32s} @19999 {c19:+.3f}{tag}")

    out = {p: {k: (v if not isinstance(v, np.ndarray) else None)
               for k, v in r.items() if k in ("arm", "round", "delta", "dnorm", "coverage")}
           for p, r in pulls.items()}
    json.dump(out, open(f"{GA}/theory_demo_report.json", "w"), indent=1)
    print(f"\n[a] wrote {GA}/theory_demo_report.json")


if __name__ == "__main__":
    main()
