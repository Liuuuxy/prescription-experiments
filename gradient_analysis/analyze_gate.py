"""Q0 encoding-gate analysis on the pi0-base gradient sketches (CPU, offline).

Question: does the pi_0 LoRA flow-matching gradient ENCODE membership in the
tall_vessel_grasp_fail region? Per the gradient-encoding principle (memory:
influence-gated-by-gradient-encoding), influence/teachability signals can only
work if it does; prior category-level best-mode AUC was 0.56 (weak). Bar for a
workable signal: best-mode AUC >~ 0.7.

Reads gradient_analysis/sketches_pi0base_19999/ (from grad_sketches.py).
Metrics (all on unit-normalized sketches; cosine is JL-preserved):
  1. best-single-SVD-mode AUC (unsupervised; the validated cheap predictor)
  2. split-half contrast AUC (supervised: direction fit on half, AUC on the
     other half; both swaps averaged) -- the honest version of the smoke gate
  3. whitened variants: same after projecting out top-k global SVD modes
     (k=10,25,50; the whitening move that lifted category AUC 0.605->0.677)
  4. controls: shuffled-label contrast AUC (want ~0.5), random-direction AUC
  5. descriptive: common-mode fraction (mean cos to global mean direction),
     within/between-set mean pairwise cos, grad-norm distributions per set
Also, when pull-set sketches are present, per-pull-set: mean cos to the gate
contrast direction, common-mode fraction, self-similarity, norms -- the Q1
inputs (does tall_j3/j4 carry MORE region signal than random_j3/j4?).

--selftest: synthetic planted-direction sanity (expect AUC>0.9 planted, ~0.5 shuffled).

Run: /data/xinyua11/conda/envs/robocasa/bin/python gradient_analysis/analyze_gate.py
"""
import argparse
import json
import os

import numpy as np

DIR = "/data/xinyua11/robocasa/gradient_analysis/sketches_pi0base_19999"
OUT = "/data/xinyua11/robocasa/gradient_analysis/q0_gate_report.json"
PULL_SETS = ["tall_vessel_grasp_fail_j3", "tall_vessel_grasp_fail_j4",
             "random_j3", "random_j4", "mid_band_j3", "mid_band_j4",
             "easy_band_j3", "easy_band_j4"]


def auc(pos, neg):
    """Mann-Whitney AUC of score separation (pos > neg), tie-aware, vectorized."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    vals, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    avg_rank = (csum - (cnt - 1) / 2.0)
    ranks = avg_rank[inv]
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def best_mode_auc(U_all, labels, n_modes=40):
    """Best single-SVD-mode AUC over the pooled (centered) cloud. Sign-free."""
    X = U_all - U_all.mean(0, keepdims=True)
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    best, best_i = 0.5, -1
    per_mode = []
    for i in range(min(n_modes, Vt.shape[0])):
        proj = X @ Vt[i]
        a = auc(proj[labels], proj[~labels])
        a = max(a, 1 - a)
        per_mode.append(round(a, 4))
        if a > best:
            best, best_i = a, i
    var_frac = (s**2 / (s**2).sum())[:n_modes]
    return best, best_i, per_mode, var_frac


def split_half_contrast_auc(U_in, U_out, seed=0):
    """Direction = mean(in)-mean(out) on half A, AUC measured on half B; averaged both ways."""
    rng = np.random.RandomState(seed)
    pi, po = rng.permutation(len(U_in)), rng.permutation(len(U_out))
    hi, ho = len(U_in) // 2, len(U_out) // 2
    aucs = []
    for a_in, b_in, a_out, b_out in [
        (pi[:hi], pi[hi:], po[:ho], po[ho:]),
        (pi[hi:], pi[:hi], po[ho:], po[:ho]),
    ]:
        d = U_in[a_in].mean(0) - U_out[a_out].mean(0)
        d /= np.linalg.norm(d) + 1e-12
        aucs.append(auc(U_in[b_in] @ d, U_out[b_out] @ d))
    return float(np.mean(aucs))


def whiten(U, k):
    """Project out the top-k global SVD modes of the pooled cloud."""
    if k == 0:
        return U
    X = U - U.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    P = Vt[:k]
    return U - (U @ P.T) @ P


def unitize(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def selftest():
    rng = np.random.RandomState(0)
    n, d = 120, 512
    common = rng.randn(d); common /= np.linalg.norm(common)
    signal = rng.randn(d); signal -= signal @ common * common; signal /= np.linalg.norm(signal)
    A = 5 * common + 2.5 * signal[None] + rng.randn(n, d)
    B = 5 * common - 2.5 * signal[None] + rng.randn(n, d)
    U = unitize(np.vstack([A, B]))
    lab = np.arange(2 * n) < n
    bm, _, _, _ = best_mode_auc(U, lab)
    sh = split_half_contrast_auc(U[:n], U[n:])
    rng.shuffle(lab)
    bm_shuf, _, _, _ = best_mode_auc(U, lab)
    print(f"[selftest] planted: best-mode {bm:.3f} (want >0.9)  split-half {sh:.3f} (want >0.9)  "
          f"shuffled best-mode {bm_shuf:.3f} (want <~0.65)")
    assert bm > 0.9 and sh > 0.9, "selftest FAILED"
    print("[selftest] PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    meta = json.load(open(os.path.join(DIR, "episodes.json")))
    S = np.load(os.path.join(DIR, "sketches.npy"))
    norms = np.load(os.path.join(DIR, "norms.npy"))
    eps = meta["episodes"]
    tags = meta["tags"]
    idx_of = {e: i for i, e in enumerate(eps)}
    print(f"[gate] {len(eps)}/{meta['n_total_planned']} demos sketched "
          f"(fidelity corr {meta.get('fidelity_corr'):.3f})")

    def rows(tag):
        return [idx_of[e] for e in eps if tag in tags[str(e)]]

    gi, go = rows("gate_in"), rows("gate_out")
    if len(gi) < 100 or len(go) < 100:
        print(f"[gate] gate sets incomplete ({len(gi)} in / {len(go)} out) -- rerun later")
        return
    U = unitize(S)
    U_in, U_out = U[gi], U[go]
    U_gate = np.vstack([U_in, U_out])
    lab = np.arange(len(U_gate)) < len(U_in)

    report = {"n_in": len(gi), "n_out": len(go), "fidelity_corr": meta.get("fidelity_corr")}

    bm, bm_i, per_mode, var_frac = best_mode_auc(U_gate, lab)
    report["best_mode_auc"] = round(bm, 4)
    report["best_mode_index"] = bm_i
    report["best_mode_var_frac"] = round(float(var_frac[bm_i]), 4)
    print(f"[gate] best-single-SVD-mode AUC = {bm:.3f} (mode {bm_i}, "
          f"{var_frac[bm_i]*100:.1f}% var) | bar: >=0.7 workable, ~0.55 weak")

    sh = split_half_contrast_auc(U_in, U_out)
    report["split_half_contrast_auc"] = round(sh, 4)
    print(f"[gate] split-half contrast AUC = {sh:.3f}")

    for k in (10, 25, 50):
        Uw = unitize(whiten(U_gate, k))
        bmk, _, _, _ = best_mode_auc(Uw, lab)
        shk = split_half_contrast_auc(Uw[lab], Uw[~lab])
        report[f"whiten{k}_best_mode_auc"] = round(bmk, 4)
        report[f"whiten{k}_split_half_auc"] = round(shk, 4)
        print(f"[gate] whiten k={k}: best-mode {bmk:.3f}  split-half {shk:.3f}")

    rng = np.random.RandomState(1)
    lab_sh = lab.copy(); rng.shuffle(lab_sh)
    report["shuffled_split_half_auc"] = round(
        split_half_contrast_auc(U_gate[lab_sh], U_gate[~lab_sh]), 4)
    # multiple-comparisons floor of the best-mode statistic (max over 40 modes on null labels)
    bm_null = []
    for s in range(8):
        ls = lab.copy(); np.random.RandomState(100 + s).shuffle(ls)
        b, _, _, _ = best_mode_auc(U_gate, ls)
        bm_null.append(b)
    report["shuffled_best_mode_auc_mean"] = round(float(np.mean(bm_null)), 4)
    report["shuffled_best_mode_auc_max"] = round(float(np.max(bm_null)), 4)
    print(f"[gate] best-mode NULL floor (8 shuffles): mean {np.mean(bm_null):.3f} "
          f"max {np.max(bm_null):.3f} -- real best-mode must clear this")
    rd = rng.randn(U.shape[1]); rd /= np.linalg.norm(rd)
    a_rd = auc(U_in @ rd, U_out @ rd)
    report["random_direction_auc"] = round(max(a_rd, 1 - a_rd), 4)
    print(f"[gate] controls: shuffled split-half {report['shuffled_split_half_auc']:.3f}  "
          f"random-direction {report['random_direction_auc']:.3f}")

    gmean = U_gate.mean(0); gmean /= np.linalg.norm(gmean) + 1e-12
    report["common_mode_cos_in"] = round(float((U_in @ gmean).mean()), 4)
    report["common_mode_cos_out"] = round(float((U_out @ gmean).mean()), 4)
    report["norm_mean_in"] = round(float(norms[gi].mean()), 4)
    report["norm_mean_out"] = round(float(norms[go].mean()), 4)
    print(f"[gate] common-mode cos: in {report['common_mode_cos_in']:.3f} / out "
          f"{report['common_mode_cos_out']:.3f} | grad-norm mean: in {report['norm_mean_in']:.3f} "
          f"/ out {report['norm_mean_out']:.3f}")

    # region contrast direction from the FULL gate sets (for scoring pull sets)
    d_region = U_in.mean(0) - U_out.mean(0)
    d_region /= np.linalg.norm(d_region) + 1e-12

    # retention / base-competence direction g_R from the D0 sample (if sketched)
    d0 = rows("d0_sample")
    g_R = None
    if len(d0) >= 100:
        g_R = U[d0].mean(0); g_R /= np.linalg.norm(g_R) + 1e-12
        report["gR_n"] = len(d0)
        report["cos_dregion_gR"] = round(float(d_region @ g_R), 4)
        print(f"[gR] D0 base-competence direction from n={len(d0)}; "
              f"cos(d_region, g_R) = {report['cos_dregion_gR']:+.3f}")

    pulls_done = {}
    for ps in PULL_SETS:
        r = rows(ps)
        if len(r) < 150:
            continue
        Up = U[r]
        pulls_done[ps] = {
            "n": len(r),
            "mean_cos_region_dir": round(float((Up @ d_region).mean()), 4),
            "common_mode_cos": round(float((Up @ gmean).mean()), 4),
            "self_sim": round(float((Up @ Up.T)[np.triu_indices(len(r), 1)].mean()), 4),
            "norm_mean": round(float(norms[r].mean()), 4),
        }
        if g_R is not None:
            cg = Up @ g_R
            pulls_done[ps]["mean_cos_gR"] = round(float(cg.mean()), 4)
            pulls_done[ps]["frac_neg_gR"] = round(float((cg < 0).mean()), 4)
    if pulls_done:
        print("\n[pulls] per-treatment-set stats (region dir fit on gate sets):")
        for ps, st in pulls_done.items():
            print(f"  {ps:28s} n={st['n']:3d} cos(region)={st['mean_cos_region_dir']:+.4f} "
                  f"common={st['common_mode_cos']:.3f} selfsim={st['self_sim']:.3f} "
                  f"|g|={st['norm_mean']:.3f}")
    report["pull_sets"] = pulls_done

    json.dump(report, open(OUT, "w"), indent=1)
    print(f"\n[gate] wrote {OUT}")


if __name__ == "__main__":
    main()
