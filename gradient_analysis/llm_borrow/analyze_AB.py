"""Analysis for TEST A (JEST/RHO-LOSS learnability) and TEST B (learning-progress reward).

Reads lossmat/*.npz produced by loss_eval.py (forward passes only) + the frozen ledger.
Prints every number with its pre-registered prediction beside it and writes results_AB.json.
"""
import itertools
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "/data/xinyua11/robocasa")

HERE = "/data/xinyua11/robocasa/gradient_analysis/llm_borrow"
LM = f"{HERE}/lossmat"
SIGMA_E = 0.0333          # measured per-pull rollout noise SD
RNG = np.random.RandomState(7)


# ------------------------------------------------------------------ small stats helpers
def load(name):
    d = np.load(f"{LM}/{name}", allow_pickle=True)
    return d["loss"].astype(np.float64), d["episodes"].astype(int)


def as_map(name):
    L, eps = load(name)
    return {int(e): L[i] for i, e in enumerate(eps)}      # ep -> [draws]


def auc(pos, neg):
    """P(pos > neg) with ties at 0.5."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def auc_null(pos, neg, n=2000):
    """Permutation null distribution of the same AUC statistic."""
    x = np.concatenate([pos, neg])
    k = len(pos)
    out = []
    for _ in range(n):
        p = RNG.permutation(x)
        out.append(auc(p[:k], p[k:]))
    out = np.asarray(out)
    return float(out.mean()), float(np.percentile(out, 97.5)), float(np.percentile(out, 2.5))


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def pearson(a, b):
    return float(np.corrcoef(np.asarray(a, float), np.asarray(b, float))[0, 1])


def perm_p(a, b, fn=spearman, n=20000):
    """Two-sided permutation p-value for a rank correlation (bootstrap CIs are anticonservative
    at these n, so every headline correlation gets one of these too)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = abs(fn(a, b))
    cnt = sum(abs(fn(a, RNG.permutation(b))) >= obs for _ in range(n))
    return float((cnt + 1) / (n + 1))


def boot_ci(a, b, fn=spearman, n=4000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = np.arange(len(a))
    vals = []
    for _ in range(n):
        s = RNG.choice(idx, len(idx), replace=True)
        if len(np.unique(a[s])) < 3 or len(np.unique(b[s])) < 3:
            continue
        vals.append(fn(a[s], b[s]))
    v = np.asarray(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def top1_accuracy(rows, score_key, higher_is_better=True, delta_key="delta"):
    """Within each round (= a shared training seed = the only valid comparison), would ranking
    the arms by `score_key` have picked the arm that actually realized the best delta?
    Chance level is reported alongside (1/n_arms averaged over rounds)."""
    byr = {}
    for r in rows:
        byr.setdefault(r["round"], []).append(r)
    hits, chance, used, detail = [], [], 0, []
    for j, rr in sorted(byr.items()):
        if len(rr) < 2:
            continue
        pick = max(rr, key=lambda z: z[score_key] if higher_is_better else -z[score_key])
        best = max(rr, key=lambda z: z[delta_key])
        hits.append(1.0 if pick["pull_id"] == best["pull_id"] else 0.0)
        chance.append(1.0 / len(rr))
        used += 1
        detail.append({"round": j, "n_arms": len(rr), "picked": pick["arm"],
                       "picked_delta_pp": pick[delta_key] * 100,
                       "best": best["arm"], "best_delta_pp": best[delta_key] * 100,
                       "regret_pp": (best[delta_key] - pick[delta_key]) * 100})
    if not hits:
        return None
    return {"rounds": used, "top1_accuracy": float(np.mean(hits)),
            "chance": float(np.mean(chance)),
            "mean_regret_pp": float(np.mean([d["regret_pp"] for d in detail])),
            "detail": detail}


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100, flush=True)


def sub(t):
    print(f"\n--- {t} " + "-" * max(0, 95 - len(t)), flush=True)


# ==================================================================== main


QUALITY_ARMS_G = {"style_hi", "style_lo", "planted_bad", "gradqual_hi", "gradqual_lo"}


def run_test_b(meta, res):
    QUALITY_ARMS = QUALITY_ARMS_G
    # ------------------------------------------------------------------ TEST B
    hdr("TEST B - learning-progress reward: held-out loss at step 5000 vs final -> realized delta?")
    print("  PRE-REGISTERED: INVALID/NULL. |Spearman| < 0.30 and within-round paired sign agreement ~0.5.")
    print("  VALID would require Spearman >= 0.50 AND sign agreement >= 0.70.")
    gb = meta["testB"]
    groupsB = {"holdout_pool": gb["holdout_pool"], "d0": gb["d0"],
               "style_hi": gb["style_hi"], "style_lo": gb["style_lo"]}

    def slice_means(fn):
        if not os.path.exists(f"{LM}/{fn}"):
            return None
        L, e = load(fn)
        m = {int(x): L[i] for i, x in enumerate(e)}
        return {g: np.stack([m[x] for x in ids if x in m]) for g, ids in groupsB.items()}

    base = slice_means("B__pi0__0.npz")
    brows = []
    for x in meta["b_pulls"]:
        pid = x["pull_id"]
        fin = slice_means(f"B__{pid}__{x['final']}.npz")
        ear = slice_means(f"B__{pid}__{x['early']}.npz") if x["early"] else None
        if fin is None:
            continue
        row = {"pull_id": pid, "arm": x["arm"], "round": x["round_j"], "seed": x["seed"],
               "delta": x["delta"], "final_step": x["final"], "recipe": "20k" if x["final"] == 19999 else "10k"}
        for g in groupsB:
            row[f"Lfin_{g}"] = float(fin[g].mean())
            if ear is not None:
                row[f"Lear_{g}"] = float(ear[g].mean())
                # paired progress (common random numbers -> per-demo, per-draw differences)
                row[f"prog_early_{g}"] = float((base[g] - ear[g]).mean())
                row[f"prog_late_{g}"] = float((ear[g] - fin[g]).mean())
                row[f"se_prog_early_{g}"] = float((base[g] - ear[g]).mean(1).std(ddof=1) /
                                                  np.sqrt(ear[g].shape[0]))
            row[f"prog_total_{g}"] = float((base[g] - fin[g]).mean())
        brows.append(row)
    brows.sort(key=lambda r: -r["delta"])

    print(f"\n  base pi_0 slice loss: " + "  ".join(f"{g}={base[g].mean():.5f}" for g in groupsB))
    print(f"\n  {'pull_id':24s} {'rec':4s} {'delta pp':>9s} | {'L5000 hold':>10s} {'Lfin hold':>10s} "
          f"{'prog_e hold':>11s} {'+-':>7s} | {'L5000 d0':>9s} {'Lfin d0':>9s} {'prog_e d0':>10s}")
    for r in brows:
        e = r.get("Lear_holdout_pool")
        print(f"  {r['pull_id']:24s} {r['recipe']:4s} {r['delta']*100:+9.2f} | "
              f"{('%10.5f' % e) if e is not None else (' ' * 10)} {r['Lfin_holdout_pool']:10.5f} "
              f"{('%+11.5f' % r['prog_early_holdout_pool']) if e is not None else (' ' * 11)} "
              f"{('%7.5f' % r['se_prog_early_holdout_pool']) if e is not None else (' ' * 7)} | "
              f"{('%9.5f' % r['Lear_d0']) if e is not None else (' ' * 9)} {r['Lfin_d0']:9.5f} "
              f"{('%+10.5f' % r['prog_early_d0']) if e is not None else (' ' * 10)}")

    res["testB"]["rows"] = brows
    res["testB"]["base_loss"] = {g: float(base[g].mean()) for g in groupsB}

    have_early = [r for r in brows if "prog_early_holdout_pool" in r]
    QUALITY_ARMS_B = QUALITY_ARMS
    setsBB = {
        "all pulls with 5000 (20k recipe)": have_early,
        "content-only arms (PRIMARY)": [r for r in have_early if r["arm"] not in QUALITY_ARMS_B],
    }
    cb = {}
    for g, rr in setsBB.items():
        if len(rr) < 5:
            continue
        y = [r["delta"] for r in rr]
        print(f"\n  [{g}] n={len(rr)}")
        for k in ["prog_early_holdout_pool", "prog_early_d0", "prog_late_holdout_pool",
                  "prog_total_holdout_pool", "Lear_holdout_pool", "Lfin_holdout_pool",
                  "Lear_d0", "Lfin_d0"]:
            x = [r[k] for r in rr]
            rs = spearman(x, y)
            lo_, hi_ = boot_ci(np.array(x), np.array(y))
            pv = perm_p(np.array(x), np.array(y))
            print(f"    Spearman({k:24s}, delta) = {rs:+.3f}  [95% CI {lo_:+.3f}, {hi_:+.3f}]  perm p={pv:.3f}")
            cb.setdefault(g, {})[k] = {"spearman": rs, "ci95": [lo_, hi_], "perm_p": pv, "n": len(rr)}
        d2 = np.array(y)
        relb = max(0.0, d2.var(ddof=1) - SIGMA_E ** 2) / d2.var(ddof=1)
        print(f"    (outcome ceiling for this group: SD(delta)={d2.std(ddof=1)*100:.2f}pp, "
              f"delta reliability {relb:.3f}, max attainable |rho| {np.sqrt(relb):.3f})")
        cb[g]["_ceiling"] = {"sd_delta_pp": float(d2.std(ddof=1) * 100),
                             "delta_reliability": float(relb),
                             "max_attainable_rho": float(np.sqrt(relb))}
    res["testB"]["correlations"] = cb
    n_tests = sum(len([k for k in v if not k.startswith("_")]) for v in cb.values())
    print(f"\n  MULTIPLICITY: {n_tests} correlations were computed above; at alpha=0.05 you expect "
          f"{0.05*n_tests:.1f} false positives by chance. Read every p against that.")
    res["testB"]["n_correlation_tests"] = n_tests

    # ---- is the reward even RESOLVABLE between two arms? (paired, common random numbers) ----
    sub("B-resolution. Can the step-5000 held-out loss even TELL TWO ARMS APART?")
    raw5 = {}
    for x in meta["b_pulls"]:
        if not x["early"]:
            continue
        s = slice_means(f"B__{x['pull_id']}__{x['early']}.npz")
        if s is not None:
            raw5[x["pull_id"]] = s["holdout_pool"]           # [n_demos, draws], same demos+CRN
    ids = sorted(raw5)
    res_pairs = []
    for a, b in itertools.combinations(ids, 2):
        d = (raw5[a] - raw5[b]).mean(1)                       # per-demo paired difference
        se = d.std(ddof=1) / np.sqrt(len(d))
        res_pairs.append({"a": a, "b": b, "diff": float(d.mean()), "se": float(se),
                          "t": float(d.mean() / se) if se > 0 else 0.0})
    tt = np.array([abs(p["t"]) for p in res_pairs])
    dd = np.array([abs(p["diff"]) for p in res_pairs])
    print(f"  {len(res_pairs)} arm pairs, PAIRED on the same 32 held-out demos and the same CRN noise draws:")
    print(f"    |L_a(5000) - L_b(5000)|: median {np.median(dd):.5f}, max {dd.max():.5f}")
    print(f"    paired SE of that difference: median {np.median([p['se'] for p in res_pairs]):.5f}")
    print(f"    pairs resolvable at |t|>2: {(tt > 2).mean()*100:.0f}%   at |t|>3: {(tt > 3).mean()*100:.0f}%")
    med_se = float(np.median([p["se"] for p in res_pairs]))
    need = 32 * (2.0 * med_se / max(np.median(dd), 1e-9)) ** 2
    print(f"  -> at this probe size (32 held-out demos x 2 noise draws) only {(tt>2).mean()*100:.0f}% of arm")
    print(f"     pairs are separated; resolving the MEDIAN arm-pair gap at |t|=2 needs ~{need:.0f} held-out")
    print(f"     demos (~{need/32:.1f}x this probe, still only ~{need*2*0.5/60:.0f} GPU-min of forward passes).")
    print(f"     So the reward is cheap to sharpen - the binding problem is on the OUTCOME side, not here.")
    res["testB"]["resolution"] = {
        "n_pairs": len(res_pairs), "median_abs_diff": float(np.median(dd)),
        "max_abs_diff": float(dd.max()),
        "median_paired_se": float(np.median([p["se"] for p in res_pairs])),
        "frac_resolvable_t2": float((tt > 2).mean()), "frac_resolvable_t3": float((tt > 3).mean())}

    sub("B-paired. Same-seed (within-round) pairs - the only valid comparison per constraint 4")
    byr = {}
    for r in have_early:
        byr.setdefault(r["round"], []).append(r)
    pb = []
    for j, rr in sorted(byr.items()):
        for a, b in itertools.combinations(rr, 2):
            pb.append({"round": j, "a": a["pull_id"], "b": b["pull_id"], "arms": [a["arm"], b["arm"]],
                       "dprog": a["prog_early_holdout_pool"] - b["prog_early_holdout_pool"],
                       "dprog_d0": a["prog_early_d0"] - b["prog_early_d0"],
                       "dD": a["delta"] - b["delta"]})
    if pb:
        for key, lbl in (("dprog", "held-out pool"), ("dprog_d0", "D0")):
            nz = [p for p in pb if p["dD"] != 0]
            ag = np.mean([np.sign(p[key]) == np.sign(p["dD"]) for p in nz])
            rs = spearman([p[key] for p in nz], [p["dD"] for p in nz])
            print(f"  {len(nz)} same-seed pairs, progress on {lbl:14s}: sign agreement = {ag:.3f} "
                  f"(chance 0.5)   Spearman = {rs:+.3f}")
            res["testB"].setdefault("paired", {})[key] = {
                "n_pairs": len(nz), "sign_agreement": float(ag), "spearman": rs}
            # split by whether the pair straddles the EXECUTION-QUALITY axis - the only axis
            # constraint 5 says the loss can see. If all the agreement lives there, the reward is
            # a quality detector, not an allocator over content.
            q = [p for p in nz if (p["arms"][0] in QUALITY_ARMS) != (p["arms"][1] in QUALITY_ARMS)
                 or (p["arms"][0] in QUALITY_ARMS and p["arms"][1] in QUALITY_ARMS)]
            c = [p for p in nz if p["arms"][0] not in QUALITY_ARMS and p["arms"][1] not in QUALITY_ARMS]
            for tag, sset in (("pairs involving a quality arm", q), ("content-vs-content pairs", c)):
                if len(sset) >= 5:
                    a2 = np.mean([np.sign(p[key]) == np.sign(p["dD"]) for p in sset])
                    print(f"      {tag:32s} n={len(sset):3d}: sign agreement = {a2:.3f}")
                    res["testB"]["paired"][f"{key}__{tag.replace(' ', '_')}"] = {
                        "n_pairs": len(sset), "sign_agreement": float(a2)}
        for key, lbl in (("prog_early_holdout_pool", "held-out pool"), ("prog_early_d0", "D0")):
            t1 = top1_accuracy(have_early, key, higher_is_better=True)
            if t1:
                print(f"  within-round top-1 pick by MOST progress on {lbl:14s}: "
                      f"{t1['top1_accuracy']:.3f} over {t1['rounds']} rounds "
                      f"(chance {t1['chance']:.3f}), mean regret {t1['mean_regret_pp']:+.2f}pp")
                res["testB"].setdefault("top1", {})[key] = t1
        print("  pair detail (round, arms, d_progress_holdout, d_delta pp):")
        for p in sorted(pb, key=lambda z: (z["round"], -abs(z["dD"])))[:24]:
            print(f"    j{p['round']} {p['arms'][0]:16s} vs {p['arms'][1]:16s} "
                  f"dprog {p['dprog']:+.5f}  dDelta {p['dD']*100:+6.2f}pp  "
                  f"{'AGREE' if np.sign(p['dprog']) == np.sign(p['dD']) else 'disagree'}")

    # quality axis with EVERY retained checkpoint as the reference (free robustness sweep)
    sub("B-bonus. The A(i) quality axis re-measured with every retained checkpoint as reference")
    hi_ids, lo_ids = gb["style_hi"], gb["style_lo"]
    aucs = []
    for x in meta["b_pulls"]:
        fn = f"B__{x['pull_id']}__{x['final']}.npz"
        s = slice_means(fn)
        if s is None:
            continue
        Sh = (base["style_hi"] - s["style_hi"]).mean(1)
        Sl = (base["style_lo"] - s["style_lo"]).mean(1)
        aucs.append(auc(Sl, Sh))
    if aucs:
        a0 = auc(base["style_lo"].mean(1), base["style_hi"].mean(1))
        print(f"  raw pi_0 loss AUC(lo>hi) on the small Test-B style slice (n={len(lo_ids)}/{len(hi_ids)}): {a0:.3f}")
        print(f"  learnability AUC(lo>hi) over {len(aucs)} different reference checkpoints: "
              f"mean {np.mean(aucs):.3f}  sd {np.std(aucs):.3f}  range [{min(aucs):.3f}, {max(aucs):.3f}]")
        res["testB"]["quality_axis_multi_reference"] = {
            "raw_pi0_auc": a0, "n_refs": len(aucs), "mean": float(np.mean(aucs)),
            "sd": float(np.std(aucs)), "min": float(min(aucs)), "max": float(max(aucs))}


def main():
    meta = json.load(open(f"{HERE}/sets_meta.json"))
    prereg = json.load(open(f"{HERE}/prereg_AB.json"))
    res = {"prereg": prereg, "testA": {}, "testB": {}}

    # ------------------------------------------------------------------ TEST A
    hdr("TEST A - JEST / RHO-LOSS learnability   score(z) = L(z; pi_0) - L(z; reference)")
    print(f"reference 1 (primary)   : {meta['ref1']['pull']} @ {meta['ref1']['step']}  "
          f"(best realized delta of any pull: +7.78pp)")
    print(f"reference 2 (robustness): {meta['ref2']['pull']} @ {meta['ref2']['step']}  "
          f"(+3.78pp, DIFFERENT seed 1004 - controls constraint 4)")
    print(f"scored demos exclude both reference draws and all of D0 "
          f"({meta['excluded_from_testA_sets']['n']} episodes) so absorption cannot manufacture the score")

    p0 = as_map("A_diag__pi0.npz")
    r1 = as_map("A_diag__ref1.npz")
    r2 = as_map("A_diag__ref2.npz")
    eps = sorted(p0)
    M0 = np.stack([p0[e] for e in eps])          # [n, draws]
    M1 = np.stack([r1[e] for e in eps])
    M2 = np.stack([r2[e] for e in eps])
    nd = M0.shape[1]

    S1 = M0 - M1                                  # paired per-draw (common random numbers)
    S2 = M0 - M2
    L0 = M0.mean(1)
    s1 = S1.mean(1)
    s2 = S2.mean(1)

    # ---- stability -------------------------------------------------------
    sub("A0. STABILITY of the estimate (how many noise/time draws are enough)")
    half = nd // 2
    def splithalf(X):
        a, b = X[:, :half].mean(1), X[:, half:].mean(1)
        r = pearson(a, b)
        return r, 2 * r / (1 + r)                 # Spearman-Brown -> reliability of the full mean
    for tag, X in (("raw loss L(pi_0)", M0), ("learnability S=L0-Lref1", S1), ("S vs ref2", S2)):
        se = X.std(1, ddof=1).mean() / np.sqrt(nd)
        between = X.mean(1).std(ddof=1)
        r, sb = splithalf(X)
        print(f"  {tag:26s}: per-demo SE(mean over {nd} draws) = {se:.5f} | between-demo SD = {between:.5f}"
              f" | SE/SD = {se/between:.3f} | split-half r = {r:.3f} -> reliability {sb:.3f}")
        res["testA"].setdefault("stability", {})[tag] = {
            "se_mean": se, "between_sd": between, "se_over_sd": se / between,
            "split_half_r": r, "reliability_spearman_brown": sb, "draws": nd}
    # numerical control: the fp32 checkpoints do not fit the 0.12 JAX memory cap, so params are
    # loaded as bfloat16 (the model's own compute dtype). float16 is a DIFFERENT rounding of the
    # same weights -> the bf16-vs-fp16 spread bounds the quantization noise in every number here.
    if os.path.exists(f"{LM}/Q__pi0__fp16.npz") and os.path.exists(f"{LM}/B__pi0__0.npz"):
        b0, _ = load("B__pi0__0.npz"); q0, _ = load("Q__pi0__fp16.npz")
        b1, _ = load("B__gradarm_b_j3__19999.npz"); q1, _ = load("Q__ref1__fp16.npz")
        sb, sf = (b0 - b1).mean(1), (q0 - q1).mean(1)
        print(f"  param-dtype control (bfloat16 vs float16 load of the SAME weights): corr(S) = "
              f"{pearson(sb, sf):.4f}, mean|dS| = {np.abs(sb-sf).mean():.5f} vs SD(S) = "
              f"{sb.std(ddof=1):.5f} -> quantization noise is "
              f"{np.abs(sb-sf).mean()/sb.std(ddof=1)*100:.1f}% of the signal SD (immaterial).")
        res["testA"]["dtype_control"] = {"corr_S": pearson(sb, sf),
                                         "mean_abs_dS": float(np.abs(sb - sf).mean()),
                                         "sd_S": float(sb.std(ddof=1))}
    print(f"  common random numbers: the SAME (noise, time, preprocess) rng per (episode, draw) is used at")
    print(f"  every checkpoint, so S is a PAIRED difference. Raw single-draw loss SD is "
          f"{M0.std(1, ddof=1).mean():.4f}; the paired difference SD is {S1.std(1, ddof=1).mean():.4f} "
          f"({M0.std(1, ddof=1).mean()/S1.std(1, ddof=1).mean():.1f}x variance reduction).")

    # ---- (i) execution-quality axis -------------------------------------
    sub("A(i). Does the score recover the EXECUTION-QUALITY axis (style_hi vs style_lo)?")
    print("  PRE-REGISTERED: raw loss AUC(style_lo > style_hi) >= 0.70 (falsifier < 0.60);")
    print("                  learnability-difference AUC >= 0.60 but WEAKER than raw loss.")
    idx = {e: i for i, e in enumerate(eps)}
    hi = [idx[e] for e in meta["style_hi"] if e in idx]
    lo = [idx[e] for e in meta["style_lo"] if e in idx]
    qa = {}
    for tag, v in (("raw L(pi_0)", L0), ("learnability S(ref1)", s1), ("learnability S(ref2)", s2),
                   ("raw L(ref1)", M1.mean(1))):
        a = auc(v[lo], v[hi])                      # P(style_lo scores higher)
        nm, nhi, nlo = auc_null(v[lo], v[hi])
        print(f"  {tag:22s}: AUC(lo>hi) = {a:.3f}   lo mean {v[lo].mean():+.5f}  hi mean {v[hi].mean():+.5f}"
              f"   perm-null 95% [{nlo:.3f},{nhi:.3f}]")
        qa[tag] = {"auc_lo_gt_hi": a, "lo_mean": float(v[lo].mean()), "hi_mean": float(v[hi].mean()),
                   "perm_null_95": [nlo, nhi]}
    res["testA"]["quality_axis"] = {"n_hi": len(hi), "n_lo": len(lo), "stats": qa}

    # ---- (ii) failure region --------------------------------------------
    sub("A(ii). Is the score correlated with the FAILURE REGION (tall-vessel grasp-fail)?")
    print("  PRE-REGISTERED: NULL. AUC in [0.45, 0.60], at/below the Q0 shuffle-null floor 0.598.")
    ri = [idx[e] for e in meta["region_in"] if e in idx]
    ro = [idx[e] for e in meta["region_out"] if e in idx]
    ra = {}
    for tag, v in (("raw L(pi_0)", L0), ("learnability S(ref1)", s1), ("learnability S(ref2)", s2)):
        a = auc(v[ri], v[ro])
        nm, nhi, nlo = auc_null(v[ri], v[ro])
        print(f"  {tag:22s}: AUC(in>out) = {a:.3f}   in mean {v[ri].mean():+.5f}  out mean {v[ro].mean():+.5f}"
              f"   perm-null 95% [{nlo:.3f},{nhi:.3f}]")
        ra[tag] = {"auc_in_gt_out": a, "in_mean": float(v[ri].mean()), "out_mean": float(v[ro].mean()),
                   "perm_null_95": [nlo, nhi]}
    res["testA"]["region_axis"] = {"n_in": len(ri), "n_out": len(ro), "stats": ra}

    # ---- (iii) pull-level -----------------------------------------------
    sub("A(iii). Does the MEAN learnability of a pull's draw predict that pull's realized delta?")
    print("  PRE-REGISTERED: NULL on content-only arms (|rho| < 0.30, CI covers 0). Any correlation")
    print("  appearing only when the execution-quality arms are pooled in is a QUALITY detector,")
    print("  reported as confounded - not evidence that learnability predicts outcome.")
    if not (os.path.exists(f"{LM}/A_pull__pi0.npz") and os.path.exists(f"{LM}/A_pull__ref1.npz")):
        print("  [A(iii) SKIPPED - pull-sample loss matrices not written yet]")
        run_test_b(meta, res)
        json.dump(res, open(f"{HERE}/results_AB.json", "w"), indent=1, default=float)
        return
    pp0 = as_map("A_pull__pi0.npz")
    pr1 = as_map("A_pull__ref1.npz")
    peps = sorted(set(pp0) & set(pr1))
    P0 = np.stack([pp0[e] for e in peps])
    P1 = np.stack([pr1[e] for e in peps])
    Sp = (P0 - P1).mean(1)
    L0p = P0.mean(1)
    pidx = {e: i for i, e in enumerate(peps)}

    bp = {x["pull_id"]: x for x in meta["b_pulls"]}
    import bandit_v1.ledger as ledger                                     # noqa: E402
    pulls_tbl = ledger.read("pulls")
    delta = {str(r.pull_id): float(r.delta) for _, r in
             pulls_tbl[pulls_tbl.status.isin(["ok", "smoke"])].iterrows()}
    arm = {str(r.pull_id): str(r.arm) for _, r in pulls_tbl.iterrows()}
    rnd = {str(r.pull_id): int(r.round_j) for _, r in pulls_tbl.iterrows()}
    dhard = {}
    for _, r in pulls_tbl[pulls_tbl.status.isin(["ok", "smoke"])].iterrows():
        try:
            dhard[str(r.pull_id)] = float(json.loads(r.delta_per_stratum_json)["hard"])
        except Exception:
            pass

    QUALITY_ARMS = QUALITY_ARMS_G
    # ABSORPTION GUARD: a demo that ref1 was trained on has near-zero loss at ref1 (Q2: own-draw
    # grad norms collapse to D0 level by step 5000), so its "learnability" is manufactured.
    # Drop those demos from every pull's mean. gradarm_b_j3 IS ref1's pull -> 25/25 dropped -> excluded.
    _ref_ids = pulls_tbl[pulls_tbl.pull_id == meta["ref1"]["pull"]].iloc[0].demo_ids
    if isinstance(_ref_ids, str):
        _ref_ids = json.loads(_ref_ids)
    absorbed = set(int(x) for x in _ref_ids)
    pool_tbl = ledger.read("pool_demos")
    absorbed |= set(int(x) for x in pool_tbl[pool_tbl.in_d0].episode_index)
    n_drop = {}
    rows = []
    for pid, samp in meta["pull_sample"].items():
        keep = [e for e in samp if e not in absorbed]
        n_drop[pid] = len(samp) - len(keep)
        ii = [pidx[e] for e in keep if e in pidx]
        if len(ii) < 10 or pid not in delta:
            continue
        rows.append({"pull_id": pid, "arm": arm[pid], "round": rnd[pid], "n": len(ii),
                     "delta": delta[pid], "delta_hard": dhard.get(pid, float("nan")),
                     "mean_S": float(Sp[ii].mean()), "mean_L0": float(L0p[ii].mean()),
                     "se_S": float(Sp[ii].std(ddof=1) / np.sqrt(len(ii)))})
    rows.sort(key=lambda r: -r["delta"])
    print(f"\n  absorption guard: dropped {sum(n_drop.values())} sampled demos that ref1 trained on "
          f"(or that are in D0); pulls excluded for <10 usable demos: "
          f"{[p for p, k in n_drop.items() if 25 - k < 10]}")
    print(f"\n  {'pull_id':24s} {'arm':16s} {'n':>3s} {'delta pp':>9s} {'mean_S':>10s} {'+-':>7s} {'mean_L0':>9s}")
    for r in rows:
        print(f"  {r['pull_id']:24s} {r['arm']:16s} {r['n']:3d} {r['delta']*100:+9.2f} "
              f"{r['mean_S']:+10.5f} {r['se_S']:7.5f} {r['mean_L0']:9.5f}")

    # measurement reliability of the pull-mean itself (25 of 200 sampled)
    n_per = np.mean([r["n"] for r in rows])
    # WITHIN-pull SE (the right one): the global per-demo SD mixes in between-arm variation and
    # would overstate the sampling noise. Each row's se_S is already the within-pull SE.
    se_pullmean = float(np.mean([r["se_S"] for r in rows]) * np.sqrt(1 - n_per / 200))
    between_pull = np.std([r["mean_S"] for r in rows], ddof=1)
    rel_S = max(0.0, 1 - se_pullmean ** 2 / between_pull ** 2)
    print(f"\n  pull-mean measurement noise: mean within-pull SE = "
          f"{np.mean([r['se_S'] for r in rows]):.5f} at n={n_per:.0f}/200 sampled "
          f"(finite-population corrected: {se_pullmean:.5f})")
    print(f"  -> observed between-pull SD of mean_S = {between_pull:.5f}"
          f"  => reliability of the pull-level learnability score = {rel_S:.3f}")
    print(f"     (global per-demo SD(S) = {Sp.std(ddof=1):.5f}, i.e. most of the per-demo spread is "
          f"WITHIN pulls, not between them)")

    # the ceiling imposed by the ROLLOUT noise floor on the outcome side
    d = np.array([r["delta"] for r in rows])
    var_true = max(0.0, d.var(ddof=1) - SIGMA_E ** 2)
    rel_delta = var_true / d.var(ddof=1)
    print(f"  outcome-side ceiling: SD(delta) over these {len(rows)} pulls = {d.std(ddof=1)*100:.2f}pp, "
          f"sigma_e = {SIGMA_E*100:.2f}pp")
    print(f"  -> reliability of the DELTA measurements = {rel_delta:.3f}; the maximum |rho| ANY predictor")
    print(f"     can attain against these deltas is sqrt(reliability) = {np.sqrt(rel_delta):.3f}")

    groups = {
        "all pulls": rows,
        "content-only arms (PRIMARY, pre-registered)": [r for r in rows if r["arm"] not in QUALITY_ARMS],
        "execution-quality arms only": [r for r in rows if r["arm"] in QUALITY_ARMS],
        "20k-recipe pulls only": [r for r in rows if bp.get(r["pull_id"], {}).get("final") == 19999],
    }
    corr = {}
    for g, rr in groups.items():
        if len(rr) < 5:
            continue
        x = [r["mean_S"] for r in rr]
        y = [r["delta"] for r in rr]
        rs, rp = spearman(x, y), pearson(x, y)
        lo_, hi_ = boot_ci(np.array(x), np.array(y))
        x2 = [r["mean_L0"] for r in rr]
        yh = [r["delta_hard"] for r in rr]
        ok = ~np.isnan(np.asarray(yh, float))
        rh = spearman(np.asarray(x)[ok], np.asarray(yh, float)[ok]) if ok.sum() >= 5 else float("nan")
        pv = perm_p(np.array(x), np.array(y))
        yv = np.asarray(y, float)
        relg = max(0.0, yv.var(ddof=1) - SIGMA_E ** 2) / yv.var(ddof=1)
        ceil = float(np.sqrt(relg * rel_S))
        dis = rs / ceil if ceil > 0.05 else float("nan")
        print(f"  [{g}] n={len(rr):2d}  Spearman(mean_S, delta) = {rs:+.3f}  [95% CI {lo_:+.3f}, {hi_:+.3f}]"
              f" perm p={pv:.3f}  Pearson {rp:+.3f}  | Spearman(mean_L0, delta) = {spearman(x2, y):+.3f}"
              f" | Spearman(mean_S, delta_HARD) = {rh:+.3f}")
        print(f"       ceiling for this group: SD(delta)={yv.std(ddof=1)*100:.2f}pp -> delta reliability "
              f"{relg:.3f}; max attainable |rho| = sqrt(rel_delta*rel_S) = {ceil:.3f}; "
              f"disattenuated rho = {dis:+.3f}")
        corr[g] = {"n": len(rr), "spearman_S": rs, "ci95": [lo_, hi_], "perm_p": pv, "pearson_S": rp,
                   "spearman_L0": spearman(x2, y), "spearman_S_vs_delta_hard": rh,
                   "delta_reliability": float(relg), "max_attainable_rho": ceil,
                   "disattenuated_rho": float(dis) if dis == dis else None}
    res["testA"]["pull_level"] = {
        "rows": rows, "correlations": corr,
        "pull_mean_reliability": float(max(0.0, 1 - se_pullmean**2/between_pull**2)),
        "delta_measurement_reliability": float(rel_delta),
        "max_attainable_rho": float(np.sqrt(rel_delta)),
        "sd_delta_pp": float(d.std(ddof=1) * 100), "sigma_e_pp": SIGMA_E * 100}

    # within-round paired (constraint 4: only same-seed comparisons are valid)
    sub("A(iii)-paired. Same-seed (within-round) pairs only - the only valid comparison per constraint 4")
    pairs = []
    byr = {}
    for r in rows:
        byr.setdefault(r["round"], []).append(r)
    for j, rr in sorted(byr.items()):
        for a, b in itertools.combinations(rr, 2):
            pairs.append({"round": j, "a": a["pull_id"], "b": b["pull_id"],
                          "dS": a["mean_S"] - b["mean_S"], "dD": a["delta"] - b["delta"]})
    if pairs:
        ag = np.mean([np.sign(p["dS"]) == np.sign(p["dD"]) for p in pairs])
        rs = spearman([p["dS"] for p in pairs], [p["dD"] for p in pairs])
        print(f"  {len(pairs)} same-seed pairs across {len(byr)} rounds: sign agreement = {ag:.3f} "
              f"(chance 0.5), Spearman(dS, dDelta) = {rs:+.3f}")
        res["testA"]["paired"] = {"n_pairs": len(pairs), "sign_agreement": float(ag), "spearman": rs}
    for hib in (True, False):
        t1 = top1_accuracy(rows, "mean_S", higher_is_better=hib)
        if t1:
            print(f"  within-round top-1 pick by {'HIGHEST' if hib else 'LOWEST'} mean learnability: "
                  f"{t1['top1_accuracy']:.3f} over {t1['rounds']} rounds (chance {t1['chance']:.3f}), "
                  f"mean regret {t1['mean_regret_pp']:+.2f}pp")
            res["testA"].setdefault("top1", {})["high" if hib else "low"] = t1

    run_test_b(meta, res)
    json.dump(res, open(f"{HERE}/results_AB.json", "w"), indent=1, default=float)
    print(f"\nwrote {HERE}/results_AB.json")


if __name__ == "__main__":
    main()
