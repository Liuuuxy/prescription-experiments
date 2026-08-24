"""Build the demo sets + checkpoint jobs for TEST A (learnability) and TEST B (learning-progress).

Writes (into gradient_analysis/llm_borrow/):
  setsA.json / jobsA.json   -- Test A: pi_0 + 2 reference checkpoints
  setsB.json / jobsB.json   -- Test B: fixed held-out slice at 5000 and final for every pull
  sets_meta.json            -- provenance + all the bookkeeping the analysis needs

Reference-checkpoint choice for Test A (pre-registered in prereg_AB.json, justified here):
  ref1 = gradarm_b_j3 @ 19999  -- the STRONGEST retained checkpoint by the only metric that
         matters on this stack: realized closed-loop delta (+7.78pp, the best pull ever run).
         RHO-LOSS/JEST want a reference that is genuinely better than the learner; "better"
         here has to mean deployed success, not training loss.
  ref2 = mid_band_j4 @ 19999   -- robustness: a different SEED (1004 vs 1003) and a different
         arm, still clearly positive (+3.78pp). Constraint 4 (the seed steers the update) means
         a single-reference result could be a seed artifact; ref2 is the control for that.
  Every demo in either reference pull's 200-demo draw, and every D0 demo, is EXCLUDED from the
  scored sets -- otherwise the reference has memorised the demo (Q2: own-draw grad norms collapse
  to D0 level by step 5000) and the "learnability" score is manufactured by absorption.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/data/xinyua11/robocasa")
from bandit_v1 import ledger  # noqa: E402

HERE = "/data/xinyua11/robocasa/gradient_analysis/llm_borrow"
GA = "/data/xinyua11/robocasa/gradient_analysis"
CK = "/data/xinyua11/openpi/checkpoints"
PI0 = f"{CK}/pi0_ppc2sink_pi0base/pi0_v1/19999/params"
REF1 = ("gradarm_b_j3", 19999, f"{CK}/pi0_ppc2sink_bandit_b/gradarm_b_j3/19999/params")
REF2 = ("mid_band_j4", 19999, f"{CK}/pi0_ppc2sink_bandit_b/mid_band_j4/19999/params")

N_STYLE = 70      # per side (hi / lo)
N_REGION = 70     # per side (in / out)
N_PER_PULL = 25   # demos sampled from each pull's 200-demo draw for test A(iii)
N_HOLDOUT = 32    # Test B: pool demos never trained by ANY pull and not in D0
N_D0 = 16         # Test B: D0 demos (in every pull's training mix)
N_STYLE_B = 8     # Test B slice: free multi-reference check on the quality axis
RNG = np.random.RandomState(20260814)


def steps_present(root):
    return sorted(int(x) for x in os.listdir(root) if x.isdigit()) if os.path.isdir(root) else []


def main():
    p = ledger.read("pulls")
    done = p[p.status.isin(["ok", "smoke"])].copy()
    pool = ledger.read("pool_demos")
    d0 = set(int(x) for x in pool[pool.in_d0].episode_index)

    draws = {}
    for _, r in done.iterrows():
        ids = r.demo_ids
        if isinstance(ids, str):
            ids = json.loads(ids)
        draws[str(r.pull_id)] = [int(x) for x in ids]
    ever_trained = set().union(*draws.values()) | d0

    ref_draw = set(draws[REF1[0]]) | set(draws[REF2[0]]) | d0   # excluded from Test A scored sets

    style = json.load(open(f"{GA}/style_assignment.json"))
    dl = json.load(open(f"{GA}/demo_lists.json"))

    def pick(cands, n):
        c = sorted(set(int(x) for x in cands) - ref_draw)
        return sorted(int(x) for x in RNG.choice(c, size=min(n, len(c)), replace=False))

    style_hi = pick(style["style_hi"], N_STYLE)
    style_lo = pick(style["style_lo"], N_STYLE)
    reg_in = pick(dl["gate_in_region"], N_REGION)
    reg_out = pick(dl["gate_out_region"], N_REGION)

    # per-pull subsample for A(iii)
    # null arms add no demos (empty draw) -> no learnability score to compute; they stay in
    # Test B (they still have checkpoints) but are excluded from A(iii) by construction.
    pull_sample = {pid: sorted(int(x) for x in RNG.choice(ids, size=min(N_PER_PULL, len(ids)),
                                                          replace=False))
                   for pid, ids in draws.items() if len(ids) > 0}
    pull_union = sorted(set().union(*pull_sample.values()))

    setsA = {"diag": sorted(set(style_hi) | set(style_lo) | set(reg_in) | set(reg_out)),
             "pullsample": pull_union}

    # ---- Test B fixed slice ----
    never = sorted(set(int(x) for x in pool.episode_index) - ever_trained)
    holdout = sorted(int(x) for x in RNG.choice(never, size=N_HOLDOUT, replace=False))
    d0_slice = sorted(int(x) for x in RNG.choice(sorted(d0), size=N_D0, replace=False))
    sB_hi = pick(style["style_hi"], N_STYLE_B)
    sB_lo = pick(style["style_lo"], N_STYLE_B)
    sliceB = sorted(set(holdout) | set(d0_slice) | set(sB_hi) | set(sB_lo))
    setsB = {"slice": sliceB}

    jobsA = [{"ckpt": PI0, "demos": "diag", "out": "A_diag__pi0.npz"},
             {"ckpt": REF1[2], "demos": "diag", "out": "A_diag__ref1.npz"},
             {"ckpt": REF2[2], "demos": "diag", "out": "A_diag__ref2.npz"},
             {"ckpt": PI0, "demos": "pullsample", "out": "A_pull__pi0.npz"},
             {"ckpt": REF1[2], "demos": "pullsample", "out": "A_pull__ref1.npz"}]

    jobsB = [{"ckpt": PI0, "demos": "slice", "out": "B__pi0__0.npz"}]
    b_pulls = []
    for _, r in done.iterrows():
        pid = str(r.pull_id)
        root = json.loads(r.training_artifacts_json)["ckpt_root"] if isinstance(
            r.training_artifacts_json, str) else r.training_artifacts_json["ckpt_root"]
        st = steps_present(root)
        if not st:
            continue
        final = max(st)
        early = 5000 if 5000 in st else None
        b_pulls.append({"pull_id": pid, "arm": str(r.arm), "round_j": int(r.round_j),
                        "seed": int(r.seed), "delta": float(r.delta), "root": root,
                        "steps": st, "final": final, "early": early})
        if early:
            jobsB.append({"ckpt": f"{root}/{early}/params", "demos": "slice",
                          "out": f"B__{pid}__{early}.npz"})
        jobsB.append({"ckpt": f"{root}/{final}/params", "demos": "slice",
                      "out": f"B__{pid}__{final}.npz"})

    json.dump(setsA, open(f"{HERE}/setsA.json", "w"))
    json.dump(setsB, open(f"{HERE}/setsB.json", "w"))
    json.dump(jobsA, open(f"{HERE}/jobsA.json", "w"), indent=1)
    json.dump(jobsB, open(f"{HERE}/jobsB.json", "w"), indent=1)
    meta = {
        "pi0": PI0, "ref1": {"pull": REF1[0], "step": REF1[1], "ckpt": REF1[2]},
        "ref2": {"pull": REF2[0], "step": REF2[1], "ckpt": REF2[2]},
        "excluded_from_testA_sets": {"n": len(ref_draw), "reason": "ref-pull draws + D0 (absorption)"},
        "style_hi": style_hi, "style_lo": style_lo,
        "region_in": reg_in, "region_out": reg_out,
        "pull_sample": pull_sample,
        "pull_draws_full": {k: len(v) for k, v in draws.items()},
        "testB": {"holdout_pool": holdout, "d0": d0_slice, "style_hi": sB_hi, "style_lo": sB_lo,
                  "n_never_trained_pool": len(never)},
        "b_pulls": b_pulls,
        "n_pull_sample_union": len(pull_union),
    }
    json.dump(meta, open(f"{HERE}/sets_meta.json", "w"), indent=1)
    print(f"setsA: diag={len(setsA['diag'])} pullsample={len(setsA['pullsample'])} "
          f"(union over {len(pull_sample)} pulls x {N_PER_PULL})")
    print(f"setsB: slice={len(sliceB)} (holdout {len(holdout)} + d0 {len(d0_slice)} + style {len(sB_hi)+len(sB_lo)})")
    print(f"jobsA={len(jobsA)} jobsB={len(jobsB)} | pulls with ckpts={len(b_pulls)} "
          f"(with 5000: {sum(1 for x in b_pulls if x['early'])})")
    print(f"never-trained pool demos available: {len(never)}")


if __name__ == "__main__":
    main()
