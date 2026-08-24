"""Analysis for grad_gate_traj.py: does region-encoding emerge along the fine-tune?

Per (trajectory, checkpoint): best-single-SVD-mode AUC (with its shuffle-null floor)
and split-half contrast AUC on the region gate sets. Reference: at pi_0 (the shared
trajectory start) Q0 measured best-mode 0.577 (null 0.598) and split-half 0.572.
Decision rule (pre-registered in chat, 2026-08-11): split-half >= 0.7 at any late
checkpoint of the tall trajectory = encoding emerges, trajectory-LESS has signal;
otherwise LESS is closed and mid-step checkpoints carry no further gradient value.

Run: /data/xinyua11/conda/envs/robocasa/bin/python gradient_analysis/analyze_gate_traj.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/data/xinyua11/robocasa/gradient_analysis")
from analyze_gate import best_mode_auc, split_half_contrast_auc, unitize  # noqa: E402

OUT = "/data/xinyua11/robocasa/gradient_analysis/gate_traj"


def main():
    print("pi_0 reference (Q0): best-mode 0.577 [null 0.598], split-half 0.572\n")
    report = {}
    for traj in sorted(os.listdir(OUT)):
        d = os.path.join(OUT, traj)
        meta = json.load(open(os.path.join(d, "meta.json")))
        labs = np.array(meta["labels"]) == "in"
        for step in meta["steps"]:
            p = os.path.join(d, f"{step}_sketches.npy")
            if not os.path.exists(p):
                continue
            U = unitize(np.load(p).astype(np.float64))
            bm, bmi, _, _ = best_mode_auc(U, labs)
            sh = split_half_contrast_auc(U[labs], U[~labs])
            nulls = []
            for s in range(8):
                ls = labs.copy(); np.random.RandomState(100 + s).shuffle(ls)
                b, _, _, _ = best_mode_auc(U, ls)
                nulls.append(b)
            nrm = np.load(os.path.join(d, f"{step}_norms.npy"))
            report[f"{traj}@{step}"] = {"best_mode": round(bm, 4), "null_mean": round(float(np.mean(nulls)), 4),
                                        "split_half": round(sh, 4),
                                        "norm_in": round(float(nrm[labs].mean()), 3),
                                        "norm_out": round(float(nrm[~labs].mean()), 3)}
            print(f"{traj:>28s} @{step:>5d}: best-mode {bm:.3f} [null {np.mean(nulls):.3f}]  "
                  f"split-half {sh:.3f}  |g| in/out {nrm[labs].mean():.2f}/{nrm[~labs].mean():.2f}")
    json.dump(report, open(f"{OUT}/report.json", "w"), indent=1)
    print(f"\nwrote {OUT}/report.json")


if __name__ == "__main__":
    main()
