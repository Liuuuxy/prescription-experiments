"""CIFAR encoding-gate mirror: the exact Q0 statistics, computed on the CIFAR sandbox.

Persists the numbers used in the CIFAR-vs-robot gate comparison (previously computed
ad hoc in-session and therefore unverifiable). Uses the SAME statistics code as the
robot gate (gradient_analysis/analyze_gate.py) on the xgradtest archives:
  per-image gradients (ResNet18-GN, last block + head, exact per-sample via torch.func),
  sketched to 2048-d, logged at ckpt step 6000 -> xgradtest/gradlog/cand_raw.npy
  ground truth: pool_labels < 20  == the 20 artificially-rare ("weak region") classes
  gfail.npy = mean gradient of the rare-class TEST set (the "fix the failure" direction)

Statistics (all mirrored from the robot Q0 gate):
  best-single-SVD-mode AUC + its label-shuffle null floor   (unsupervised)
  split-half contrast AUC, 120 vs 120                       (supervised, held out)
  full-pool cosine-to-g_fail AUC                            (target direction = LESS score)

Run: /data/xinyua11/conda/envs/robocasa/bin/python cifar_experiments/robot_mirrors/cifar_gate.py
Writes cifar_experiments/robot_mirrors/cifar_gate_report.json
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/data/xinyua11/robocasa/gradient_analysis")
from analyze_gate import auc, best_mode_auc, split_half_contrast_auc, unitize  # noqa: E402

G = "/data/xinyua11/xgradtest/gradlog"
OUT = "/data/xinyua11/robocasa/cifar_experiments/robot_mirrors/cifar_gate_report.json"
CKPT = 6000
N_GATE = 120          # same per-side sample size as the robot Q0 gate
SEED = 20260730       # same seed discipline as the robot gate


def main():
    meta = json.load(open(f"{G}/meta.json"))
    ti = meta["ckpts"].index(CKPT)
    raw = np.asarray(np.load(f"{G}/cand_raw.npy", mmap_mode="r")[ti], dtype=np.float64)
    y = np.load(f"{G}/pool_labels.npy")
    gfail = np.load(f"{G}/gfail.npy")[ti].astype(np.float64)
    is_rare = y < meta["n_rare"]
    U = unitize(raw)
    print(f"[cifar] pool {len(y)} images ({int(is_rare.sum())} rare) @ ckpt {CKPT}")

    rng = np.random.RandomState(SEED)
    gi = rng.choice(np.where(is_rare)[0], N_GATE, replace=False)
    go = rng.choice(np.where(~is_rare)[0], N_GATE, replace=False)
    Ug = np.vstack([U[gi], U[go]])
    lab = np.arange(2 * N_GATE) < N_GATE

    bm, bmi, _, _ = best_mode_auc(Ug, lab)
    nulls = []
    for s in range(8):
        ls = lab.copy(); np.random.RandomState(100 + s).shuffle(ls)
        nulls.append(best_mode_auc(Ug, ls)[0])
    sh = split_half_contrast_auc(U[gi], U[go])
    s_gf = U @ (gfail / (np.linalg.norm(gfail) + 1e-12))
    a_gf = auc(s_gf[is_rare], s_gf[~is_rare])

    rep = {
        "checkpoint": CKPT, "n_pool": int(len(y)), "n_rare_pool": int(is_rare.sum()),
        "gate_n_per_side": N_GATE, "seed": SEED,
        "best_mode_auc": round(float(bm), 4), "best_mode_index": int(bmi),
        "best_mode_null_mean": round(float(np.mean(nulls)), 4),
        "best_mode_null_max": round(float(np.max(nulls)), 4),
        "split_half_contrast_auc": round(float(sh), 4),
        "full_pool_cos_to_gfail_auc": round(float(a_gf), 4),
        "robot_reference": {"best_mode_auc": 0.577, "best_mode_null_mean": 0.598,
                            "split_half_contrast_auc": 0.572,
                            "source": "gradient_analysis/q0_gate_report.json"},
    }
    print(f"  best-single-SVD-mode AUC  {bm:.3f}  [null floor {np.mean(nulls):.3f}]   (robot 0.577 [0.598])")
    print(f"  split-half contrast AUC   {sh:.3f}                      (robot 0.572)")
    print(f"  full-pool cos-to-g_fail   {a_gf:.3f}                      (robot ~0.60)")
    json.dump(rep, open(OUT, "w"), indent=1)
    print(f"[cifar] wrote {OUT}")


if __name__ == "__main__":
    main()
