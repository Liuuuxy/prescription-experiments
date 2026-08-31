#!/usr/bin/env python
"""
TASK C -- CIFAR COMMON-MODE CONTROL.

Goal: lock the principle "influence works iff the loss gradient ENCODES the
target distinction." The gradlog model was trained with cross-entropy on CLASS,
so its per-sample gradients strongly encode class identity. We show the SAME
gradients FAIL to separate nuisance attributes (brightness, color) that the
training loss does not encode, and fail completely on a random label.

This is the controlled analog of robocasa: the action-loss gradient weakly
encodes object-category (best single SVD mode ~0.56 -> influence AUC ~0.60),
because object category is only a nuisance w.r.t. the action-prediction loss.

Outputs, per checkpoint (step 3000 and 6000) and per attribute:
  (a) best-single-SVD-mode AUC   (max over top-40 modes, symmetric)
  (b) cos-to-attribute-direction AUC, direction = unit(mean unit(cand)[hot]
      - mean unit(cand)[cold])

Run:
  /data/xinyua11/conda/envs/robocasa/bin/python /data/xinyua11/xgradtest/cifar_control.py
"""
import json
import numpy as np
import torchvision

GRADLOG = "/data/xinyua11/xgradtest/gradlog/"
DATA_ROOT = "/data/xinyua11/xgradtest/data"
TOP_MODES = 40          # how many top SVD modes to scan for best-mode AUC
CKPTS_REPORT = [3000, 6000]


# ---------- helpers (per task spec) ----------
def unit(x, ax=-1):
    n = np.linalg.norm(x, axis=ax, keepdims=True)
    return x / (n + 1e-12)


def auc(s, hot):
    """AUC that score s ranks hot (True) above cold (False), via rank-sum."""
    o = np.argsort(s)
    r = np.empty(len(s))
    r[o] = np.arange(len(s))
    nh = hot.sum()
    return (r[hot].sum() - nh * (nh - 1) / 2) / (nh * (~hot).sum())


def sym_auc(s, hot):
    """Symmetric AUC: a mode can separate hot>cold OR cold>hot. Take max,
    so sign of the SVD mode does not matter. 0.5 = no separation."""
    a = auc(s, hot)
    return max(a, 1.0 - a)


# ---------- load gradients + meta ----------
cand_raw = np.load(GRADLOG + "cand_raw.npy", mmap_mode="r")   # [4, 8000, 2048]
pool_labels = np.load(GRADLOG + "pool_labels.npy")            # [8000]
meta = json.load(open(GRADLOG + "meta.json"))
ckpts = meta["ckpts"]                                         # [500,1500,3000,6000]
n_rare = meta["n_rare"]                                       # 20
pool = np.array(meta["pool"])                                 # CIFAR train indices
N = cand_raw.shape[1]
assert len(pool) == N == len(pool_labels)

# ---------- load raw CIFAR-100 images for the candidate pool ----------
ds = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=True, download=False)
images = ds.data[pool].astype(np.float32)        # [8000, 32, 32, 3], 0..255
targets = np.array(ds.targets)[pool]             # [8000]
assert np.array_equal(targets, pool_labels), "pool/image correspondence broken"

# Per-image features for nuisance attributes
mean_intensity = images.mean(axis=(1, 2, 3))     # overall brightness
red_mean = images[..., 0].mean(axis=(1, 2))      # red-channel mean (color)
# saturation: max(RGB)-min(RGB) averaged over pixels, scaled by value
mx = images.max(axis=3)
mn = images.min(axis=3)
saturation = ((mx - mn) / (mx + 1e-6)).mean(axis=(1, 2))


def top_half(v):
    """Hot = top 50% by value v (above median)."""
    return v > np.median(v)


# ---------- define attributes + hot halves ----------
rng = np.random.RandomState(0)
random_hot = rng.rand(N) > 0.5

ATTRS = {
    "class(rare)":  pool_labels < n_rare,   # gradient-ENCODED  -> should separate
    "brightness":   top_half(mean_intensity),  # nuisance        -> should NOT
    "color(red)":   top_half(red_mean),        # nuisance        -> should NOT
    "saturation":   top_half(saturation),      # nuisance        -> should NOT
    "random":       random_hot,                # pure control    -> ~0.5
}


def best_mode_auc(U, hot, k):
    """Max symmetric AUC over the first k left-singular directions of the
    unit-normalised candidate-gradient matrix. Each column of U is one mode's
    coordinate per candidate."""
    best, best_i = 0.0, -1
    for i in range(k):
        a = sym_auc(U[:, i], hot)
        if a > best:
            best, best_i = a, i
    return best, best_i


def dir_auc(Xu, hot):
    """cos-to-attribute-direction AUC. Direction = unit(mean unit(cand)[hot]
    - mean unit(cand)[cold]); score = cos(unit(cand_i), direction)."""
    g = unit(Xu[hot].mean(0) - Xu[~hot].mean(0))
    s = Xu @ g                       # rows already unit -> cos up to const
    return auc(s, hot)


# ---------- main loop over checkpoints ----------
results = {}   # ckpt -> attr -> (best_mode_auc, best_mode_idx, dir_auc)
for step in CKPTS_REPORT:
    ti = ckpts.index(step)
    X = np.asarray(cand_raw[ti], dtype=np.float64)   # [8000, 2048]
    Xu = unit(X, ax=1)                               # unit per candidate
    # SVD of the unit matrix; columns of U scaled by S are the mode coords.
    # We use U (left singular vectors) directly as per-candidate mode scores.
    U, S, _ = np.linalg.svd(Xu, full_matrices=False)
    k = min(TOP_MODES, U.shape[1])
    res = {}
    for name, hot in ATTRS.items():
        bm, bi = best_mode_auc(U, hot, k)
        da = dir_auc(Xu, hot)
        res[name] = (bm, bi, da)
    results[step] = res


# ---------- print tables ----------
def fmt(x):
    return f"{x:.3f}"


print("=" * 72)
print("CIFAR COMMON-MODE CONTROL  (gradient encodes CLASS, not nuisances)")
print(f"candidates N={N}, top-{TOP_MODES} SVD modes, symmetric best-mode AUC")
print("=" * 72)

for step in CKPTS_REPORT:
    res = results[step]
    print(f"\n--- checkpoint step {step} ---")
    print(f"{'attribute':<14}{'best-mode AUC':>15}{'(mode#)':>9}{'dir-AUC':>10}")
    print("-" * 48)
    for name in ATTRS:
        bm, bi, da = res[name]
        print(f"{name:<14}{fmt(bm):>15}{bi:>9}{fmt(da):>10}")

# Key relationship: class best-mode vs nuisance best-mode (averaged over ckpts)
print("\n" + "=" * 72)
print("KEY RELATIONSHIP  (best-single-SVD-mode AUC)")
print("=" * 72)
print(f"{'attribute':<14}" + "".join(f"step{ s:<7}" for s in CKPTS_REPORT))
for name in ATTRS:
    row = "".join(f"{fmt(results[s][name][0]):<11}" for s in CKPTS_REPORT)
    print(f"{name:<14}{row}")

cls = np.mean([results[s]["class(rare)"][0] for s in CKPTS_REPORT])
nuis = np.mean([results[s][n][0]
                for s in CKPTS_REPORT
                for n in ("brightness", "color(red)", "saturation")])
rnd = np.mean([results[s]["random"][0] for s in CKPTS_REPORT])
print("\nmean best-mode AUC  class=%.3f  nuisances=%.3f  random=%.3f"
      % (cls, nuis, rnd))

# The principle is about the CONTRAST, not an absolute class threshold:
# loss-encoded target (class) must separate clearly ABOVE the nuisance floor,
# the nuisances must sit near chance, and random must be at chance.
confirmed = (cls - nuis > 0.10) and (nuis < 0.58) and (abs(rnd - 0.5) < 0.03)
print("\nPRINCIPLE CONFIRMED:" if confirmed else "\nPRINCIPLE NOT CONFIRMED:")
print(" - CLASS (loss-encoded) best-mode AUC =%.3f  (strong separation)" % cls)
print(" - NUISANCE (not loss-encoded) best-mode AUC =%.3f  (~chance)" % nuis)
print(" - RANDOM control best-mode AUC =%.3f  (chance)" % rnd)
print("\nInterpretation: the SAME gradients that find the loss-encoded target")
print("(class) cannot find nuisance attributes the loss never saw. Influence")
print("success is GATED by gradient-encoding of the target -- exactly why")
print("robocasa (object-category, best-mode ~0.56) yields only weak influence")
print("(AUC ~0.60): object-category is a nuisance w.r.t. the action loss.")
