"""Accuracy-region sandbox (advisor request, 2026-08-03): CIFAR-100 analogue of
the data-prescription experiment, run on CPU while the H100s train robots.

Question: does WHERE a model sits on the accuracy ladder (>90 easy / 70-90
medium / <70 hard, Ransalu's bands) determine what added data buys? Mirrors the
robot setup: imbalanced base training creates a per-class accuracy spread; arms
add B images targeted at one band vs random vs null; fine-tune from the same
base checkpoint with a ~70/30 old/new mix (the LLM-practice rule; the robot
recipe's D0+B is 67/33); measure per-band deltas with 6 seeds per arm.

Also: a long-fine-tune arm (15 epochs, dense eval) to test "fine-tuning a good
model for many iterations hurts", and per-sample base-model gradients saved
LOSSLESSLY for the whole pool (last-layer CE grad = feat (x) (p - y); we save
the factors: feats 512 + errs 100 per sample) for gradient-gate/cluster
analysis later.

CPU-only by design. Outputs in this directory: base_ckpt.pt, results.parquet,
longft_curve.parquet, pool_grads.npz, report.json.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms as T

torch.set_num_threads(int(os.environ.get("SANDBOX_THREADS", "24")))
DEV = "cpu"
OUT = "/data/xinyua11/robocasa/cifar_experiments/sandbox_regions"
DATA = "/data/xinyua11/xgradtest/data"
SEED = 20260803
BASE_COUNTS = [400, 200, 100, 50, 25]     # tiled over 100 classes -> acc spread
B = 2000
OLD_MIX = 4700                             # old:new = 7000:3000 = 70/30
ARMS = ["hard", "med", "easy", "random", "null"]
N_SEEDS = 6
BANDS = {"easy": (0.90, 1.01), "med": (0.70, 0.90), "hard": (-0.01, 0.70)}


def log(*a):
    print(f"[sandbox {time.strftime('%H:%M:%S')}]", *a, flush=True)


def make_model():
    m = torchvision.models.resnet18(num_classes=100)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)   # CIFAR stem
    m.maxpool = nn.Identity()
    return m


MEAN, STD = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
AUG = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                 T.ToTensor(), T.Normalize(MEAN, STD)])
PLAIN = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])


class Sub(Dataset):
    def __init__(self, base, idxs, tf):
        self.b, self.i, self.tf = base, list(idxs), tf
    def __len__(self):
        return len(self.i)
    def __getitem__(self, k):
        img, y = self.b.data[self.i[k]], self.b.targets[self.i[k]]
        from PIL import Image
        return self.tf(Image.fromarray(img)), y


def train(model, ds, epochs, lr, seed, eval_every=None, eval_fn=None):
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(ds, batch_size=256, shuffle=True, generator=g, num_workers=0)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(dl))
    curve = []
    model.train()
    for ep in range(epochs):
        for x, y in dl:
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step(); sched.step()
        if eval_every and (ep + 1) % eval_every == 0:
            curve.append((ep + 1, eval_fn(model)))
            model.train()
    return curve


@torch.no_grad()
def per_class_acc(model, test):
    model.eval()
    dl = DataLoader(test, batch_size=512, num_workers=0)
    hit = np.zeros(100); tot = np.zeros(100)
    for x, y in dl:
        p = model(x).argmax(1)
        for c in range(100):
            m = y == c
            tot[c] += int(m.sum()); hit[c] += int((p[m] == c).sum())
    return hit / np.maximum(tot, 1)


def band_of(acc):
    return {c: next(b for b, (lo, hi) in BANDS.items() if lo <= acc[c] < hi)
            for c in range(100)}


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    tr = torchvision.datasets.CIFAR100(DATA, train=True, download=False)
    te_ds = torchvision.datasets.CIFAR100(DATA, train=False, download=False,
                                          transform=PLAIN)
    targets = np.array(tr.targets)

    counts = {c: BASE_COUNTS[c % len(BASE_COUNTS)] for c in range(100)}
    base_idx, pool_idx = [], []
    for c in range(100):
        idxs = rng.permutation(np.where(targets == c)[0])
        base_idx += list(idxs[:counts[c]]); pool_idx += list(idxs[counts[c]:])
    log(f"base={len(base_idx)} pool={len(pool_idx)}")

    torch.manual_seed(SEED)
    base = make_model()
    t0 = time.time()
    train(base, Sub(tr, base_idx, AUG), epochs=45, lr=0.1, seed=SEED)
    torch.save(base.state_dict(), f"{OUT}/base_ckpt.pt")
    acc0 = per_class_acc(base, te_ds)
    bands = band_of(acc0)
    bcnt = {b: sum(1 for c in bands.values() if c == b) for b in BANDS}
    bacc0 = {b: float(np.mean([acc0[c] for c in range(100) if bands[c] == b]))
             for b in BANDS}
    log(f"base trained in {(time.time()-t0)/60:.1f} min; overall {acc0.mean():.3f}; "
        f"bands {bcnt} band-acc {bacc0}")

    pool = np.array(pool_idx)
    pool_by_band = {b: pool[[bands[targets[i]] == b for i in pool]] for b in BANDS}
    old_fixed = rng.choice(base_idx, OLD_MIX, replace=False)

    rows = []
    for arm in ARMS:
        for s in range(N_SEEDS):
            r = np.random.default_rng(1000 + 61 * s + hash(arm) % 997)
            if arm == "null":
                add = np.array([], dtype=int)
            elif arm == "random":
                add = r.choice(pool, B, replace=False)
            else:
                src = pool_by_band[arm]
                add = r.choice(src, min(B, len(src)), replace=False)
            m = make_model(); m.load_state_dict(torch.load(f"{OUT}/base_ckpt.pt"))
            t0 = time.time()
            train(m, Sub(tr, list(old_fixed) + list(add), AUG),
                  epochs=3, lr=0.01, seed=2000 + s)
            acc = per_class_acc(m, te_ds)
            d = {b: float(np.mean([acc[c] - acc0[c] for c in range(100)
                                   if bands[c] == b])) for b in BANDS}
            rows.append({"arm": arm, "seed": s, "n_add": len(add),
                         "overall": float(acc.mean()),
                         "d_overall": float(acc.mean() - acc0.mean()),
                         **{f"d_{b}": d[b] for b in BANDS},
                         "acc_json": json.dumps([round(float(a), 4) for a in acc])})
            log(f"ARM DONE {arm} seed={s} ({(time.time()-t0)/60:.1f}min) "
                f"d_overall={rows[-1]['d_overall']:+.4f} "
                f"d_hard={d['hard']:+.4f} d_med={d['med']:+.4f} d_easy={d['easy']:+.4f}")
            pd.DataFrame(rows).to_parquet(f"{OUT}/results.parquet")

    # long fine-tune: does a good model degrade with many iterations?
    m = make_model(); m.load_state_dict(torch.load(f"{OUT}/base_ckpt.pt"))
    r = np.random.default_rng(999)
    add = r.choice(pool, B, replace=False)
    curve = train(m, Sub(tr, list(old_fixed) + list(add), AUG), epochs=15, lr=0.01,
                  seed=777, eval_every=1,
                  eval_fn=lambda mm: {b: float(np.mean([per_class_acc(mm, te_ds)[c]
                        for c in range(100) if bands[c] == b])) for b in BANDS})
    pd.DataFrame([{"epoch": e, **v} for e, v in curve]).to_parquet(
        f"{OUT}/longft_curve.parquet")
    log("long-ft curve saved")

    # exact last-layer gradients for the pool: grad_W = feat (x) (p - y)
    m = make_model(); m.load_state_dict(torch.load(f"{OUT}/base_ckpt.pt")); m.eval()
    feats_l = []
    hook = m.avgpool.register_forward_hook(
        lambda mod, i, o: feats_l.append(o.flatten(1).detach()))
    errs_l, ys = [], []
    dl = DataLoader(Sub(tr, list(pool), PLAIN), batch_size=512, num_workers=0)
    with torch.no_grad():
        for x, y in dl:
            p = F.softmax(m(x), 1)
            e = p.clone(); e[torch.arange(len(y)), y] -= 1.0
            errs_l.append(e); ys.append(y)
    hook.remove()
    np.savez_compressed(f"{OUT}/pool_grads.npz",
                        feats=torch.cat(feats_l).numpy().astype(np.float16),
                        errs=torch.cat(errs_l).numpy().astype(np.float16),
                        labels=torch.cat(ys).numpy(),
                        pool_idx=pool,
                        band=np.array([bands[targets[i]] for i in pool]))
    log("pool gradients saved (exact rank-1 factors)")

    json.dump({"base_overall": float(acc0.mean()), "band_counts": bcnt,
               "band_acc0": bacc0, "counts_pattern": BASE_COUNTS,
               "B": B, "old_mix": OLD_MIX, "n_seeds": N_SEEDS,
               "base_acc": [round(float(a), 4) for a in acc0]},
              open(f"{OUT}/report.json", "w"))
    log("SANDBOX COMPLETE")


if __name__ == "__main__":
    main()
