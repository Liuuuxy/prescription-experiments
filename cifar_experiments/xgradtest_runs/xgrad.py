"""Gradient-influence sandbox on a HARDER, slow-converging vision task with GROUND TRUTH.

Why this exists: the ViT-B/16 + CIFAR-10 sandbox converged in ~100 steps (pretrained ViT already
solves CIFAR-10), so the gradient-cosine signal had no window and `cos(g_train, .)` collapsed to
noise immediately. Two fixes here:
  (1) harder task -> ResNet-18 (GroupNorm) trained FROM SCRATCH on CIFAR-100 (thousands of steps).
  (2) log all grads (sparse-JL sketch + norm, per checkpoint) -> compute every scoring variant OFFLINE.

The key upgrade over the robotics setup: a failure region with KNOWN ground truth. We make 20 of the
100 classes RARE in the base set (keep only `rare_frac` of their train examples). The model is then
genuinely bad at those classes (the "failure region"), and the candidate pool's truly-helpful examples
are KNOWN = the held-out rare-class examples. So "does gradient influence find the helpful data?"
becomes a clean AUC(rare-class candidate scores > common-class) with a real answer.

Subcommands:
  train : ResNet18-GN on the base set; checkpoints; track cos(EMA train grad, g_val/g_fail) over steps.
  log   : at given checkpoints, save per-candidate gradient SKETCH+NORM (raw, per-ckpt) + g_val/g_fail.
  menu  : offline -- score candidates by many variants; report AUC vs ground truth, cos(g_val,g_fail),
          signal decay, cosine-vs-projection (magnitude), and a 1-step short-fit validation.

  python xgrad.py train  --steps 6000
  python xgrad.py log     --ckpts 500,1500,3000,6000
  python xgrad.py menu
Run in the robocasa conda env (torch 2.7 + torchvision + torch.func).
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap
import torchvision as tv
import torchvision.transforms as T

ROOT = "/data/xinyua11/xgradtest"
DATA = f"{ROOT}/data"
CKPTS = f"{ROOT}/ckpts"
LOGS = f"{ROOT}/gradlog"
DEV = "cuda"
N_RARE = 20            # first 20 of the 100 classes are the failure region
RARE_FRAC = 0.1        # base set keeps only 10% of rare-class train examples


# ---------------------------------------------------------------- data
def cifar100(train):
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
    return tv.datasets.CIFAR100(DATA, train=train, download=True, transform=tf)


def splits(seed=0):
    """base (rare classes downsampled) | candidate pool (the rest of train) | clean val test | failure test."""
    rng = np.random.RandomState(seed)
    tr = cifar100(True)
    y = np.array(tr.targets)
    rare = set(range(N_RARE))
    base_idx, pool_idx = [], []
    for c in range(100):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        if c in rare:
            k = int(len(idx) * RARE_FRAC)          # base: only 10% of rare-class examples
            base_idx += idx[:k].tolist()
            pool_idx += idx[k:].tolist()           # the held-out rare examples are the "helpful" pool
        else:
            half = len(idx) // 2
            base_idx += idx[:half].tolist()
            pool_idx += idx[half:].tolist()
    te = cifar100(False)
    yte = np.array(te.targets)
    fail_idx = np.where(np.isin(yte, list(rare)))[0]    # failure set = rare-class test examples
    return tr, te, sorted(base_idx), sorted(pool_idx), list(range(len(te))), fail_idx.tolist()


def loader(ds, idx, bs, shuffle):
    return torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=bs,
                                       shuffle=shuffle, num_workers=4, drop_last=shuffle)


# ---------------------------------------------------------------- model (GroupNorm -> clean per-sample grads)
def make_model():
    m = tv.models.resnet18(num_classes=100,
                           norm_layer=lambda c: nn.GroupNorm(min(32, c), c))
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)   # CIFAR stem (32x32)
    m.maxpool = nn.Identity()
    return m.to(DEV)


def accuracy(model, ds, idx, classes=None):
    model.eval()
    c = t = 0
    with torch.no_grad():
        for x, y in loader(ds, idx, 512, False):
            x = x.to(DEV); p = model(x).argmax(1).cpu()
            m = torch.ones_like(y, dtype=torch.bool) if classes is None else torch.isin(y, torch.tensor(list(classes)))
            c += (p[m] == y[m]).sum().item(); t += m.sum().item()
    return c / max(t, 1)


# ---------------------------------------------------------------- sparse JL sketch
def make_proj(P, d=2048, nnz=512, seed=7):
    rng = np.random.RandomState(seed)
    idx = torch.as_tensor(rng.randint(0, P, size=(d, nnz)), device=DEV)
    sgn = torch.as_tensor((rng.randint(0, 2, size=(d, nnz)) * 2 - 1).astype(np.float32), device=DEV)
    return idx, sgn


def sketch_flat(flat, idx, sgn):              # flat [B,P] -> [B,d]
    return (flat[:, idx] * sgn).sum(-1)


# ---------------------------------------------------------------- per-sample gradients (torch.func)
def grad_params(model):
    # differentiate the last block + head only (cheap, clean, where the class signal lives)
    names = [n for n, _ in model.named_parameters() if n.startswith(("layer4", "fc"))]
    return names


def persample_flatgrad(model, names, x, y):
    """Per-sample gradient (flattened) w.r.t. `names`, for a batch -> [B, P]."""
    params = {n: p.detach() for n, p in model.named_parameters()}
    buffers = {n: b.detach() for n, b in model.named_buffers()}
    diff = {n: params[n] for n in names}
    rest = {n: params[n] for n in params if n not in names}

    def loss1(diffp, xi, yi):
        out = functional_call(model, ({**rest, **diffp}, buffers), (xi.unsqueeze(0),))
        return F.cross_entropy(out, yi.unsqueeze(0))

    g = vmap(grad(loss1), in_dims=(None, 0, 0))(diff, x, y)   # dict of [B,*shape]
    return torch.cat([g[n].reshape(x.shape[0], -1) for n in names], 1)


def mean_grad_sketch(model, names, ds, idx, idxp, sgn, bs=256, max_n=2048):
    """Mean (raw) gradient sketch over a probe set (e.g., val or failure set)."""
    acc = torch.zeros(idxp.shape[0], device=DEV); n = 0
    for x, y in loader(ds, idx[:max_n], bs, False):
        x, y = x.to(DEV), y.to(DEV)
        fg = persample_flatgrad(model, names, x, y)      # [B,P]
        acc += sketch_flat(fg, idxp, sgn).sum(0); n += x.shape[0]
    return (acc / n).cpu().numpy()


# ---------------------------------------------------------------- train
def run_train(a):
    os.makedirs(CKPTS, exist_ok=True)
    tr, te, base, pool, val, fail = splits()
    json.dump({"base": base, "pool": pool, "val": val, "fail": fail},
              open(f"{ROOT}/splits.json", "w"))
    print(f"[train] base {len(base)} | pool {len(pool)} | val {len(val)} | fail {len(fail)} "
          f"({N_RARE} rare classes @ {RARE_FRAC})", flush=True)
    model = make_model()
    names = grad_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=5e-4)
    dl = loader(tr, base, a.bs, True)
    idxp, sgn = None, None  # built lazily once we know P

    # EMA of the training gradient direction + fixed probe grads, to replicate the cos(g_train,.) plot
    ema = None; cos_log = []
    step = 0; it = iter(dl)
    while step < a.steps:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(dl); x, y = next(it)
        x, y = x.to(DEV), y.to(DEV)
        model.train()
        out = model(x); loss = F.cross_entropy(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
        step += 1

        if step % a.cos_every == 0:
            # flat grad of this batch over `names`
            flat = torch.cat([p.grad.reshape(-1) for n, p in model.named_parameters()
                              if n in names and p.grad is not None]).detach()
            ema = flat if ema is None else 0.9 * ema + 0.1 * flat
            if idxp is None:
                idxp, sgn = make_proj(ema.numel(), a.sketch_dim, a.nnz)
            gv = torch.as_tensor(mean_grad_sketch(model, names, te, np.array(val), idxp, sgn), device=DEV)
            gf = torch.as_tensor(mean_grad_sketch(model, names, te, np.array(fail), idxp, sgn), device=DEV)
            es = sketch_flat(ema.unsqueeze(0), idxp, sgn)[0]
            cv = F.cosine_similarity(es, gv, 0).item(); cf = F.cosine_similarity(es, gf, 0).item()
            cos_log.append({"step": step, "loss": loss.item(),
                            "cos_train_val": cv, "cos_train_fail": cf,
                            "cos_val_fail": F.cosine_similarity(gv, gf, 0).item()})
            print(f"  step {step:5d} loss {loss.item():.3f}  cos(train,val) {cv:+.3f}  "
                  f"cos(train,fail) {cf:+.3f}  cos(val,fail) {cos_log[-1]['cos_val_fail']:+.3f}", flush=True)
        if step in a.save_at or step == a.steps:
            torch.save(model.state_dict(), f"{CKPTS}/step{step}.pt")
            va = accuracy(model, te, val); fa = accuracy(model, te, fail, classes=set(range(N_RARE)))
            print(f"  [ckpt {step}] val_acc {va:.3f}  FAIL(rare)_acc {fa:.3f}  saved", flush=True)
    json.dump(cos_log, open(f"{ROOT}/cos_log.json", "w"), indent=2)
    print(f"[train] done -> ckpts {a.save_at}; cos_log.json (the signal-decay curve)", flush=True)


# ---------------------------------------------------------------- log grads
def run_log(a):
    os.makedirs(LOGS, exist_ok=True)
    sp = json.load(open(f"{ROOT}/splits.json"))
    tr, te = cifar100(True), cifar100(False)
    pool = np.array(sp["pool"])[:a.max_pool]
    ypool = np.array(tr.targets)[pool]
    ckpts = [int(c) for c in a.ckpts.split(",")]
    model = make_model(); names = grad_params(model)
    P = sum(p.numel() for n, p in model.named_parameters() if n in names)
    idxp, sgn = make_proj(P, a.sketch_dim, a.nnz)
    print(f"[log] pool {len(pool)} | grad-params P={P} -> sketch {a.sketch_dim} | ckpts {ckpts}", flush=True)

    T_ck = len(ckpts)
    cand_raw = np.zeros((T_ck, len(pool), a.sketch_dim), np.float32)
    cand_nrm = np.zeros((T_ck, len(pool)), np.float32)
    gval = np.zeros((T_ck, a.sketch_dim), np.float32)
    gfail = np.zeros((T_ck, a.sketch_dim), np.float32)
    for ti, ck in enumerate(ckpts):
        model.load_state_dict(torch.load(f"{CKPTS}/step{ck}.pt", map_location=DEV)); model.eval()
        for s in range(0, len(pool), a.bs):
            ids = pool[s:s + a.bs]
            xb = torch.stack([tr[i][0] for i in ids]).to(DEV)
            yb = torch.tensor([tr[i][1] for i in ids]).to(DEV)
            fg = persample_flatgrad(model, names, xb, yb)          # [b,P]
            cand_raw[ti, s:s + len(ids)] = sketch_flat(fg, idxp, sgn).cpu().numpy()
            cand_nrm[ti, s:s + len(ids)] = fg.norm(dim=1).cpu().numpy()
            if (s // a.bs) % 20 == 0:
                print(f"    ckpt {ck}: {s+len(ids)}/{len(pool)}", flush=True)
        gval[ti] = mean_grad_sketch(model, names, te, np.array(sp["val"]), idxp, sgn)
        gfail[ti] = mean_grad_sketch(model, names, te, np.array(sp["fail"]), idxp, sgn)
        print(f"  [log] ckpt {ck} done", flush=True)
    np.save(f"{LOGS}/cand_raw.npy", cand_raw); np.save(f"{LOGS}/cand_norms.npy", cand_nrm)
    np.save(f"{LOGS}/gval.npy", gval); np.save(f"{LOGS}/gfail.npy", gfail)
    np.save(f"{LOGS}/pool_labels.npy", ypool)
    json.dump({"ckpts": ckpts, "n_rare": N_RARE, "pool": pool.tolist()}, open(f"{LOGS}/meta.json", "w"))
    print(f"[log] saved raw per-ckpt sketches + norms -> {LOGS} (offline menu ready)", flush=True)


# ---------------------------------------------------------------- offline menu
def _unit(x, ax=-1):
    return x / (np.linalg.norm(x, axis=ax, keepdims=True) + 1e-12)


def _auc(score, is_hot):
    h, e = score[is_hot], score[~is_hot]
    o = np.argsort(score); r = np.empty(len(score)); r[o] = np.arange(len(score))
    return (r[is_hot].sum() - len(h) * (len(h) - 1) / 2) / (len(h) * len(e))


def run_menu(a):
    m = json.load(open(f"{LOGS}/meta.json"))
    raw = np.load(f"{LOGS}/cand_raw.npy"); nrm = np.load(f"{LOGS}/cand_norms.npy")
    gval = np.load(f"{LOGS}/gval.npy"); gfail = np.load(f"{LOGS}/gfail.npy")
    y = np.load(f"{LOGS}/pool_labels.npy")
    is_helpful = y < m["n_rare"]            # GROUND TRUTH: rare-class pool examples are the helpful ones
    ck = m["ckpts"]
    print(f"[menu] pool {raw.shape[1]} (helpful/rare {is_helpful.sum()}) | ckpts {ck}\n")

    print("=== cos(g_val, g_fail) per checkpoint (collinearity trap; ~1 => can't separate) ===")
    for ti, c in enumerate(ck):
        print(f"  step {c}: cos(g_val,g_fail) = {float(_unit(gval[ti])@_unit(gfail[ti])):+.3f}")

    print("\n=== AUC(helpful > common) per scoring variant, at EACH checkpoint ===")
    print(f"  {'variant':28s} " + " ".join(f"@{c:>5}" for c in ck))
    def row(name, fn):
        print(f"  {name:28s} " + " ".join(f"{fn(ti):>6.3f}" for ti in range(len(ck))))
    Zu = lambda ti: _unit(raw[ti])                      # candidate directions at ckpt ti
    row("cosine to g_fail", lambda ti: _auc(Zu(ti) @ _unit(gfail[ti]), is_helpful))
    row("cosine to g_val", lambda ti: _auc(Zu(ti) @ _unit(gval[ti]), is_helpful))
    row("contrast cos(fail-val)", lambda ti: _auc(Zu(ti) @ _unit(gfail[ti] - gval[ti]), is_helpful))
    row("MAG raw-dot g_fail", lambda ti: _auc(raw[ti] @ _unit(gfail[ti]), is_helpful))
    row("MAG projection (||.||cos)", lambda ti: _auc((raw[ti] @ _unit(gfail[ti])), is_helpful))  # = raw-dot
    row("MAG contrast raw-dot", lambda ti: _auc(raw[ti] @ _unit(gfail[ti] - gval[ti]), is_helpful))
    row("grad-norm only", lambda ti: _auc(nrm[ti], is_helpful))

    print("\nNote: 'projection' = ||grad||*cos = raw-dot with the unit target (magnitude-AWARE);")
    print("'cosine' divides that by ||grad|| (magnitude-BLIND). Compare the two rows above.")
    print("AUC 0.5 = no signal; >0.6 real; the gap cosine->projection is the magnitude contribution.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train"); tr.add_argument("--steps", type=int, default=6000)
    tr.add_argument("--bs", type=int, default=128); tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--cos_every", type=int, default=50)
    tr.add_argument("--save_at", type=lambda s: [int(x) for x in s.split(",")], default=[500, 1500, 3000, 6000])
    tr.add_argument("--sketch_dim", type=int, default=2048); tr.add_argument("--nnz", type=int, default=512)
    lg = sub.add_parser("log"); lg.add_argument("--ckpts", default="500,1500,3000,6000")
    lg.add_argument("--bs", type=int, default=128); lg.add_argument("--max_pool", type=int, default=8000)
    lg.add_argument("--sketch_dim", type=int, default=2048); lg.add_argument("--nnz", type=int, default=512)
    sub.add_parser("menu")
    a = ap.parse_args()
    {"train": run_train, "log": run_log, "menu": run_menu}[a.cmd](a)


if __name__ == "__main__":
    main()
