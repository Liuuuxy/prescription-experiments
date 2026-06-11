"""Failure predictor: does object geometry + position predict pi0's success?

Pools across ALL episodes (vs noisy per-bucket rates) by fitting a logistic
regression `P(success | obj_height, obj_width, lateral_pos, depth_pos)` with
numpy only (no sklearn). Reports:
  - cross-validated AUC: if ~0.5, these features DON'T explain failures
    (targeting on them won't help); if high, there's a real targeting signal.
  - standardized coefficients: which feature drives failure, and its sign.
  - univariate binned success rates per feature (with counts).
  - the lowest-predicted-success corner of feature space (candidate target).

Usage: python predict_failure.py --weakregion_json <weakregion.json>
"""
import argparse
import json
import numpy as np

FEATS = ["obj_height", "obj_width", "lateral", "depth"]


def load(path):
    eps = json.load(open(path))["episodes"]
    X, y, raw = [], [], []
    for e in eps:
        h, w = e.get("obj_height"), e.get("obj_width")
        rel = e.get("obj_xy_rel", [None, None])
        if None in (h, w, rel[0], rel[1]):
            continue
        X.append([h, w, rel[0], rel[1]])
        y.append(1.0 if e["success"] else 0.0)
        raw.append(e)
    return np.array(X), np.array(y), raw


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_logreg(X, y, iters=4000, lr=0.1, reg=1e-2):
    Xb = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        g = Xb.T @ (sigmoid(Xb @ w) - y) / len(y)
        g[1:] += reg * w[1:]
        w -= lr * g
    return w


def auc(scores, y):
    pos, neg = scores[y == 1], scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def cv_auc(X, y, k=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    folds = np.array_split(idx, k)
    scores = np.full(len(X), np.nan)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    for f in folds:
        tr = np.setdiff1d(idx, f)
        Xtr = (X[tr] - mu) / sd
        w = fit_logreg(Xtr, y[tr])
        Xte = (X[f] - mu) / sd
        scores[f] = sigmoid(np.hstack([np.ones((len(f), 1)), Xte]) @ w)
    return auc(scores, y), scores


def binned(X, y, j, name, nbins=3):
    v = X[:, j]
    qs = np.quantile(v, np.linspace(0, 1, nbins + 1))
    out = []
    for b in range(nbins):
        lo, hi = qs[b], qs[b + 1]
        sel = (v >= lo) & (v <= hi) if b == nbins - 1 else (v >= lo) & (v < hi)
        n = int(sel.sum()); k = int(y[sel].sum())
        out.append(f"    {name}[{lo:.3f},{hi:.3f}]: {k}/{n} = {k/n:.0%}" if n else f"    {name}[{lo:.3f},{hi:.3f}]: (empty)")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weakregion_json", required=True)
    args = p.parse_args()
    X, y, raw = load(args.weakregion_json)
    print(f"n={len(y)} episodes, success rate={y.mean():.1%}")
    if len(y) < 20 or y.sum() in (0, len(y)):
        print("Not enough data / no variance — skipping."); return

    cvauc, _ = cv_auc(X, y)
    print(f"\n5-fold CV AUC (geometry+position -> success): {cvauc:.3f}")
    print("  interpretation: ~0.5 = features DON'T predict failure (don't target on them);")
    print("                  >0.65 = real structure worth targeting.")

    mu, sd = X.mean(0), X.std(0) + 1e-9
    w = fit_logreg((X - mu) / sd, y)
    print("\nStandardized logistic coefficients (sign: + => higher feature -> more SUCCESS):")
    for name, c in sorted(zip(FEATS, w[1:]), key=lambda kv: -abs(kv[1])):
        print(f"  {name:<10} {c:+.3f}")

    print("\nUnivariate binned success (does each feature separate success?):")
    for j, name in enumerate(FEATS):
        for line in binned(X, y, j, name):
            print(line)

    # lowest-predicted-success corner (candidate target region)
    sc = sigmoid(np.hstack([np.ones((len(X), 1)), (X - mu) / sd]) @ w)
    order = np.argsort(sc)
    print("\nLowest predicted-success episodes (candidate targeting region):")
    for i in order[:5]:
        e = raw[i]
        print(f"  pred={sc[i]:.2f} success={e['success']} {e['object_category']} "
              f"h={e['obj_height']} w={e['obj_width']} rel={e['obj_xy_rel']}")


if __name__ == "__main__":
    main()
