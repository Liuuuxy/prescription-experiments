"""Angle: category-signal variance decomposition.
Ridge regression (manual, numpy) on run-demeaned success.
Families: shape / location / layout / style / category.
Cross-validated across runs (7 folds of runs, size-balanced): R^2 alone and added-last,
plus within-run AUC on held-out runs.
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SP = '/data/xinyua11/tmp/factor_analysis_scratch'
df = pd.read_csv(f'{SP}/pooled_episodes.csv')
df['success'] = df['success'].astype(int)
df['y'] = df['success'] - df.groupby('run')['success'].transform('mean')

# impute the 50 missing shape rows (one run) with overall median
for c in ('obj_height', 'obj_width'):
    df[c] = df[c].fillna(df[c].median())
df['hw_ratio'] = df['obj_height'] / np.maximum(df['obj_width'], 1e-6)

y = df['y'].to_numpy()
suc = df['success'].to_numpy()
runs = df['run'].to_numpy()
run_names = np.sort(df['run'].unique())

def onehot(col):
    u, inv = np.unique(df[col].to_numpy(), return_inverse=True)
    X = np.zeros((len(df), len(u)))
    X[np.arange(len(df)), inv] = 1.0
    return X

xr, yr = df['obj_x_rel'].to_numpy(), df['obj_y_rel'].to_numpy()
FAM = {
    'shape': np.column_stack([df['obj_height'], df['obj_width'], df['hw_ratio']]),
    'location': np.column_stack([xr, yr, xr**2, yr**2, xr*yr]),
    'layout': onehot('layout_id'),
    'style': onehot('style_id'),
    'category': onehot('object_category'),
}
fam_names = list(FAM)

# --- folds: 7 outer folds of runs, snake-assigned by run size for balance ---
sizes = df['run'].value_counts()
order = sizes.index.to_numpy()  # descending size
K = 7
fold_of_run = {}
for i, r in enumerate(order):
    k = i % (2 * K)
    fold_of_run[r] = k if k < K else 2 * K - 1 - k
fold = np.array([fold_of_run[r] for r in runs])

LAMBDAS = [10.0**e for e in range(-2, 5)]  # 0.01 .. 10000

def ridge_fit(X, yv, lam):
    # standardize continuous-scale columns jointly: center all, scale by col std
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1
    Xs = (X - mu) / sd
    d = Xs.shape[1]
    A = Xs.T @ Xs + lam * np.eye(d)
    b = Xs.T @ yv
    w = np.linalg.solve(A, b)
    return (mu, sd, w, yv.mean())

def ridge_pred(model, X):
    mu, sd, w, b0 = model
    return (X - mu) / sd @ w + b0

def auc_within_runs(score, s, run_arr):
    num = 0.0; den = 0.0
    for r in np.unique(run_arr):
        m = run_arr == r
        sc, ss = score[m], s[m]
        pos, neg = sc[ss == 1], sc[ss == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        # rank-based AUC
        allv = np.concatenate([pos, neg])
        ranks = pd.Series(allv).rank().to_numpy()
        auc = (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
        w = len(pos) * len(neg)
        num += auc * w; den += w
    return num / den

def cv_eval(X):
    """outer-CV R^2 (vs 0-baseline) and within-run AUC; inner-CV lambda per outer fold."""
    r2s, preds = [], np.zeros(len(y))
    for k in range(K):
        tr = fold != k; te = fold == k
        # inner CV over train runs: reuse fold ids among train folds
        best_lam, best = None, -np.inf
        for lam in LAMBDAS:
            sse = 0.0; sst = 0.0
            for j in range(K):
                if j == k:
                    continue
                itr = tr & (fold != j); ite = fold == j
                m = ridge_fit(X[itr], y[itr], lam)
                e = y[ite] - ridge_pred(m, X[ite])
                sse += (e**2).sum(); sst += (y[ite]**2).sum()
            r2 = 1 - sse / sst
            if r2 > best:
                best, best_lam = r2, lam
        m = ridge_fit(X[tr], y[tr], best_lam)
        p = ridge_pred(m, X[te])
        preds[te] = p
        r2s.append((1 - ((y[te] - p)**2).sum() / (y[te]**2).sum(), best_lam))
    # pooled out-of-fold R^2 and AUC
    r2_pooled = 1 - ((y - preds)**2).sum() / (y**2).sum()
    auc = auc_within_runs(preds, suc, runs)
    return r2_pooled, np.array([x[0] for x in r2s]), [x[1] for x in r2s], auc

print("=== FAMILY ALONE (CV across runs, out-of-fold) ===")
alone = {}
for name in fam_names:
    r2, r2f, lams, auc = cv_eval(FAM[name])
    alone[name] = (r2, r2f, auc)
    print(f"{name:9s}: R2 = {r2:.4f} (fold SD {r2f.std():.4f}), within-run AUC = {auc:.3f}, lambdas {sorted(set(lams))}")

X_full = np.column_stack([FAM[n] for n in fam_names])
r2_full, r2f_full, lams_full, auc_full = cv_eval(X_full)
print(f"\nFULL model: R2 = {r2_full:.4f} (fold SD {r2f_full.std():.4f}), AUC = {auc_full:.3f}")

print("\n=== ADDED-LAST (full minus leave-one-family-out) ===")
added = {}
for name in fam_names:
    Xm = np.column_stack([FAM[n] for n in fam_names if n != name])
    r2m, r2fm, _, aucm = cv_eval(Xm)
    added[name] = (r2_full - r2m, auc_full - aucm, r2m, aucm)
    print(f"{name:9s}: dR2 = {r2_full - r2m:+.4f}, dAUC = {auc_full - aucm:+.3f}  (without-it R2={r2m:.4f}, AUC={aucm:.3f})")

# shape+location only vs category alone: does interpretable geometry match category?
X_sl = np.column_stack([FAM['shape'], FAM['location']])
r2_sl, _, _, auc_sl = cv_eval(X_sl)
print(f"\nshape+location: R2 = {r2_sl:.4f}, AUC = {auc_sl:.3f}")

# ---------- PLOT ----------
fig, ax = plt.subplots(figsize=(8, 4.5))
xpos = np.arange(len(fam_names))
a_vals = [alone[n][0] for n in fam_names]
l_vals = [added[n][0] for n in fam_names]
ax.bar(xpos - 0.2, a_vals, 0.38, label='alone (CV R2)', color='steelblue')
ax.bar(xpos + 0.2, l_vals, 0.38, label='added-last (dR2 over other families)', color='darkorange')
ax.axhline(r2_full, color='k', ls='--', lw=1, label=f'full model R2 = {r2_full:.3f}')
ax.axhline(0, color='0.5', lw=0.8)
ax.set_xticks(xpos); ax.set_xticklabels(fam_names)
ax.set_ylabel('out-of-fold R2 on run-demeaned success')
ax.set_title('Variance decomposition of success (CV across runs)')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f'{SP}/catsignal_family_r2_bars.png', dpi=140)
print('\nsaved catsignal_family_r2_bars.png')

# top/bottom categories by ridge coefficient (full-data category-only fit, lam=10)
m = ridge_fit(FAM['category'], y, 10.0)
coefs = m[2] / m[1]  # de-standardize
u = np.unique(df['object_category'].to_numpy())
srt = np.argsort(coefs)
print("\nbottom-8 categories (ridge effect on demeaned SR):")
for i in srt[:8]:
    print(f"  {u[i]:22s} {coefs[i]:+.3f}")
print("top-8 categories:")
for i in srt[-8:]:
    print(f"  {u[i]:22s} {coefs[i]:+.3f}")
