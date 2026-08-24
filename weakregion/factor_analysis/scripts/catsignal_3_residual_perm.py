"""Angle: category-signal residual permutation test.
Does category retain signal after controlling shape / location / layout / style?
Ladder of controls; ridge residuals; permute category labels WITHIN RUN.
"""
import numpy as np, pandas as pd

RNG = np.random.default_rng(7)
SP = '/data/xinyua11/tmp/factor_analysis_scratch'
df = pd.read_csv(f'{SP}/pooled_episodes.csv')
df['success'] = df['success'].astype(int)
df['y'] = df['success'] - df.groupby('run')['success'].transform('mean')
for c in ('obj_height', 'obj_width'):
    df[c] = df[c].fillna(df[c].median())
df['hw_ratio'] = df['obj_height'] / np.maximum(df['obj_width'], 1e-6)

y = df['y'].to_numpy()
runs = df['run'].to_numpy()
run_names = np.sort(df['run'].unique())
run_groups = [np.flatnonzero(runs == r) for r in run_names]

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
}
cats = np.sort(df['object_category'].unique())
ci = df['object_category'].map({c: i for i, c in enumerate(cats)}).to_numpy()
n_cat = len(cats)
MIN_N = 20
counts = np.bincount(ci, minlength=n_cat)
mask = counts >= MIN_N

def ridge_residuals(fams, lam):
    if not fams:
        return y.copy()
    X = np.column_stack([FAM[f] for f in fams])
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1
    Xs = (X - mu) / sd
    w = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ y)
    return y - (Xs @ w + y.mean())

def spread_stats(ci_arr, r):
    cnt = np.bincount(ci_arr, minlength=n_cat)
    m = np.bincount(ci_arr, weights=r, minlength=n_cat) / np.maximum(cnt, 1)
    sd = np.std(m[mask], ddof=1)
    T = np.sum(cnt[mask] * m[mask] ** 2)
    return sd, T

def perm_test(r, B=2000):
    obs_sd, obs_T = spread_stats(ci, r)
    null_sd = np.empty(B); null_T = np.empty(B)
    ci_p = ci.copy()
    for b in range(B):
        for g in run_groups:
            ci_p[g] = ci[g][RNG.permutation(len(g))]
        null_sd[b], null_T[b] = spread_stats(ci_p, r)
    p_sd = (1 + (null_sd >= obs_sd).sum()) / (1 + B)
    p_T = (1 + (null_T >= obs_T).sum()) / (1 + B)
    sig = np.sqrt(max(obs_sd**2 - null_sd.mean()**2, 0))
    return obs_sd, null_sd.mean(), p_sd, p_T, sig

# lambdas roughly matching CV-selected values from decomposition script
LADDER = [
    ('none (raw demeaned)', [], 0),
    ('shape', ['shape'], 10.0),
    ('shape+location', ['shape', 'location'], 10.0),
    ('shape+loc+layout+style', ['shape', 'location', 'layout', 'style'], 100.0),
]
print("Category spread on residuals (cats n>=%d: %d), permutation within run, B=2000" % (MIN_N, mask.sum()))
print(f"{'controls':26s} {'obs SD':>8s} {'null SD':>8s} {'p(SD)':>8s} {'p(T)':>8s} {'true sig SD':>11s}")
base_sig = None
for name, fams, lam in LADDER:
    r = ridge_residuals(fams, lam)
    obs, nul, p_sd, p_T, sig = perm_test(r)
    if base_sig is None:
        base_sig = sig
    print(f"{name:26s} {obs:8.4f} {nul:8.4f} {p_sd:8.4f} {p_T:8.4f} {sig:11.4f}  (absorbed {100*(1-sig**2/max(base_sig**2,1e-9)):.0f}% of category variance)")
