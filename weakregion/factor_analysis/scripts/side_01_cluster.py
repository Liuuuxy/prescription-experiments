"""Side-of-sink angle, step 1: per-layout 2-means clustering on (obj_x_abs, obj_y_abs),
bimodality validation, and unsigned within run x layout permutation test."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
SP = '/data/xinyua11/tmp/factor_analysis_scratch'
df = pd.read_csv(f'{SP}/pooled_episodes.csv')

# run-demeaned success
df['dm'] = df['success'] - df.groupby('run')['success'].transform('mean')


def twomeans(X, iters=50):
    # deterministic init: split on first principal axis
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    proj = Xc @ Vt[0]
    lab = (proj > np.median(proj)).astype(int)
    for _ in range(iters):
        c0, c1 = X[lab == 0].mean(0), X[lab == 1].mean(0)
        d0 = ((X - c0) ** 2).sum(1)
        d1 = ((X - c1) ** 2).sum(1)
        nl = (d1 < d0).astype(int)
        if (nl == lab).all():
            break
        if nl.min() == nl.max():  # collapsed
            break
        lab = nl
    return lab


rows = []
labels = pd.Series(index=df.index, dtype=float)
for L, g in df.groupby('layout_id'):
    X = g[['obj_x_abs', 'obj_y_abs']].to_numpy()
    if len(g) < 20:
        continue
    lab = twomeans(X)
    if lab.min() == lab.max():
        continue
    c0, c1 = X[lab == 0].mean(0), X[lab == 1].mean(0)
    gap = np.linalg.norm(c1 - c0)
    # within-cluster spread along the separation axis
    axis = (c1 - c0) / gap
    p = X @ axis
    p0, p1 = p[lab == 0], p[lab == 1]
    wsd = np.sqrt((p0.var(ddof=1) * (len(p0) - 1) + p1.var(ddof=1) * (len(p1) - 1)) / (len(p) - 2))
    sep = gap / wsd if wsd > 0 else np.inf
    # silhouette (mean) on the 1-d projection is fine; do 2-d euclid silhouette cheaply
    from scipy.spatial.distance import cdist
    D = cdist(X, X)
    sil = []
    for i in range(len(X)):
        same = lab == lab[i]
        same[i] = False
        a = D[i][same].mean() if same.any() else 0
        b = D[i][~ (lab == lab[i])].mean()
        sil.append((b - a) / max(a, b))
    sil = float(np.mean(sil))
    frac_small = min((lab == 0).mean(), (lab == 1).mean())
    rows.append(dict(layout=L, n=len(g), gap=gap, wsd=wsd, sep=sep, sil=sil,
                     frac_small=frac_small, dx=abs(c1[0]-c0[0]), dy=abs(c1[1]-c0[1])))
    labels.loc[g.index] = lab

df['cluster'] = labels
bi = pd.DataFrame(rows).set_index('layout')
bi['bimodal'] = (bi['sep'] > 2.0) & (bi['sil'] > 0.45) & (bi['frac_small'] > 0.15)
print(bi.round(3).to_string())
print('\nclearly bimodal layouts: %d / %d  (sep>2 & sil>0.45 & minor frac>0.15)'
      % (bi['bimodal'].sum(), len(bi)))
print('sep distribution:', bi['sep'].describe().round(2).to_dict())

bi.to_csv(f'{SP}/side_bimodality.csv')
df[['run', 'episode', 'layout_id', 'cluster', 'dm', 'success',
    'obj_x_rel', 'obj_y_rel', 'obj_x_abs', 'obj_y_abs']].to_csv(f'{SP}/side_clusters.csv', index=False)

# ---------- unsigned within run x layout test ----------
BIML = bi.index[bi['bimodal']].tolist()
sub = df[df['layout_id'].isin(BIML)].copy()

def layout_deltas(d, ycol):
    """per-layout SR delta between clusters, computed from run x layout cells
    containing both clusters (cell-demeaned to kill run and layout main effects)."""
    out = {}
    for L, g in d.groupby('layout_id'):
        num = 0.0; den = 0.0
        for _, cell in g.groupby('run'):
            if cell['cluster'].nunique() < 2:
                continue
            y = cell[ycol] - cell[ycol].mean()
            m1 = y[cell['cluster'] == 1].mean()
            m0 = y[cell['cluster'] == 0].mean()
            n1 = (cell['cluster'] == 1).sum(); n0 = (cell['cluster'] == 0).sum()
            w = 1.0 / (1.0 / n1 + 1.0 / n0)  # harmonic weight
            num += w * (m1 - m0); den += w
        if den > 0:
            out[L] = (num / den, den)
    return out


def stat_absmean(deltas):
    if not deltas:
        return np.nan
    ws = np.array([w for _, w in deltas.values()])
    ds = np.array([d for d, _ in deltas.values()])
    return float(np.sum(ws * np.abs(ds)) / ws.sum())


obs_deltas = layout_deltas(sub, 'success')
obs = stat_absmean(obs_deltas)

# permutation: shuffle cluster labels within run x layout
NPERM = 2000
null = np.empty(NPERM)
grp = sub.groupby(['run', 'layout_id'], sort=False).indices
idx_groups = [np.asarray(v) for v in grp.values()]
clu = sub['cluster'].to_numpy().copy()
sub2 = sub.copy()
for b in range(NPERM):
    c = clu.copy()
    for g in idx_groups:
        c[g] = rng.permutation(c[g])
    sub2['cluster'] = c
    null[b] = stat_absmean(layout_deltas(sub2, 'success'))

p = (np.sum(null >= obs) + 1) / (NPERM + 1)
print('\n=== unsigned side test (bimodal layouts, weighted mean |delta|) ===')
print('n layouts=%d, n eps=%d' % (len(obs_deltas), len(sub)))
print('observed weighted mean|delta| = %.4f' % obs)
print('null mean = %.4f, null 95pct = %.4f' % (null.mean(), np.quantile(null, 0.95)))
print('perm p = %.4f' % p)

# per-layout deltas with per-layout permutation p
print('\nper-layout signed deltas (cluster1 - cluster0):')
for L in sorted(obs_deltas, key=lambda k: -abs(obs_deltas[k][0])):
    d, w = obs_deltas[L]
    print('  layout %2d: delta=%+.3f (eff n=%.0f)' % (L, d, w))

np.save(f'{SP}/side_null_unsigned.npy', null)
pd.DataFrame([(L, d, w) for L, (d, w) in obs_deltas.items()],
             columns=['layout', 'delta', 'wn']).to_csv(f'{SP}/side_layout_deltas.csv', index=False)
