"""Side angle, step 3: vectorized run x layout-stratified permutation tests.
- unsigned side effect (weighted mean |delta_L|)
- quad(x_rel,y_rel)-controlled version
- near/far-of-sink signed version (sign from fail_no_grasp min_sink_dist)
- rim effect before/after side control
- method-of-moments true SD of side gap
"""
import numpy as np
import pandas as pd

SP = '/data/xinyua11/tmp/factor_analysis_scratch'
rng = np.random.default_rng(1)
NPERM = 5000

df = pd.read_csv(f'{SP}/pooled_episodes.csv')
clu = pd.read_csv(f'{SP}/side_clusters.csv')
df['cluster'] = clu['cluster'].to_numpy()
bi = pd.read_csv(f'{SP}/side_bimodality.csv').set_index('layout')
CLEAR = bi.index[bi['sep'] > 8].tolist()          # 41 clean bimodal layouts
print('clear bimodal layouts: %d; ambiguous: %s' % (len(CLEAR), sorted(set(bi.index) - set(CLEAR))))

df['dm'] = df['success'] - df.groupby('run')['success'].transform('mean')

sub = df[df['layout_id'].isin(CLEAR) & df['cluster'].notna()].reset_index(drop=True)
n = len(sub)
lay_codes, lay_uniq = pd.factorize(sub['layout_id'])
cell_codes, _ = pd.factorize(sub['run'].astype(str) + '|' + sub['layout_id'].astype(str))
z = sub['cluster'].to_numpy().astype(int)
NL = len(lay_uniq)

# permutation machinery: shuffle z within cells via lexsort trick
order_by_cell = np.argsort(cell_codes, kind='stable')
z_sorted = z[order_by_cell]  # labels listed cell by cell (original order within cell)

def perm_z():
    u = rng.random(n)
    order = np.lexsort((u, cell_codes))  # cells in same order, random within cell
    zp = np.empty(n, dtype=int)
    zp[order] = z_sorted
    return zp

def layout_deltas(y, zz):
    s1 = np.bincount(lay_codes, weights=y * zz, minlength=NL)
    n1 = np.bincount(lay_codes, weights=zz, minlength=NL)
    s0 = np.bincount(lay_codes, weights=y * (1 - zz), minlength=NL)
    n0 = np.bincount(lay_codes, weights=(1 - zz), minlength=NL)
    with np.errstate(invalid='ignore', divide='ignore'):
        d = s1 / n1 - s0 / n0
    return d, n1, n0

y = sub['dm'].to_numpy()
d_obs, n1, n0 = layout_deltas(y, z)
w = 1.0 / (1.0 / n1 + 1.0 / n0)   # fixed under within-cell permutation? n1 per layout varies!
# NOTE: within-cell permutation preserves per-CELL counts hence per-layout counts too -> w fixed.

def stat_abs(d):
    return float(np.sum(w * np.abs(d)) / w.sum())

def stat_signed(d, sign):
    m = ~np.isnan(sign)
    return float(np.sum((w * sign * d)[m]) / w[m].sum())

obs_abs = stat_abs(d_obs)

# ---- residualized on quadratic + rim features of (x_rel, y_rel) ----
xr = sub['obj_x_rel'].to_numpy(); yr = sub['obj_y_rel'].to_numpy()
rim = np.maximum(np.abs(xr), np.abs(yr))
F = np.column_stack([np.ones(n), xr, yr, xr**2, yr**2, xr*yr, rim, (rim > 0.65).astype(float)])
beta, *_ = np.linalg.lstsq(F, y, rcond=None)
yres = y - F @ beta
d_res, _, _ = layout_deltas(yres, z)
obs_res = stat_abs(d_res)

# ---- near/far sign from fail_no_grasp min_sink_dist (object unmoved at start) ----
ng = sub[sub['failure_phase'] == 'fail_no_grasp']
sink_d = ng.groupby(['layout_id', 'cluster'])['min_sink_dist'].agg(['mean', 'count']).unstack()
sign = np.full(NL, np.nan)
dist_gap = np.full(NL, np.nan)
for i, L in enumerate(lay_uniq):
    try:
        m0 = sink_d.loc[L, ('mean', 0.0)]; m1 = sink_d.loc[L, ('mean', 1.0)]
        c0 = sink_d.loc[L, ('count', 0.0)]; c1 = sink_d.loc[L, ('count', 1.0)]
    except KeyError:
        continue
    if np.isnan(m0) or np.isnan(m1) or c0 < 5 or c1 < 5:
        continue
    # sign=+1 means cluster1 is the NEAR-sink cluster
    sign[i] = 1.0 if m1 < m0 else -1.0
    dist_gap[i] = abs(m1 - m0)
obs_near = stat_signed(d_obs, sign)  # >0 => near-sink side has higher SR

# ---- permutation loop ----
null_abs = np.empty(NPERM); null_res = np.empty(NPERM); null_near = np.empty(NPERM)
null_d = np.empty((NPERM, NL))
for b in range(NPERM):
    zp = perm_z()
    d, _, _ = layout_deltas(y, zp)
    null_d[b] = d
    null_abs[b] = stat_abs(d)
    dr, _, _ = layout_deltas(yres, zp)
    null_res[b] = stat_abs(dr)
    null_near[b] = stat_signed(d, sign)

def pval(obs, nullv, two=False):
    if two:
        return (np.sum(np.abs(nullv) >= abs(obs)) + 1) / (len(nullv) + 1)
    return (np.sum(nullv >= obs) + 1) / (len(nullv) + 1)

print('\n=== (2) unsigned side effect, %d clear-bimodal layouts, n=%d eps ===' % (NL, n))
print('observed weighted mean|delta| = %.4f   null mean = %.4f (95pct %.4f)   p = %.4f'
      % (obs_abs, null_abs.mean(), np.quantile(null_abs, 0.95), pval(obs_abs, null_abs)))

print('\n=== (4) after residualizing quad(x_rel,y_rel)+rim ===')
print('observed = %.4f   null mean = %.4f   p = %.4f   (raw obs %.4f -> ratio %.2f)'
      % (obs_res, null_res.mean(), pval(obs_res, null_res), obs_abs, obs_res / obs_abs))

print('\n=== (3-data) signed near-sink-minus-far stat (sign from no-grasp min_sink_dist) ===')
m = ~np.isnan(sign)
print('layouts signed: %d; median |dist gap| between clusters = %.3f m' % (m.sum(), np.nanmedian(dist_gap)))
print('observed signed stat = %+.4f   two-sided p = %.4f' % (obs_near, pval(obs_near, null_near, two=True)))
nearpos = (sign[m] * d_obs[m] > 0)
print('layouts where NEAR side better: %d / %d' % (nearpos.sum(), m.sum()))

# ---- method-of-moments true SD of the side gap ----
noise_var = null_d.var(axis=0)
mom = d_obs**2 - noise_var
true_var = float(np.sum(w * mom) / w.sum())
print('\n=== effect-size calibration ===')
print('weighted E[delta^2_obs]=%.4f, E[noise]=%.4f -> true var=%.4f, true SD of side gap = %.3f'
      % (float(np.sum(w * d_obs**2) / w.sum()), float(np.sum(w * noise_var) / w.sum()),
         true_var, np.sqrt(max(true_var, 0))))
print('implied per-episode SD (gap/2) = %.3f  [compare style SD ~0.08]' % (np.sqrt(max(true_var, 0)) / 2))

# per-layout table with per-layout p
pl = pd.DataFrame({'layout': lay_uniq, 'delta': d_obs, 'w': w, 'sign_near1': sign,
                   'dist_gap': dist_gap,
                   'p_layout': [(np.sum(np.abs(null_d[:, i]) >= abs(d_obs[i])) + 1) / (NPERM + 1)
                                for i in range(NL)]})
pl = pl.sort_values('p_layout')
print('\nper-layout deltas (cluster1-cluster0), sorted by p:')
print(pl.round(3).to_string(index=False))
pl.to_csv(f'{SP}/side_layout_deltas_v2.csv', index=False)

# ---- rim effect before/after side control ----
r = (rim > 0.65).astype(float)
def fe_slope(yv, xv, groups):
    gy = pd.Series(yv).groupby(groups).transform('mean').to_numpy()
    gx = pd.Series(xv).groupby(groups).transform('mean').to_numpy()
    yc = yv - gy; xc = xv - gx
    return float(np.dot(yc, xc) / np.dot(xc, xc))

g_rl = pd.Series(cell_codes)                      # run x layout
g_rlc = pd.Series(cell_codes * 2 + z)             # run x layout x cluster
b_before = fe_slope(y, r, g_rl)
b_after = fe_slope(y, r, g_rlc)
print('\n=== rim(>0.65) penalty before/after side control ===')
print('within run x layout: %.4f ; within run x layout x side: %.4f' % (b_before, b_after))

# sanity: is cluster correlated with x_rel/y_rel?
cors = []
for L, g in sub.groupby('layout_id'):
    for col in ('obj_x_rel', 'obj_y_rel'):
        v = g[col].to_numpy(); c = g['cluster'].to_numpy()
        if c.std() > 0 and v.std() > 0:
            cors.append(np.corrcoef(v, c)[0, 1])
print('\ncluster vs (x_rel,y_rel) per-layout |corr|: median %.3f, max %.3f'
      % (np.median(np.abs(cors)), np.max(np.abs(cors))))

np.save(f'{SP}/side_null_d.npy', null_d)
