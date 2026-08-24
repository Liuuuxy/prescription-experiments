"""Side angle, step 6 (final):
- full 41-layout signed LEFT/RIGHT-of-sink test, episode level AND config level
- power-boosted config-level unsigned test (residualize category + quad position first)
- validation: per-layout corr(|s|, no-grasp start sink dist), NaN-safe
- failure-phase composition vs |s| (approach vs transport mechanism)
"""
import json
import numpy as np
import pandas as pd

SP = '/data/xinyua11/tmp/factor_analysis_scratch'
rng = np.random.default_rng(4)
NPERM = 5000

df = pd.read_csv(f'{SP}/pooled_episodes.csv')
clu = pd.read_csv(f'{SP}/side_clusters.csv')
df['cluster'] = clu['cluster'].to_numpy()
bi = pd.read_csv(f'{SP}/side_bimodality.csv').set_index('layout')
CLEAR = bi.index[bi['sep'] > 8].tolist()
df['dm'] = df['success'] - df.groupby('run')['success'].transform('mean')
sub = df[df['layout_id'].isin(CLEAR) & df['cluster'].notna()].reset_index(drop=True)

probe = json.load(open(f'{SP}/side_sinkpos.json'))
probe.update(json.load(open(f'{SP}/side_sinkpos_rest.json')))

# geometry: sep coord, cluster centers -> sink-frame lateral sign
lay_uniq_all = sorted(sub['layout_id'].unique())
sgn_map = {}
lat = []
for L in lay_uniq_all:
    g = sub[sub['layout_id'] == L]
    Xw = g[['obj_x_abs', 'obj_y_abs']].to_numpy()
    c1 = Xw[g['cluster'] == 1].mean(0); c0 = Xw[g['cluster'] == 0].mean(0)
    e = probe[str(L)]
    sp = np.array(e['sink_pos'][:2]); yaw = e['sink_rot'][0]
    R = np.array([[np.cos(-yaw), -np.sin(-yaw)], [np.sin(-yaw), np.cos(-yaw)]])
    l0 = R @ (c0 - sp); l1 = R @ (c1 - sp)
    lat.append((L, l0[0], l1[0], (R @ ((c0 + c1) / 2 - sp))[1]))
    if np.sign(l0[0]) != np.sign(l1[0]):
        sgn_map[L] = 1.0 if l1[0] > 0 else -1.0   # +1: cluster1 is sink-frame RIGHT
lat = pd.DataFrame(lat, columns=['layout', 'c0_lx', 'c1_lx', 'mid_ly'])
print('clean opposite-side layouts: %d/%d; lateral |cluster offset| median %.2f m; mid depth offset median %.2f m'
      % (len(sgn_map), len(lay_uniq_all), np.median(np.abs(np.r_[lat.c0_lx, lat.c1_lx])), lat.mid_ly.median()))

# ---------- config aggregation ----------
sub['cfg'] = (sub['layout_id'].astype(str) + '|' + sub['style_id'].astype(str) + '|'
              + sub['object_category'] + '|' + sub['obj_x_abs'].round(3).astype(str)
              + '|' + sub['obj_y_abs'].round(3).astype(str))
# residualized outcome: remove category + quad(x_rel,y_rel)+rim global effects from dm
n = len(sub)
xr = sub['obj_x_rel'].to_numpy(); yr = sub['obj_y_rel'].to_numpy()
rim = np.maximum(np.abs(xr), np.abs(yr))
cat_codes, cats = pd.factorize(sub['object_category'])
Fq = np.column_stack([np.ones(n), xr, yr, xr**2, yr**2, xr*yr, rim, (rim > 0.65).astype(float)])
y_ep = sub['dm'].to_numpy()
# category effects via one-hot ridge-free demeaning: subtract category mean, then quad fit
cat_mean = pd.Series(y_ep).groupby(pd.Series(cat_codes)).transform('mean').to_numpy()
y_r = y_ep - cat_mean
beta, *_ = np.linalg.lstsq(Fq, y_r, rcond=None)
y_r = y_r - Fq @ beta
sub['dmr'] = y_r

cfg = sub.groupby('cfg').agg(y=('dm', 'mean'), yres=('dmr', 'mean'), nrep=('dm', 'size'),
                             layout=('layout_id', 'first'), side=('cluster', 'first')).reset_index()
lay_codes, lay_uniq = pd.factorize(cfg['layout'])
NL = len(lay_uniq)
z = cfg['side'].to_numpy().astype(int)
wrep = cfg['nrep'].to_numpy().astype(float)
sgn = np.array([sgn_map.get(L, np.nan) for L in lay_uniq])

def layout_deltas_cfg(y_, z_):
    s1 = np.bincount(lay_codes, weights=wrep * y_ * z_, minlength=NL)
    n1 = np.bincount(lay_codes, weights=wrep * z_, minlength=NL)
    s0 = np.bincount(lay_codes, weights=wrep * y_ * (1 - z_), minlength=NL)
    n0 = np.bincount(lay_codes, weights=wrep * (1 - z_), minlength=NL)
    with np.errstate(invalid='ignore', divide='ignore'):
        return s1 / n1 - s0 / n0, n1, n0

for ycol, tag in [('y', 'raw dm'), ('yres', 'category+quad residualized')]:
    yv = cfg[ycol].to_numpy()
    d_obs, n1, n0 = layout_deltas_cfg(yv, z)
    w = 1.0 / (1.0 / n1 + 1.0 / n0)
    stat_abs = float(np.sum(w * np.abs(d_obs)) / w.sum())
    m = ~np.isnan(sgn)
    stat_sgn = float(np.sum((w * sgn * d_obs)[m]) / w[m].sum())
    order_by_lay = np.argsort(lay_codes, kind='stable')
    z_sorted = z[order_by_lay]
    na = np.empty(NPERM); ns = np.empty(NPERM); nd = np.empty((NPERM, NL))
    for b in range(NPERM):
        u = rng.random(len(cfg))
        order = np.lexsort((u, lay_codes))
        zp = np.empty(len(cfg), dtype=int); zp[order] = z_sorted
        d, _, _ = layout_deltas_cfg(yv, zp)
        nd[b] = d
        na[b] = float(np.sum(w * np.abs(d)) / w.sum())
        ns[b] = float(np.sum((w * sgn * d)[m]) / w[m].sum())
    pa = (np.sum(na >= stat_abs) + 1) / (NPERM + 1)
    ps = (np.sum(np.abs(ns) >= abs(stat_sgn)) + 1) / (NPERM + 1)
    tv = float(np.sum(w * (d_obs**2 - nd.var(axis=0))) / w.sum())
    kpos = int(np.sum((sgn * d_obs)[m] > 0))
    print('\n=== CONFIG-level [%s] ===' % tag)
    print('unsigned: obs %.4f null %.4f p=%.4f | MoM true SD %.3f'
          % (stat_abs, na.mean(), pa, np.sqrt(max(tv, 0))))
    print('signed R-L: obs %+.4f p=%.4f | right better in %d/%d layouts'
          % (stat_sgn, ps, kpos, int(m.sum())))

# episode-level signed test (for reference, anticonservative)
cell_codes, _ = pd.factorize(sub['run'].astype(str) + '|' + sub['layout_id'].astype(str))
z_ep = sub['cluster'].to_numpy().astype(int)
lay_ep, lay_uniq_ep = pd.factorize(sub['layout_id'])
sgn_ep = np.array([sgn_map.get(L, np.nan) for L in lay_uniq_ep])
def layout_deltas_ep(y_, z_):
    s1 = np.bincount(lay_ep, weights=y_ * z_, minlength=len(lay_uniq_ep))
    n1 = np.bincount(lay_ep, weights=z_, minlength=len(lay_uniq_ep))
    s0 = np.bincount(lay_ep, weights=y_ * (1 - z_), minlength=len(lay_uniq_ep))
    n0 = np.bincount(lay_ep, weights=(1 - z_), minlength=len(lay_uniq_ep))
    with np.errstate(invalid='ignore', divide='ignore'):
        return s1 / n1 - s0 / n0, n1, n0
d_ep, n1e, n0e = layout_deltas_ep(y_ep, z_ep)
w_ep = 1.0 / (1.0 / n1e + 1.0 / n0e)
me = ~np.isnan(sgn_ep)
obs_se = float(np.sum((w_ep * sgn_ep * d_ep)[me]) / w_ep[me].sum())
order_by_cell = np.argsort(cell_codes, kind='stable')
zs = z_ep[order_by_cell]
nse = np.empty(NPERM)
for b in range(NPERM):
    u = rng.random(n)
    order = np.lexsort((u, cell_codes))
    zp = np.empty(n, dtype=int); zp[order] = zs
    d, _, _ = layout_deltas_ep(y_ep, zp)
    nse[b] = float(np.sum((w_ep * sgn_ep * d)[me]) / w_ep[me].sum())
print('\nepisode-level signed R-L (anticonservative): obs %+.4f p=%.4f'
      % (obs_se, (np.sum(np.abs(nse) >= abs(obs_se)) + 1) / (NPERM + 1)))

# ---------- validation: |s| vs no-grasp start sink distance ----------
geo = {}
for L in lay_uniq_all:
    g = sub[sub['layout_id'] == L]
    cx = np.corrcoef(g['obj_x_rel'], g['cluster'])[0, 1]
    cy = np.corrcoef(g['obj_y_rel'], g['cluster'])[0, 1]
    geo[L] = 'obj_x_rel' if abs(cx) > abs(cy) else 'obj_y_rel'
abs_s = np.empty(n)
for L in lay_uniq_all:
    idx = sub.index[sub['layout_id'] == L]
    abs_s[idx] = np.abs(sub.loc[idx, geo[L]].to_numpy())
ng = sub['failure_phase'].eq('fail_no_grasp').to_numpy() & np.isfinite(sub['min_sink_dist'].to_numpy())
cors = []
for L in lay_uniq_all:
    idx = (sub['layout_id'] == L).to_numpy() & ng
    if idx.sum() >= 20:
        cors.append(np.corrcoef(abs_s[idx], sub.loc[idx, 'min_sink_dist'])[0, 1])
print('\nper-layout corr(|s|, no-grasp start sink dist): median %.3f, IQR %.3f..%.3f (n=%d layouts)'
      % (np.median(cors), np.quantile(cors, .25), np.quantile(cors, .75), len(cors)))

# ---------- failure-phase composition vs |s| ----------
print('\nfailure-phase mix by |s| bin (row-normalized among failures):')
fails = sub[sub['success'] == 0].copy()
fails['sbin'] = pd.cut(abs_s[fails.index], [0, 0.45, 0.55, 0.65, 0.75, 0.9])
tab = pd.crosstab(fails['sbin'], fails['failure_phase'], normalize='index').round(3)
cnt = fails.groupby('sbin', observed=True).size()
tab['n_fail'] = cnt
print(tab.to_string())
# grasp rate (ever grasped) and success by |s| bin, cell-demeaned
sub['grasped'] = (~sub['failure_phase'].isin(['fail_no_grasp'])).astype(float)
gcell = sub['grasped'] - sub.groupby(['run'])['grasped'].transform('mean')
scell = sub['dm']
sb = pd.cut(pd.Series(abs_s), [0, 0.45, 0.55, 0.65, 0.75, 0.9])
print('\nrun-demeaned grasp rate & success by |s| bin:')
print(pd.DataFrame({'grasp_dm': gcell.groupby(sb, observed=True).mean().round(4),
                    'succ_dm': scell.groupby(sb, observed=True).mean().round(4),
                    'n': sb.groupby(sb, observed=True).size()}).to_string())
lat.to_csv(f'{SP}/side_sinkframe_geometry.csv', index=False)
