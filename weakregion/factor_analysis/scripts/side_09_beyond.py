"""Config-level test of layout-specific side variation BEYOND the common L/R shift:
stat = weighted mean |signed_delta - weighted mean signed_delta| on residualized outcomes."""
import json
import numpy as np
import pandas as pd

SP = '/data/xinyua11/tmp/factor_analysis_scratch'
rng = np.random.default_rng(6)
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
sgn_map = {}
for L in sorted(sub['layout_id'].unique()):
    g = sub[sub['layout_id'] == L]
    Xw = g[['obj_x_abs', 'obj_y_abs']].to_numpy()
    c1 = Xw[g['cluster'] == 1].mean(0); c0 = Xw[g['cluster'] == 0].mean(0)
    e = probe[str(L)]
    sp = np.array(e['sink_pos'][:2]); yaw = e['sink_rot'][0]
    R = np.array([[np.cos(-yaw), -np.sin(-yaw)], [np.sin(-yaw), np.cos(-yaw)]])
    l0, l1 = R @ (c0 - sp), R @ (c1 - sp)
    sgn_map[L] = 1.0 if l1[0] > 0 else -1.0

# residualize category + quad + rim
n = len(sub)
xr = sub['obj_x_rel'].to_numpy(); yr = sub['obj_y_rel'].to_numpy()
rim = np.maximum(np.abs(xr), np.abs(yr))
cat_codes, _ = pd.factorize(sub['object_category'])
y_ep = sub['dm'].to_numpy()
y_r = y_ep - pd.Series(y_ep).groupby(pd.Series(cat_codes)).transform('mean').to_numpy()
Fq = np.column_stack([np.ones(n), xr, yr, xr**2, yr**2, xr*yr, rim, (rim > 0.65).astype(float)])
beta, *_ = np.linalg.lstsq(Fq, y_r, rcond=None)
sub['dmr'] = y_r - Fq @ beta

sub['cfg'] = (sub['layout_id'].astype(str) + '|' + sub['style_id'].astype(str) + '|'
              + sub['object_category'] + '|' + sub['obj_x_abs'].round(3).astype(str)
              + '|' + sub['obj_y_abs'].round(3).astype(str))
cfg = sub.groupby('cfg').agg(y=('dmr', 'mean'), nrep=('dmr', 'size'),
                             layout=('layout_id', 'first'), side=('cluster', 'first')).reset_index()
lay_codes, lay_uniq = pd.factorize(cfg['layout'])
NL = len(lay_uniq)
z = cfg['side'].to_numpy().astype(int)
wrep = cfg['nrep'].to_numpy().astype(float)
sgn = np.array([sgn_map[L] for L in lay_uniq])
yv = cfg['y'].to_numpy()

def deltas(z_):
    s1 = np.bincount(lay_codes, weights=wrep * yv * z_, minlength=NL)
    n1 = np.bincount(lay_codes, weights=wrep * z_, minlength=NL)
    s0 = np.bincount(lay_codes, weights=wrep * yv * (1 - z_), minlength=NL)
    n0 = np.bincount(lay_codes, weights=wrep * (1 - z_), minlength=NL)
    with np.errstate(invalid='ignore', divide='ignore'):
        return s1 / n1 - s0 / n0, n1, n0

d_obs, n1, n0 = deltas(z)
w = 1.0 / (1.0 / n1 + 1.0 / n0)

def stat_beyond(d):
    ds = sgn * d
    mu = float(np.sum(w * ds) / w.sum())
    return float(np.sum(w * np.abs(ds - mu)) / w.sum()), mu

obs, mu_obs = stat_beyond(d_obs)
order = np.argsort(lay_codes, kind='stable'); zs = z[order]
null = np.empty(NPERM)
for b in range(NPERM):
    u = rng.random(len(cfg))
    o2 = np.lexsort((u, lay_codes))
    zp = np.empty(len(cfg), dtype=int); zp[o2] = zs
    dn, _, _ = deltas(zp)
    null[b], _ = stat_beyond(dn)
p = (np.sum(null >= obs) + 1) / (NPERM + 1)
print('common signed mean (resid, config) = %+.4f' % mu_obs)
print('BEYOND-common spread: obs %.4f, null mean %.4f (95pct %.4f), p = %.4f'
      % (obs, null.mean(), np.quantile(null, 0.95), p))
tv = obs2 = float(np.sum(w * ((sgn * d_obs - mu_obs) ** 2)) / w.sum())
noise = null.var()  # not directly comparable; report percentile only
