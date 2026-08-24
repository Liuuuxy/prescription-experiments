"""Per-policy-family signed L/R effect + summary figure."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SP = '/data/xinyua11/tmp/factor_analysis_scratch'
rng = np.random.default_rng(5)
NPERM = 3000

df = pd.read_csv(f'{SP}/pooled_episodes.csv')
clu = pd.read_csv(f'{SP}/side_clusters.csv')
df['cluster'] = clu['cluster'].to_numpy()
bi = pd.read_csv(f'{SP}/side_bimodality.csv').set_index('layout')
CLEAR = bi.index[bi['sep'] > 8].tolist()
df['dm'] = df['success'] - df.groupby('run')['success'].transform('mean')
sub = df[df['layout_id'].isin(CLEAR) & df['cluster'].notna()].reset_index(drop=True)
print('policy families:', sub['policy'].value_counts().to_dict())

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
    if np.sign(l0[0]) != np.sign(l1[0]):
        sgn_map[L] = 1.0 if l1[0] > 0 else -1.0

sub['cfg'] = (sub['layout_id'].astype(str) + '|' + sub['style_id'].astype(str) + '|'
              + sub['object_category'] + '|' + sub['obj_x_abs'].round(3).astype(str)
              + '|' + sub['obj_y_abs'].round(3).astype(str))

def config_signed(d):
    cfg = d.groupby('cfg').agg(y=('dm', 'mean'), nrep=('dm', 'size'),
                               layout=('layout_id', 'first'), side=('cluster', 'first')).reset_index()
    lay_codes, lay_uniq = pd.factorize(cfg['layout'])
    NL = len(lay_uniq)
    z = cfg['side'].to_numpy().astype(int)
    wrep = cfg['nrep'].to_numpy().astype(float)
    sgn = np.array([sgn_map.get(L, np.nan) for L in lay_uniq])
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
    m = ~np.isnan(sgn) & np.isfinite(d_obs)
    obs = float(np.sum((w * sgn * d_obs)[m]) / w[m].sum())
    order = np.argsort(lay_codes, kind='stable'); zs = z[order]
    null = np.empty(NPERM)
    for b in range(NPERM):
        u = rng.random(len(cfg))
        o2 = np.lexsort((u, lay_codes))
        zp = np.empty(len(cfg), dtype=int); zp[o2] = zs
        dn, _, _ = deltas(zp)
        null[b] = float(np.sum((w * sgn * dn)[m]) / w[m].sum())
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (NPERM + 1)
    return obs, p, int(m.sum()), d_obs, w, sgn, m, lay_uniq

print('\nsigned R-L (config level) by policy family:')
for pol, d in sub.groupby('policy'):
    obs, p, nl, *_ = config_signed(d)
    print('  %-12s n=%5d: %+.4f (p=%.4f, %d layouts)' % (pol, len(d), obs, p, nl))
obs_all, p_all, nl_all, d_obs, w, sgn, m, lay_uniq = config_signed(sub)
print('  ALL          n=%5d: %+.4f (p=%.4f)' % (len(sub), obs_all, p_all))

# figure
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axes[0]
g = sub[sub['layout_id'] == 15]
for c, col in ((0, 'tab:blue'), (1, 'tab:red')):
    gg = g[g['cluster'] == c]
    ax.scatter(gg['obj_x_abs'], gg['obj_y_abs'], s=8, c=col, label=f'cluster {c}')
e = probe['15']; spx = e['sink_pos']
ax.scatter([spx[0]], [spx[1]], marker='*', s=250, c='k', label='sink')
ax.set_title('layout 15: two placement regions flank sink')
ax.set_xlabel('x_abs (m)'); ax.set_ylabel('y_abs (m)'); ax.legend(fontsize=8); ax.axis('equal')

ax = axes[1]
dd = (sgn * d_obs)[m]
order = np.argsort(dd)
ax.barh(range(len(dd)), dd[order], color=['tab:green' if v < 0 else 'tab:orange' for v in dd[order]])
ax.axvline(0, c='k', lw=0.8); ax.axvline(obs_all, c='b', ls='--', label=f'weighted mean {obs_all:+.3f}')
ax.set_title('per-layout SR gap: right-of-sink minus left\n(config level, run-demeaned)')
ax.set_xlabel('SR gap'); ax.legend(fontsize=8)

ax = axes[2]
# |s| gradient curve (cell-demeaned) recomputed here
geo = {}
for L in sorted(sub['layout_id'].unique()):
    g = sub[sub['layout_id'] == L]
    cx = np.corrcoef(g['obj_x_rel'], g['cluster'])[0, 1]
    cy = np.corrcoef(g['obj_y_rel'], g['cluster'])[0, 1]
    geo[L] = 'obj_x_rel' if abs(cx) > abs(cy) else 'obj_y_rel'
abs_s = np.empty(len(sub))
for L, col in geo.items():
    idx = sub.index[sub['layout_id'] == L]
    abs_s[idx] = np.abs(sub.loc[idx, col].to_numpy())
cell, _ = pd.factorize(sub['run'].astype(str) + '|' + sub['layout_id'].astype(str))
yc = sub['dm'] - sub.groupby(pd.Series(cell))['dm'].transform('mean')
bins = np.array([0.2, 0.35, 0.45, 0.55, 0.65, 0.7, 0.75, 0.8, 0.9])
mids, means, ses = [], [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    mm = (abs_s >= lo) & (abs_s < hi)
    if mm.sum() > 30:
        mids.append((lo + hi) / 2); means.append(yc[mm].mean()); ses.append(yc[mm].std() / np.sqrt(mm.sum()))
ax.errorbar(mids, means, yerr=ses, marker='o')
ax.axhline(0, c='k', lw=0.8)
ax.set_title('success vs |s| = distance from sink along counter\n(run x layout demeaned) — the "rim" IS this axis')
ax.set_xlabel('|s| (normalized along-counter dist from sink)'); ax.set_ylabel('demeaned SR')
plt.tight_layout()
plt.savefig(f'{SP}/side_fig.png', dpi=130)
print('saved fig')
