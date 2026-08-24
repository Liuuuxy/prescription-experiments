"""Side angle, step 5:
A) config-level permutation test of the side effect (correct unit of analysis:
   same eval config repeats across runs; permute side at config level within layout).
B) rim reinterpretation: dm vs binned |s| (inner->outer along counter) with run x layout FE;
   joint FE regression on s_in and o_in; per-layout validation corr(|s|, no-grasp sink dist).
"""
import numpy as np
import pandas as pd

SP = '/data/xinyua11/tmp/factor_analysis_scratch'
rng = np.random.default_rng(3)
NPERM = 5000

df = pd.read_csv(f'{SP}/pooled_episodes.csv')
clu = pd.read_csv(f'{SP}/side_clusters.csv')
df['cluster'] = clu['cluster'].to_numpy()
bi = pd.read_csv(f'{SP}/side_bimodality.csv').set_index('layout')
CLEAR = bi.index[bi['sep'] > 8].tolist()
df['dm'] = df['success'] - df.groupby('run')['success'].transform('mean')
sub = df[df['layout_id'].isin(CLEAR) & df['cluster'].notna()].reset_index(drop=True)

# ---------- A) config-level test ----------
sub['cfg'] = (sub['layout_id'].astype(str) + '|' + sub['style_id'].astype(str) + '|'
              + sub['object_category'] + '|' + sub['obj_x_abs'].round(3).astype(str)
              + '|' + sub['obj_y_abs'].round(3).astype(str))
cfg = sub.groupby('cfg').agg(y=('dm', 'mean'), nrep=('dm', 'size'),
                             layout=('layout_id', 'first'), side=('cluster', 'first'),
                             side_check=('cluster', 'nunique')).reset_index()
assert (cfg['side_check'] == 1).all()
print('configs: %d (of %d eps); repeated >1 run: %d' % (len(cfg), len(sub), (cfg['nrep'] > 1).sum()))

lay_codes, lay_uniq = pd.factorize(cfg['layout'])
NL = len(lay_uniq)
z = cfg['side'].to_numpy().astype(int)
yv = cfg['y'].to_numpy()
wrep = cfg['nrep'].to_numpy().astype(float)   # weight configs by #episodes

def layout_deltas_cfg(y_, z_):
    s1 = np.bincount(lay_codes, weights=wrep * y_ * z_, minlength=NL)
    n1 = np.bincount(lay_codes, weights=wrep * z_, minlength=NL)
    s0 = np.bincount(lay_codes, weights=wrep * y_ * (1 - z_), minlength=NL)
    n0 = np.bincount(lay_codes, weights=wrep * (1 - z_), minlength=NL)
    with np.errstate(invalid='ignore', divide='ignore'):
        return s1 / n1 - s0 / n0, n1, n0

d_obs, n1, n0 = layout_deltas_cfg(yv, z)
w = 1.0 / (1.0 / n1 + 1.0 / n0)
stat = lambda d: float(np.sum(w * np.abs(d)) / w.sum())
obs = stat(d_obs)

# permute side labels among configs within layout
order_by_lay = np.argsort(lay_codes, kind='stable')
z_sorted = z[order_by_lay]
null = np.empty(NPERM)
null_d = np.empty((NPERM, NL))
for b in range(NPERM):
    u = rng.random(len(cfg))
    order = np.lexsort((u, lay_codes))
    zp = np.empty(len(cfg), dtype=int)
    zp[order] = z_sorted
    d, _, _ = layout_deltas_cfg(yv, zp)
    null_d[b] = d
    null[b] = stat(d)
p = (np.sum(null >= obs) + 1) / (NPERM + 1)
print('\n=== A) CONFIG-level unsigned side test (%d layouts) ===' % NL)
print('observed weighted mean|delta| = %.4f  null mean = %.4f (95pct %.4f)  p = %.4f'
      % (obs, null.mean(), np.quantile(null, 0.95), p))
noise = null_d.var(axis=0)
tv = float(np.sum(w * (d_obs ** 2 - noise)) / w.sum())
print('MoM true SD of side gap (config level) = %.3f' % np.sqrt(max(tv, 0)))
pl = pd.DataFrame({'layout': lay_uniq, 'delta_cfg': d_obs,
                   'p_layout': [(np.sum(np.abs(null_d[:, i]) >= abs(d_obs[i])) + 1) / (NPERM + 1)
                                for i in range(NL)]}).sort_values('p_layout')
print('top layouts:'); print(pl.head(8).round(3).to_string(index=False))
pl.to_csv(f'{SP}/side_cfg_layout_deltas.csv', index=False)

# ---------- B) rim reinterpretation ----------
# sep/orth coords per layout
geo = {}
for L in lay_uniq:
    g = sub[sub['layout_id'] == L]
    cx = np.corrcoef(g['obj_x_rel'], g['cluster'])[0, 1]
    cy = np.corrcoef(g['obj_y_rel'], g['cluster'])[0, 1]
    geo[L] = ('obj_x_rel', 'obj_y_rel') if abs(cx) > abs(cy) else ('obj_y_rel', 'obj_x_rel')
n = len(sub)
s_arr = np.empty(n); o_arr = np.empty(n)
for L in lay_uniq:
    idx = sub.index[sub['layout_id'] == L]
    s_arr[idx] = sub.loc[idx, geo[L][0]].to_numpy()
    o_arr[idx] = sub.loc[idx, geo[L][1]].to_numpy()
abs_s = np.abs(s_arr)
print('\n|s| range: %.2f..%.2f;  |o| range: %.2f..%.2f; frac |o|>0.65 = %.4f; frac |s|>0.65 = %.3f'
      % (abs_s.min(), abs_s.max(), np.abs(o_arr).min(), np.abs(o_arr).max(),
         (np.abs(o_arr) > 0.65).mean(), (abs_s > 0.65).mean()))

# per-layout-side min-max normalized coords: s_in 0=inner(sink) edge, 1=outer edge
s_in = np.empty(n); o_in = np.empty(n)
for (L, c), g in sub.groupby(['layout_id', 'cluster']):
    idx = g.index
    a = abs_s[idx]
    s_in[idx] = (a - a.min()) / (a.max() - a.min())
    b_ = o_arr[idx]
    o_in[idx] = (b_ - b_.min()) / (b_.max() - b_.min())

y = sub['dm'].to_numpy()
cell_codes, _ = pd.factorize(sub['run'].astype(str) + '|' + sub['layout_id'].astype(str))

def fe_reg(yv_, X, groups):
    gser = pd.Series(groups)
    Xc = np.column_stack([x - pd.Series(x).groupby(gser).transform('mean').to_numpy() for x in X.T])
    yc = yv_ - pd.Series(yv_).groupby(gser).transform('mean').to_numpy()
    beta, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
    resid = yc - Xc @ beta
    XtXinv = np.linalg.pinv(Xc.T @ Xc)
    dof = len(yv_) - Xc.shape[1] - len(set(groups))
    se = np.sqrt(np.diag(XtXinv) * (resid @ resid) / dof)
    return beta, se

bJ, seJ = fe_reg(y, np.column_stack([s_in, o_in]), cell_codes)
print('\n=== B) joint FE regression (run x layout FE) on normalized in-region coords ===')
print('s_in (inner->outer along counter): %+.4f (se %.4f)' % (bJ[0], seJ[0]))
print('o_in (across depth band):          %+.4f (se %.4f)' % (bJ[1], seJ[1]))

# binned curve for |s| with FE: demean within cells, bin |s|
BINS = [0.25, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.9]
yc = y - pd.Series(y).groupby(pd.Series(cell_codes)).transform('mean').to_numpy()
lab = np.digitize(abs_s, BINS)
print('\nbinned |s| (cell-demeaned dm):')
for bidx in sorted(set(lab)):
    mm = lab == bidx
    lo = BINS[bidx - 1] if bidx > 0 else abs_s.min()
    hi = BINS[bidx] if bidx < len(BINS) else abs_s.max()
    print('  |s| %.2f-%.2f: n=%4d  mean=%+.4f (se %.4f)'
          % (lo, hi, mm.sum(), yc[mm].mean(), yc[mm].std() / np.sqrt(mm.sum())))
print('\nbinned s_in (0=sink-side inner edge, 1=outer edge):')
for lo, hi in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.001)]:
    mm = (s_in >= lo) & (s_in < hi)
    print('  s_in %.1f-%.1f: n=%4d  mean=%+.4f (se %.4f)'
          % (lo, hi, mm.sum(), yc[mm].mean(), yc[mm].std() / np.sqrt(mm.sum())))
print('\nbinned o_in (depth band position):')
for lo, hi in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.001)]:
    mm = (o_in >= lo) & (o_in < hi)
    print('  o_in %.1f-%.1f: n=%4d  mean=%+.4f (se %.4f)'
          % (lo, hi, mm.sum(), yc[mm].mean(), yc[mm].std() / np.sqrt(mm.sum())))

# validation: per-layout corr(|s|, min_sink_dist) on no-grasp episodes
cors_s, cors_o = [], []
ng = sub[sub['failure_phase'] == 'fail_no_grasp']
for L, g in ng.groupby('layout_id'):
    idx = g.index
    if len(idx) < 20:
        continue
    cors_s.append(np.corrcoef(abs_s[idx], g['min_sink_dist'])[0, 1])
    cors_o.append(np.corrcoef(o_in[idx], g['min_sink_dist'])[0, 1])
print('\nper-layout corr(|s|, no-grasp start sink-dist): median %.3f (IQR %.3f..%.3f, n=%d layouts)'
      % (np.median(cors_s), np.quantile(cors_s, .25), np.quantile(cors_s, .75), len(cors_s)))
print('per-layout corr(o_in, no-grasp sink-dist): median %.3f' % np.median(cors_o))

# does the |s|>0.65 threshold effect equal the established rim effect?
b1, se1 = fe_reg(y, np.column_stack([(abs_s > 0.65).astype(float)]), cell_codes)
rim = np.maximum(np.abs(sub['obj_x_rel']), np.abs(sub['obj_y_rel'])).to_numpy()
b0, se0 = fe_reg(y, np.column_stack([(rim > 0.65).astype(float)]), cell_codes)
print('\nI(rim>0.65) FE coef (established form): %+.4f (se %.4f)' % (b0[0], se0[0]))
print('I(|s|>0.65) FE coef (sep axis only):    %+.4f (se %.4f)' % (b1[0], se1[0]))
print('frac of rim>0.65 episodes where the exceeding coord IS s: %.3f'
      % ((abs_s > 0.65) | (rim <= 0.65)).mean())
agree = ((rim > 0.65) == (abs_s > 0.65)).mean()
print('agreement I(rim>.65)==I(|s|>.65): %.4f' % agree)
