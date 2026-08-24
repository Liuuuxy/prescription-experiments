"""Side angle, step 4: signed left/right-of-sink test (sink frame from env probe)
+ decomposition of rim effect into along-counter (|s|, distance-from-sink proxy)
vs depth (|o|) axes + validation of |s| against no-grasp min_sink_dist."""
import json
import numpy as np
import pandas as pd

SP = '/data/xinyua11/tmp/factor_analysis_scratch'
rng = np.random.default_rng(2)
NPERM = 5000

df = pd.read_csv(f'{SP}/pooled_episodes.csv')
clu = pd.read_csv(f'{SP}/side_clusters.csv')
df['cluster'] = clu['cluster'].to_numpy()
bi = pd.read_csv(f'{SP}/side_bimodality.csv').set_index('layout')
CLEAR = bi.index[bi['sep'] > 8].tolist()
df['dm'] = df['success'] - df.groupby('run')['success'].transform('mean')

probe = json.load(open(f'{SP}/side_sinkpos.json'))
try:
    probe.update(json.load(open(f'{SP}/side_sinkpos_rest.json')))
except FileNotFoundError:
    pass
print('probed layouts:', sorted(int(k) for k in probe if 'sink_pos' in probe[k]))

sub = df[df['layout_id'].isin(CLEAR) & df['cluster'].notna()].reset_index(drop=True)
n = len(sub)
lay_codes, lay_uniq = pd.factorize(sub['layout_id'])
cell_codes, _ = pd.factorize(sub['run'].astype(str) + '|' + sub['layout_id'].astype(str))
z = sub['cluster'].to_numpy().astype(int)
NL = len(lay_uniq)
y = sub['dm'].to_numpy()

# ---------- geometry per layout: separating coord s, orth coord o, sink frame ----------
geo = {}
for i, L in enumerate(lay_uniq):
    g = sub[sub['layout_id'] == L]
    cx = np.corrcoef(g['obj_x_rel'], g['cluster'])[0, 1]
    cy = np.corrcoef(g['obj_y_rel'], g['cluster'])[0, 1]
    sep_col = 'obj_x_rel' if abs(cx) > abs(cy) else 'obj_y_rel'
    orth_col = 'obj_y_rel' if sep_col == 'obj_x_rel' else 'obj_x_rel'
    # world direction of increasing sep coordinate:
    Xw = g[['obj_x_abs', 'obj_y_abs']].to_numpy()
    s_raw = g[sep_col].to_numpy()
    # regress world coords on s_raw -> direction
    s_c = s_raw - s_raw.mean()
    vx = np.dot(Xw[:, 0] - Xw[:, 0].mean(), s_c) / np.dot(s_c, s_c)
    vy = np.dot(Xw[:, 1] - Xw[:, 1].mean(), s_c) / np.dot(s_c, s_c)
    v = np.array([vx, vy]); v = v / np.linalg.norm(v)
    c1 = Xw[g['cluster'] == 1].mean(0); c0 = Xw[g['cluster'] == 0].mean(0)
    mid = (c1 + c0) / 2
    entry = dict(sep_col=sep_col, orth_col=orth_col, v_world=v, mid=mid, c0=c0, c1=c1)
    pk = str(L)
    if pk in probe and 'sink_pos' in probe[pk]:
        sp = np.array(probe[pk]['sink_pos'][:2])
        yaw = probe[pk]['sink_rot'][0]
        R = np.array([[np.cos(-yaw), -np.sin(-yaw)], [np.sin(-yaw), np.cos(-yaw)]])
        entry['sink_xy'] = sp
        entry['yaw'] = yaw
        entry['c0_local'] = R @ (c0 - sp)
        entry['c1_local'] = R @ (c1 - sp)
        entry['mid_local'] = R @ (mid - sp)
    geo[L] = entry

# ---------- permutation machinery (same as step 3) ----------
order_by_cell = np.argsort(cell_codes, kind='stable')
z_sorted = z[order_by_cell]

def perm_z():
    u = rng.random(n)
    order = np.lexsort((u, cell_codes))
    zp = np.empty(n, dtype=int)
    zp[order] = z_sorted
    return zp

def layout_deltas(yv, zz):
    s1 = np.bincount(lay_codes, weights=yv * zz, minlength=NL)
    n1 = np.bincount(lay_codes, weights=zz, minlength=NL)
    s0 = np.bincount(lay_codes, weights=yv * (1 - zz), minlength=NL)
    n0 = np.bincount(lay_codes, weights=(1 - zz), minlength=NL)
    with np.errstate(invalid='ignore', divide='ignore'):
        return s1 / n1 - s0 / n0, n1, n0

d_obs, n1, n0 = layout_deltas(y, z)
w = 1.0 / (1.0 / n1 + 1.0 / n0)

# sign vector: +1 if cluster1 is RIGHT of sink (local x > 0)
sgn = np.full(NL, np.nan)
print('\nlayout geometry (sink frame, lateral = local x):')
print('%6s %10s %10s %10s %8s' % ('layout', 'c0_lx', 'c1_lx', 'mid_lx', 'mid_ly'))
for i, L in enumerate(lay_uniq):
    e = geo[L]
    if 'c0_local' not in e:
        continue
    l0, l1 = e['c0_local'], e['c1_local']
    print('%6d %10.2f %10.2f %10.2f %8.2f' % (L, l0[0], l1[0], e['mid_local'][0], e['mid_local'][1]))
    # require clusters on opposite lateral sides for a clean L/R call
    if np.sign(l0[0]) != np.sign(l1[0]):
        sgn[i] = 1.0 if l1[0] > 0 else -1.0

m = ~np.isnan(sgn)
print('\nlayouts with clean opposite-side geometry: %d' % m.sum())

def stat_signed(d, sign):
    mm = ~np.isnan(sign)
    return float(np.sum((w * sign * d)[mm]) / w[mm].sum())

obs_lr = stat_signed(d_obs, sgn)
null_lr = np.empty(NPERM)
for b in range(NPERM):
    dnull, _, _ = layout_deltas(y, perm_z())
    null_lr[b] = stat_signed(dnull, sgn)
p_lr = (np.sum(np.abs(null_lr) >= abs(obs_lr)) + 1) / (NPERM + 1)
kpos = int(np.sum(sgn[m] * d_obs[m] > 0))
print('\n=== (3) signed RIGHT-minus-LEFT of sink stat ===')
print('observed = %+.4f, two-sided perm p = %.4f; right better in %d/%d layouts'
      % (obs_lr, p_lr, kpos, m.sum()))

# ---------- rim decomposition: |s| (along counter, dist-from-sink) vs |o| (depth) ----------
s_arr = np.empty(n); o_arr = np.empty(n)
for L in lay_uniq:
    idx = sub.index[sub['layout_id'] == L]
    e = geo[L]
    s_arr[idx] = sub.loc[idx, e['sep_col']].to_numpy()
    o_arr[idx] = sub.loc[idx, e['orth_col']].to_numpy()
abs_s = np.abs(s_arr); abs_o = np.abs(o_arr)

# validate |s| & |o| against min_sink_dist on no-grasp episodes (object unmoved)
ngm = (sub['failure_phase'] == 'fail_no_grasp').to_numpy()
msd = sub['min_sink_dist'].to_numpy()
ok = ngm & np.isfinite(msd)
cs = np.corrcoef(abs_s[ok], msd[ok])[0, 1]
co = np.corrcoef(abs_o[ok], msd[ok])[0, 1]
print('\nvalidation on %d no-grasp eps: corr(|s|, start sink dist)=%.3f, corr(|o|, .)=%.3f' % (ok.sum(), cs, co))

def fe_multireg(yv, X, groups):
    gser = pd.Series(groups)
    Xc = np.column_stack([x - pd.Series(x).groupby(gser).transform('mean').to_numpy() for x in X.T])
    yc = yv - pd.Series(yv).groupby(gser).transform('mean').to_numpy()
    beta, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
    resid = yc - Xc @ beta
    XtXinv = np.linalg.inv(Xc.T @ Xc)
    se = np.sqrt(np.diag(XtXinv) * (resid @ resid) / (len(yv) - Xc.shape[1] - len(set(groups))))
    return beta, se

X2 = np.column_stack([(abs_s > 0.65).astype(float), (abs_o > 0.65).astype(float)])
b2, se2 = fe_multireg(y, X2, cell_codes)
print('\n=== rim decomposition (run x layout FE, joint) ===')
print('I(|s|>0.65) [far-from-sink along counter]: %+.4f (se %.4f)' % (b2[0], se2[0]))
print('I(|o|>0.65) [counter depth edge]:          %+.4f (se %.4f)' % (b2[1], se2[1]))
X3 = np.column_stack([abs_s, abs_o])
b3, se3 = fe_multireg(y, X3, cell_codes)
print('linear |s|: %+.4f (se %.4f); linear |o|: %+.4f (se %.4f)' % (b3[0], se3[0], b3[1], se3[1]))

# permutation p for the CONTRAST b(|s|) - b(|o|) under swapping s/o roles per episode? simpler:
# run-stratified bootstrap CIs for difference
B = 1000
diffs_t = np.empty(B); diffs_l = np.empty(B)
runs_codes, _ = pd.factorize(sub['run'])
by_run = [np.where(runs_codes == rr)[0] for rr in range(runs_codes.max() + 1)]
for b in range(B):
    idx = np.concatenate([r[rng.integers(0, len(r), len(r))] for r in by_run])
    yv = y[idx]
    cc = cell_codes[idx]
    bt, _ = fe_multireg(yv, X2[idx], cc)
    bl, _ = fe_multireg(yv, X3[idx], cc)
    diffs_t[b] = bt[0] - bt[1]
    diffs_l[b] = bl[0] - bl[1]
print('\nb(|s|>.65)-b(|o|>.65): %.4f, boot 95%% CI [%.4f, %.4f]'
      % (b2[0] - b2[1], np.quantile(diffs_t, 0.025), np.quantile(diffs_t, 0.975)))
print('b(|s|)-b(|o|) linear:  %.4f, boot 95%% CI [%.4f, %.4f]'
      % (b3[0] - b3[1], np.quantile(diffs_l, 0.025), np.quantile(diffs_l, 0.975)))

# mid-point of union vs sink location (is sink centered between the two regions?)
lat_mid = [abs(geo[L]['mid_local'][0]) for L in lay_uniq if 'mid_local' in geo[L]]
print('\n|lateral offset of union midpoint from sink center|: median %.2f m (n=%d)'
      % (np.median(lat_mid), len(lat_mid)))
pd.DataFrame([dict(layout=int(L), sep_col=geo[L]['sep_col'],
                   c0_lx=geo[L].get('c0_local', [np.nan])[0],
                   c1_lx=geo[L].get('c1_local', [np.nan])[0],
                   delta=d_obs[i], w=w[i], sign_right1=sgn[i])
              for i, L in enumerate(lay_uniq)]).to_csv(f'{SP}/side_signed_table.csv', index=False)
