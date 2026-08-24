"""Angle: start-location. Part 1: SR by 3x3 region with run control; Part 4: failure-phase mix by region."""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
SP = '/data/xinyua11/tmp/factor_analysis_scratch'
df = pd.read_csv(f'{SP}/pooled_episodes.csv')
df['success'] = df['success'].astype(int)
df['run_mean'] = df.groupby('run')['success'].transform('mean')
df['dm'] = df['success'] - df['run_mean']

reg = df.dropna(subset=['region']).copy()
reg['depth'] = reg.region.str.split('-').str[0]   # near/mid/far
reg['col'] = reg.region.str.split('-').str[1]     # left/center/right
reg['dcode'] = reg.depth.map({'near': 0, 'mid': 1, 'far': 2}).astype(float)
reg['ccode'] = reg.col.map({'left': -1, 'center': 0, 'right': 1}).astype(float)
reg['edge'] = (reg.ccode != 0).astype(float)
grand = reg.success.mean()
print(f'region subset: {len(reg)} rows, {reg.run.nunique()} runs, grand SR {grand:.3f}')

# ---- adjusted SR by region ----
tab = reg.groupby('region').agg(n=('dm', 'size'), dm=('dm', 'mean'), raw=('success', 'mean'))
tab['adjSR'] = grand + tab['dm']
# stratified-within-run bootstrap CI for each region's adjusted SR
B = 3000
run_groups = {r: g.index.values for r, g in reg.groupby('run')}
boot_dm = {rgn: [] for rgn in tab.index}
idx_by_run = list(run_groups.values())
region_arr = reg['region'].values
dm_arr = reg['dm'].values
pos = {ix: k for k, ix in enumerate(reg.index)}
run_pos = [np.array([pos[i] for i in ixs]) for ixs in idx_by_run]
for b in range(B):
    samp = np.concatenate([p[rng.integers(0, len(p), len(p))] for p in run_pos])
    s_reg, s_dm = region_arr[samp], dm_arr[samp]
    dfb = pd.DataFrame({'r': s_reg, 'd': s_dm}).groupby('r')['d'].mean()
    for rgn in tab.index:
        boot_dm[rgn].append(dfb.get(rgn, np.nan))
for rgn in tab.index:
    lo, hi = np.nanpercentile(boot_dm[rgn], [2.5, 97.5])
    tab.loc[rgn, 'adj_lo'] = grand + lo
    tab.loc[rgn, 'adj_hi'] = grand + hi
print(tab[['n', 'raw', 'adjSR', 'adj_lo', 'adj_hi']].round(3))

# ---- contrasts with stratified permutation ----
def stat_all(d):
    far = d.dm[d.depth == 'far'].mean(); near = d.dm[d.depth == 'near'].mean()
    right = d.dm[d.col == 'right'].mean(); left = d.dm[d.col == 'left'].mean()
    edge = d.dm[d.edge == 1].mean(); cen = d.dm[d.edge == 0].mean()
    # OLS slopes on codes
    sd = np.polyfit(d.dcode, d.dm, 1)[0]
    sc = np.polyfit(d.ccode, d.dm, 1)[0]
    return np.array([far - near, sd, right - left, sc, edge - cen])

obs = stat_all(reg)
names = ['far_minus_near', 'depth_slope_per_band', 'right_minus_left', 'lr_slope_per_band', 'edge_minus_center']
NP = 10000
perm = np.zeros((NP, 5))
work = reg[['dm', 'depth', 'col', 'dcode', 'ccode', 'edge', 'run']].copy()
runs_idx = [np.where(work.run.values == r)[0] for r in work.run.unique()]
labels = work[['depth', 'col', 'dcode', 'ccode', 'edge']].values
dmv = work.dm.values
for p in range(NP):
    lab = labels.copy()
    for ix in runs_idx:
        lab[ix] = lab[ix[rng.permutation(len(ix))]]
    d = pd.DataFrame({'dm': dmv, 'depth': lab[:, 0], 'col': lab[:, 1],
                      'dcode': lab[:, 2].astype(float), 'ccode': lab[:, 3].astype(float),
                      'edge': lab[:, 4].astype(float)})
    perm[p] = stat_all(d)
print('\ncontrast  effect  perm_p(two-sided)  boot95CI')
# bootstrap CIs for contrasts (stratified within-run)
cboot = np.zeros((B, 5))
depth_arr = reg.depth.values; col_arr = reg.col.values
dc = reg.dcode.values; cc = reg.ccode.values; ed = reg.edge.values
for b in range(B):
    samp = np.concatenate([p[rng.integers(0, len(p), len(p))] for p in run_pos])
    d = pd.DataFrame({'dm': dm_arr[samp], 'depth': depth_arr[samp], 'col': col_arr[samp],
                      'dcode': dc[samp], 'ccode': cc[samp], 'edge': ed[samp]})
    cboot[b] = stat_all(d)
for k, nm in enumerate(names):
    pv = (np.sum(np.abs(perm[:, k]) >= abs(obs[k])) + 1) / (NP + 1)
    lo, hi = np.percentile(cboot[:, k], [2.5, 97.5])
    print(f'{nm:22s} {obs[k]:+.4f}  p={pv:.4f}  [{lo:+.4f},{hi:+.4f}]')

# per-run effect distribution (run-level view)
per_run = reg.groupby('run').apply(
    lambda g: pd.Series({
        'far_near': g.success[g.depth == 'far'].mean() - g.success[g.depth == 'near'].mean(),
        'edge_cen': g.success[g.edge == 1].mean() - g.success[g.edge == 0].mean()}), include_groups=False)
print('\nper-run far-near: mean %.3f sd %.3f, %d/%d runs negative' % (
    per_run.far_near.mean(), per_run.far_near.std(), (per_run.far_near < 0).sum(), len(per_run)))
print('per-run edge-center: mean %.3f sd %.3f, %d/%d runs negative' % (
    per_run.edge_cen.mean(), per_run.edge_cen.std(), (per_run.edge_cen < 0).sum(), len(per_run)))

# ---- Part 4: failure phase mix by region ----
fails = reg[reg.success == 0].copy()
fails['ng'] = (fails.failure_phase == 'fail_no_grasp').astype(float)
mix = fails.groupby(['region', 'failure_phase']).size().unstack(fill_value=0)
mixp = mix.div(mix.sum(1), axis=0)
print('\nfailure phase mix by region:\n', mixp.round(3))
print('\nno-grasp share of failures by depth:\n', fails.groupby('depth')['ng'].agg(['mean', 'size']))
print('no-grasp share by col:\n', fails.groupby('col')['ng'].agg(['mean', 'size']))
# permutation: no-grasp share far-near and edge-center among failures, stratified by run
fw = fails[['ng', 'depth', 'edge', 'run']].copy()
fruns = [np.where(fw.run.values == r)[0] for r in fw.run.unique()]
flab = fw[['depth', 'edge']].values
ngv = fw.ng.values
def fstat(dep, edg, ng):
    a = ng[dep == 'far'].mean() - ng[dep == 'near'].mean()
    b = ng[edg == 1].mean() - ng[edg == 0].mean()
    return np.array([a, b])
fobs = fstat(flab[:, 0], flab[:, 1].astype(float), ngv)
fperm = np.zeros((NP, 2))
for p in range(NP):
    lab = flab.copy()
    for ix in fruns:
        lab[ix] = lab[ix[rng.permutation(len(ix))]]
    fperm[p] = fstat(lab[:, 0], lab[:, 1].astype(float), ngv)
for k, nm in enumerate(['ng_share far-near', 'ng_share edge-center']):
    pv = (np.sum(np.abs(fperm[:, k]) >= abs(fobs[k])) + 1) / (NP + 1)
    print(f'{nm}: {fobs[k]:+.4f} p={pv:.4f}')
# max_lift among failures
print('\nmax_lift among failures by depth:\n', fails.groupby('depth')['max_lift'].mean().round(3))
print('never-lifted (<2cm) share of failures by depth:\n',
      fails.groupby('depth').apply(lambda g: (g.max_lift < 0.02).mean(), include_groups=False).round(3))
# sanity: what does near/far mean physically? start->sink distance proxy from no-grasp failures
ngf = fails[(fails.failure_phase == 'fail_no_grasp') & (fails.max_lift < 0.02)]
print('\nstart-to-sink dist proxy (min_sink_dist of untouched no-grasp fails) by depth:\n',
      ngf.groupby('depth')['min_sink_dist'].agg(['mean', 'size']).round(3))
print('by col:\n', ngf.groupby('col')['min_sink_dist'].agg(['mean', 'size']).round(3))

# ---- plots ----
order = [f'{d}-{c}' for d in ['near', 'mid', 'far'] for c in ['left', 'center', 'right']]
t = tab.loc[order]
fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(9)
ax.bar(x, t.adjSR, yerr=[t.adjSR - t.adj_lo, t.adj_hi - t.adjSR], capsize=3,
       color=['#4878cf'] * 3 + ['#6acc65'] * 3 + ['#d65f5f'] * 3)
ax.axhline(grand, ls='--', c='k', lw=1, label=f'grand SR {grand:.3f}')
ax.set_xticks(x); ax.set_xticklabels(order, rotation=45, ha='right')
ax.set_ylabel('run-adjusted SR'); ax.set_title('Run-adjusted success rate by start region (95% stratified-bootstrap CI)')
ax.legend()
for xi, (n_, v) in enumerate(zip(t.n, t.adjSR)):
    ax.text(xi, 0.02, f'n={n_}', ha='center', fontsize=7, rotation=90)
plt.tight_layout(); plt.savefig(f'{SP}/loc_region_bars.png', dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(9, 4.5))
phases = ['fail_no_grasp', 'fail_grasped_no_transport', 'fail_reached_sink_no_place']
cols = {'fail_no_grasp': '#d65f5f', 'fail_grasped_no_transport': '#ee854a', 'fail_reached_sink_no_place': '#956cb4'}
bot = np.zeros(9)
mixp2 = mixp.reindex(order)
for ph in phases:
    if ph in mixp2:
        v = mixp2[ph].values
        ax.bar(np.arange(9), v, bottom=bot, label=ph, color=cols[ph])
        bot += v
ax.set_xticks(np.arange(9)); ax.set_xticklabels(order, rotation=45, ha='right')
ax.set_ylabel('share of failures'); ax.set_title('Failure-phase mix by start region')
ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f'{SP}/loc_phase_mix.png', dpi=130); plt.close()
print('\nplots saved')
