"""Angle: start-location. Part 5: randomization/balance checks (category vs region/layout).
Part 6: variance explained by location vs shape. Plus rim-effect size with CI."""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')

rng = np.random.default_rng(2)
SP = '/data/xinyua11/tmp/factor_analysis_scratch'
df = pd.read_csv(f'{SP}/pooled_episodes.csv')
df['success'] = df['success'].astype(int)
df['run_mean'] = df.groupby('run')['success'].transform('mean')
df['dm'] = df['success'] - df['run_mean']
df['rcheb'] = np.maximum(df.obj_x_rel.abs(), df.obj_y_rel.abs())
grand = df.success.mean()

# ============ RIM CONTRAST (effect size for the radial finding) ============
thr = 0.65
df['rim'] = (df.rcheb > thr).astype(float)
print(f'rim = r_cheb>{thr}: {int(df.rim.sum())} eps ({df.rim.mean():.1%})')
rim_eff = df.dm[df.rim == 1].mean() - df.dm[df.rim == 0].mean()
runs_idx = [np.where(df.run.values == r)[0] for r in df.run.unique()]
NP = 10000
rimv = df.rim.values; dmv = df.dm.values
perm = np.zeros(NP)
for p in range(NP):
    v = rimv.copy()
    for ix in runs_idx:
        v[ix] = v[ix[rng.permutation(len(ix))]]
    perm[p] = dmv[v == 1].mean() - dmv[v == 0].mean()
pv = (np.sum(np.abs(perm) >= abs(rim_eff)) + 1) / (NP + 1)
B = 3000
boot = np.zeros(B)
for b in range(B):
    samp = np.concatenate([ix[rng.integers(0, len(ix), len(ix))] for ix in runs_idx])
    rb, db = rimv[samp], dmv[samp]
    boot[b] = db[rb == 1].mean() - db[rb == 0].mean()
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f'rim - interior adjusted SR diff = {rim_eff:+.4f}, perm p={pv:.4f}, boot95CI [{lo:+.4f},{hi:+.4f}]')
print(f'adjusted SR rim {grand + df.dm[df.rim==1].mean():.3f} vs interior {grand + df.dm[df.rim==0].mean():.3f}')
# per-run consistency
pr = df.groupby('run').apply(lambda g: g.success[g.rim == 1].mean() - g.success[g.rim == 0].mean(),
                             include_groups=False)
print(f'per-run rim effect: mean {pr.mean():+.3f}, {int((pr<0).sum())}/{len(pr)} runs negative')

# ============ PART 5: BALANCE ============
# dedupe to unique eval configs (configs repeat across runs)
df['xr4'] = df.obj_x_abs.round(4)
uc = df.drop_duplicates(subset=['object_category', 'layout_id', 'style_id', 'xr4']).copy()
print(f'\nunique configs: {len(uc)} (from {len(df)} rows)')

def mc_chi2(a, b, nperm=3000, seed=3):
    r = np.random.default_rng(seed)
    tab = pd.crosstab(a, b)
    exp = np.outer(tab.sum(1), tab.sum(0)) / tab.values.sum()
    chi = ((tab.values - exp) ** 2 / np.where(exp > 0, exp, 1)).sum()
    n = tab.values.sum()
    V = np.sqrt(chi / (n * (min(tab.shape) - 1)))
    av = np.asarray(a); bv = np.asarray(b)
    cnt = 0
    for _ in range(nperm):
        chi_p_tab = pd.crosstab(av, r.permutation(bv))
        exp_p = np.outer(chi_p_tab.sum(1), chi_p_tab.sum(0)) / n
        chi_p = ((chi_p_tab.values - exp_p) ** 2 / np.where(exp_p > 0, exp_p, 1)).sum()
        cnt += chi_p >= chi
    return chi, V, (cnt + 1) / (nperm + 1), tab.shape

ucr = uc.dropna(subset=['region'])
chi, V, p, shp = mc_chi2(ucr.object_category, ucr.region, 2000)
print(f'category x region (unique configs, n={len(ucr)}): chi2={chi:.0f} shape={shp} CramersV={V:.3f} MC p={p:.4f}')
chi, V, p, shp = mc_chi2(uc.object_category, uc.layout_id, 500)
print(f'category x layout (n={len(uc)}): chi2={chi:.0f} shape={shp} CramersV={V:.3f} MC p={p:.4f}')
# continuous balance: does rim placement depend on category / object size?
big = uc.object_category.value_counts()
kw_groups = [g.rcheb.values for c, g in uc.groupby('object_category') if len(g) >= 20]
gm = uc.rcheb.mean()
f_obs = np.sum([len(g) * (g.mean() - gm) ** 2 for g in kw_groups]) / np.var(uc.rcheb)
catv = uc.object_category.values; rv = uc.rcheb.values
fperm = []
for _ in range(2000):
    cp = rng.permutation(catv)
    s = pd.Series(rv).groupby(cp).agg(['mean', 'size'])
    s = s[s['size'] >= 20]
    fperm.append(np.sum(s['size'] * (s['mean'] - gm) ** 2) / np.var(rv))
p_kw = (np.sum(np.array(fperm) >= f_obs) + 1) / 2001
print(f'rcheb ~ category between-group stat: obs={f_obs:.1f}, perm p={p_kw:.4f}')
uch = uc.dropna(subset=['obj_height', 'obj_width'])
print(f'corr(rcheb, obj_height)={np.corrcoef(uch.rcheb, uch.obj_height)[0,1]:+.3f}, '
      f'corr(rcheb, obj_width)={np.corrcoef(uch.rcheb, uch.obj_width)[0,1]:+.3f} '
      f'(unique configs, n={len(uch)}; {len(uc)-len(uch)} missing h/w dropped)')

# ============ PART 6: VARIANCE — location vs shape ============
dfc = df.dropna(subset=['obj_height', 'obj_width']).copy()
print(f'\nshape-complete rows: {len(dfc)} of {len(df)}')
runs_idx = [np.where(dfc.run.values == r)[0] for r in dfc.run.unique()]
y = dfc.dm.values
x, yy = dfc.obj_x_rel.values, dfc.obj_y_rel.values
h, w = dfc.obj_height.values, dfc.obj_width.values

def r2(X, yv):
    X = np.column_stack([np.ones(len(yv)), X])
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    return 1 - (yv - X @ beta).var() / yv.var()

loc = np.column_stack([x, yy, x**2, yy**2, x * yy])
shape = np.column_stack([h, w, h**2, w**2, h * w])
both = np.column_stack([loc, shape])
r2_loc, r2_shape, r2_both = r2(loc, y), r2(shape, y), r2(both, y)
print(f'\nR2 (run-demeaned success, 5df each): location(quad rel)={r2_loc:.4f}  '
      f'shape(quad h,w)={r2_shape:.4f}  both={r2_both:.4f}')
print(f'incremental: loc|shape={r2_both-r2_shape:.4f}  shape|loc={r2_both-r2_loc:.4f}')

# permutation null for each R2 (shuffle features within run, keep y) — chance level for 5df
NP2 = 500
null_loc = np.zeros(NP2); null_shape = np.zeros(NP2)
for p in range(NP2):
    lp = loc.copy(); sp_ = shape.copy()
    for ix in runs_idx:
        pr_ = rng.permutation(len(ix))
        lp[ix] = lp[ix[pr_]]; sp_[ix] = sp_[ix[pr_]]
    null_loc[p] = r2(lp, y); null_shape[p] = r2(sp_, y)
print(f'null R2 (5df, within-run shuffled): loc mean {null_loc.mean():.4f} '
      f'(95pct {np.percentile(null_loc,95):.4f}); shape mean {null_shape.mean():.4f} '
      f'(95pct {np.percentile(null_shape,95):.4f})')
print(f'p(R2_loc<=null)={np.mean(null_loc>=r2_loc):.4f}, p(R2_shape<=null)={np.mean(null_shape>=r2_shape):.4f}')

# bootstrap CI on R2s (stratified within run)
b_loc = np.zeros(1000); b_shape = np.zeros(1000)
for b in range(1000):
    samp = np.concatenate([ix[rng.integers(0, len(ix), len(ix))] for ix in runs_idx])
    b_loc[b] = r2(loc[samp], y[samp]); b_shape[b] = r2(shape[samp], y[samp])
print(f'boot95CI R2_loc [{np.percentile(b_loc,2.5):.4f},{np.percentile(b_loc,97.5):.4f}]  '
      f'R2_shape [{np.percentile(b_shape,2.5):.4f},{np.percentile(b_shape,97.5):.4f}]')
# note: same-config repeats across runs slightly inflate effective n; caveat only.
