"""Angle: start-location. Part 2: run-demeaned SR vs x_rel/y_rel curves + 2D heatmap.
Part 3: does absolute position add beyond rel position (within-layout)?"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)
SP = '/data/xinyua11/tmp/factor_analysis_scratch'
df = pd.read_csv(f'{SP}/pooled_episodes.csv')
df['success'] = df['success'].astype(int)
df['run_mean'] = df.groupby('run')['success'].transform('mean')
df['dm'] = df['success'] - df['run_mean']
grand = df.success.mean()
print(f'all rows {len(df)}, {df.run.nunique()} runs, grand {grand:.3f}')

# ---- binned curves ----
def curve(v, nb=8):
    q = pd.qcut(v, nb, duplicates='drop')
    g = df.groupby(q, observed=True)['dm'].agg(['mean', 'std', 'size'])
    g['se'] = g['std'] / np.sqrt(g['size'])
    g['mid'] = [iv.mid for iv in g.index]
    return g

cx = curve(df.obj_x_rel); cy = curve(df.obj_y_rel)
cax = curve(df.obj_x_rel.abs()); cay = curve(df.obj_y_rel.abs())
print('\nadjSR by x_rel octile:\n', (grand + cx[['mean']].T).round(3))
print('adjSR by y_rel octile:\n', (grand + cy[['mean']].T).round(3))
print('adjSR by |x_rel| octile:\n', (grand + cax[['mean']].T).round(3))
print('adjSR by |y_rel| octile:\n', (grand + cay[['mean']].T).round(3))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, c, lab in [(axes[0], cx, 'obj_x_rel (left - right)'), (axes[1], cy, 'obj_y_rel (near - far)')]:
    ax.errorbar(c['mid'], grand + c['mean'], yerr=1.96 * c['se'], fmt='o-', capsize=3)
    ax.axhline(grand, ls='--', c='k', lw=1)
    ax.set_xlabel(lab); ax.set_ylabel('run-adjusted SR'); ax.grid(alpha=0.3)
axes[0].set_title('Adjusted SR vs start x (octiles, 95% CI)')
axes[1].set_title('Adjusted SR vs start y (octiles, 95% CI)')
plt.tight_layout(); plt.savefig(f'{SP}/loc_binned_curves.png', dpi=130); plt.close()

# ---- 2D heatmap 5x5 ----
nb = 5
xe = np.quantile(df.obj_x_rel, np.linspace(0, 1, nb + 1))
ye = np.quantile(df.obj_y_rel, np.linspace(0, 1, nb + 1))
xi = np.clip(np.searchsorted(xe, df.obj_x_rel, 'right') - 1, 0, nb - 1)
yi = np.clip(np.searchsorted(ye, df.obj_y_rel, 'right') - 1, 0, nb - 1)
H = np.full((nb, nb), np.nan); N = np.zeros((nb, nb), int)
for i in range(nb):
    for j in range(nb):
        m = (xi == j) & (yi == i)
        N[i, j] = m.sum()
        if m.sum() > 10:
            H[i, j] = grand + df.dm[m].mean()
fig, ax = plt.subplots(figsize=(7, 5.6))
im = ax.imshow(H, origin='lower', cmap='RdYlGn', vmin=np.nanmin(H), vmax=np.nanmax(H), aspect='auto')
for i in range(nb):
    for j in range(nb):
        if not np.isnan(H[i, j]):
            ax.text(j, i, f'{H[i,j]:.2f}\nn={N[i,j]}', ha='center', va='center', fontsize=8)
ax.set_xticks(range(nb)); ax.set_xticklabels([f'{(xe[k]+xe[k+1])/2:.2f}' for k in range(nb)])
ax.set_yticks(range(nb)); ax.set_yticklabels([f'{(ye[k]+ye[k+1])/2:.2f}' for k in range(nb)])
ax.set_xlabel('obj_x_rel (left to right)'); ax.set_ylabel('obj_y_rel (near to far)')
ax.set_title('Run-adjusted SR over start position (quintile grid)')
plt.colorbar(im, label='adjusted SR')
plt.tight_layout(); plt.savefig(f'{SP}/loc_heatmap_xy.png', dpi=130); plt.close()

# ---- R2 helpers (LPM on demeaned success) ----
def r2(X, y):
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return 1 - resid.var() / y.var()

y = df.dm.values
x, yy = df.obj_x_rel.values, df.obj_y_rel.values
quad_rel = np.column_stack([x, yy, x**2, yy**2, x * yy])
abs_feats = np.column_stack([df.obj_x_abs, df.obj_y_abs])
quad_abs = np.column_stack([df.obj_x_abs, df.obj_y_abs, df.obj_x_abs**2, df.obj_y_abs**2,
                            df.obj_x_abs * df.obj_y_abs])
r2_rel = r2(quad_rel, y)
r2_rel_absxy = r2(np.column_stack([quad_rel, abs_feats]), y)
r2_rel_quadabs = r2(np.column_stack([quad_rel, quad_abs]), y)
r2_absonly = r2(quad_abs, y)
print(f'\nR2 dm~quad(rel) = {r2_rel:.4f}')
print(f'R2 + linear abs = {r2_rel_absxy:.4f} (delta {r2_rel_absxy-r2_rel:.4f})')
print(f'R2 + quad abs   = {r2_rel_quadabs:.4f} (delta {r2_rel_quadabs-r2_rel:.4f})')
print(f'R2 quad(abs) only = {r2_absonly:.4f}')

# layout FE benchmark: how much does layout identity itself explain
lay = pd.get_dummies(df.layout_id).values.astype(float)
r2_lay = r2(lay, y)
r2_lay_rel = r2(np.column_stack([lay, quad_rel]), y)
print(f'R2 layout FE only ({lay.shape[1]} dummies) = {r2_lay:.4f}; layout FE + quad(rel) = {r2_lay_rel:.4f}')

# within-layout collinearity abs vs rel
cols_r2 = []
for L, g in df.groupby('layout_id'):
    if len(g) >= 30:
        rx = np.corrcoef(g.obj_x_abs, g.obj_x_rel)[0, 1] ** 2
        ry = np.corrcoef(g.obj_y_abs, g.obj_y_rel)[0, 1] ** 2
        cols_r2.append((L, rx, ry, len(g)))
cr = pd.DataFrame(cols_r2, columns=['layout', 'R2_x', 'R2_y', 'n'])
print('\nwithin-layout R2(abs~rel): x median %.3f (min %.3f), y median %.3f (min %.3f), layouts=%d' % (
    cr.R2_x.median(), cr.R2_x.min(), cr.R2_y.median(), cr.R2_y.min(), len(cr)))

# permutation for delta R2 of abs beyond rel: abs extra info is layout-level (counter placement).
# Null: shuffle layout centroid assignment across layouts, rebuild abs = centroid + (abs - own centroid)?
# Simpler exact framing: abs approx= layout centroid + scale*rel. Test whether layout centroid predicts
# layout-level residual SR. Residualize dm on quad(rel), average by layout, correlate with centroid.
Xq = np.column_stack([np.ones(len(y)), quad_rel])
beta, *_ = np.linalg.lstsq(Xq, y, rcond=None)
resid = y - Xq @ beta
lr = pd.DataFrame({'layout': df.layout_id, 'resid': resid,
                   'xa': df.obj_x_abs, 'ya': df.obj_y_abs})
lt = lr.groupby('layout').agg(m=('resid', 'mean'), xc=('xa', 'mean'), yc=('ya', 'mean'), n=('resid', 'size'))
lt = lt[lt.n >= 30]
w = lt.n.values
def wcorr(a, b, w):
    aw = a - np.average(a, weights=w); bw = b - np.average(b, weights=w)
    return np.average(aw * bw, weights=w) / np.sqrt(np.average(aw**2, weights=w) * np.average(bw**2, weights=w))
rx_c = wcorr(lt.m.values, lt.xc.values, w); ry_c = wcorr(lt.m.values, lt.yc.values, w)
NPm = 10000
pxs, pys = [], []
mv = lt.m.values
for p in range(NPm):
    per = rng.permutation(len(lt))
    pxs.append(wcorr(mv, lt.xc.values[per], w))
    pys.append(wcorr(mv, lt.yc.values[per], w))
px = (np.sum(np.abs(pxs) >= abs(rx_c)) + 1) / (NPm + 1)
py = (np.sum(np.abs(pys) >= abs(ry_c)) + 1) / (NPm + 1)
print(f'\nlayout-level: corr(residual SR, counter centroid x_abs) = {rx_c:+.3f} perm p={px:.4f} (layouts={len(lt)})')
print(f'layout-level: corr(residual SR, counter centroid y_abs) = {ry_c:+.3f} perm p={py:.4f}')

# significance of |x| effect in continuous form (all 28 runs): dm ~ |x_rel| slope, stratified perm
def slope(v, y):
    v = v - v.mean()
    return (v * y).sum() / (v**2).sum()
sx = slope(np.abs(x), y); sy = slope(np.abs(yy), y)
runs_idx = [np.where(df.run.values == r)[0] for r in df.run.unique()]
NP = 10000
ps_x = np.zeros(NP); ps_y = np.zeros(NP)
ax_, ay_ = np.abs(x), np.abs(yy)
for p in range(NP):
    vx = ax_.copy(); vy = ay_.copy()
    for ix in runs_idx:
        pr = rng.permutation(len(ix))
        vx[ix] = vx[ix[pr]]; vy[ix] = vy[ix[pr]]
    ps_x[p] = slope(vx, y); ps_y[p] = slope(vy, y)
pvx = (np.sum(np.abs(ps_x) >= abs(sx)) + 1) / (NP + 1)
pvy = (np.sum(np.abs(ps_y) >= abs(sy)) + 1) / (NP + 1)
print(f'\ndm ~ |x_rel| slope = {sx:+.4f} (SR per unit |x|), stratified perm p = {pvx:.4f}')
print(f'dm ~ |y_rel| slope = {sy:+.4f}, stratified perm p = {pvy:.4f}')
# effect size: predicted SR gap center (|x|=0.05) vs edge (|x|=0.7)
print(f'implied SR gap |x| 0.05 -> 0.7: {sx*0.65:+.4f}')
print('plots saved')
