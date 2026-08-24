"""Angle: episode-level repeatability. Variance split of run-demeaned success into
factors (LORO model) / config-idiosyncratic (deterministic-given-config, targetable) /
stochastic. Plus bootstrap CIs for the 2x2 interaction contrasts, and object_split check."""
import pandas as pd, numpy as np

SP = '/data/xinyua11/tmp/factor_analysis_scratch'
rng = np.random.default_rng(0)
df = pd.read_csv(f'{SP}/pooled_episodes.csv')
df['y_dm'] = df['success'] - df.groupby('run')['success'].transform('mean')
df['tall'] = (df['obj_height'] > 0.15).astype(float)
df['rimr'] = df[['obj_x_rel', 'obj_y_rel']].abs().max(axis=1)
df['rim'] = (df['rimr'] > 0.65).astype(float)
df['sig'] = (df['object_category'].astype(str) + '|' + df['layout_id'].astype(str) + '|'
             + df['style_id'].astype(str) + '|' + df['obj_x_rel'].astype(str) + '|'
             + df['obj_y_rel'].astype(str) + '|' + df['obj_height'].astype(str))
runs = sorted(df['run'].unique())

# ---------- bootstrap CIs for 2x2 interaction contrasts (run-stratified bootstrap) ----------
halfA = set(runs[0::2])
df['half'] = np.where(df['run'].isin(halfA), 'A', 'B')
def hard_styles(sub, n_bottom=15, min_n=30):
    g = sub.groupby('style_id')['y_dm'].agg(['mean', 'size'])
    g = g[g['size'] >= min_n]
    return set(g.sort_values('mean').head(n_bottom).index)
hsA = hard_styles(df[df.half == 'A']); hsB = hard_styles(df[df.half == 'B'])
df['hardstyle'] = 0.0
df.loc[(df.half == 'B') & df['style_id'].isin(hsA), 'hardstyle'] = 1.0
df.loc[(df.half == 'A') & df['style_id'].isin(hsB), 'hardstyle'] = 1.0

def contrast(sub, a, b):
    m = {}
    for va in [0, 1]:
        for vb in [0, 1]:
            s = sub[(sub[a] == va) & (sub[b] == vb)]['y_dm']
            m[(va, vb)] = s.mean()
    return (m[(1, 1)] - m[(1, 0)]) - (m[(0, 1)] - m[(0, 0)])

print("=== bootstrap 95% CIs for interaction contrasts (run-stratified resample) ===")
run_groups = {r: df[df['run'] == r].index.to_numpy() for r in runs}
for a, b in [('tall', 'rim'), ('tall', 'hardstyle'), ('rim', 'hardstyle')]:
    obs = contrast(df, a, b)
    boots = []
    for _ in range(2000):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in run_groups.values()])
        boots.append(contrast(df.loc[idx], a, b))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"{a} x {b}: contrast={obs:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")

# ---------- LORO factor-model predictions (base quadratic shape+location+style+cat) ----------
dfm = df.dropna(subset=['obj_height', 'obj_width']).copy()
def scores(train, col, k=20):
    g = train.groupby(col)['y_dm'].agg(['mean', 'size'])
    return (g['mean'] * g['size'] / (g['size'] + k)).to_dict()
def build_X(sub, sscore, cscore):
    h = sub['obj_height'].to_numpy(); w = sub['obj_width'].to_numpy(); r = sub['rimr'].to_numpy()
    ss = sub['style_id'].map(sscore).fillna(0.0).to_numpy()
    cs = sub['object_category'].map(cscore).fillna(0.0).to_numpy()
    return np.column_stack([np.ones(len(sub)), h, h**2, w, w**2, h*w, r, r**2, ss, cs])
def ridge(X, y, lam=1e-3):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)

dfm['pred'] = np.nan
for held in runs:
    tr = dfm[dfm['run'] != held]; te = dfm[dfm['run'] == held]
    if len(te) == 0: continue
    ss = scores(tr, 'style_id'); cs = scores(tr, 'object_category')
    b = ridge(build_X(tr, ss, cs), tr['y_dm'].to_numpy())
    dfm.loc[te.index, 'pred'] = build_X(te, ss, cs) @ b
dfm['resid'] = dfm['y_dm'] - dfm['pred']

# ---------- one-way random-effects ANOVA over config signatures ----------
def anova_sigma_b(sub, col):
    g = sub.groupby('sig')[col]
    n_c = g.size(); keep = n_c[n_c >= 2].index
    s = sub[sub['sig'].isin(keep)]
    g = s.groupby('sig')[col]
    n_c = g.size().to_numpy(); means = g.mean().to_numpy()
    N = n_c.sum(); C = len(n_c); grand = s[col].mean()
    ssw = ((s[col] - s.groupby('sig')[col].transform('mean')) ** 2).sum()
    msw = ssw / (N - C)
    msb = (n_c * (means - grand) ** 2).sum() / (C - 1)
    k0 = (N - (n_c ** 2).sum() / N) / (C - 1)
    s2b = (msb - msw) / k0
    return s2b, msw, N, C, s

def report_split(sub, label):
    tot = sub['y_dm'].var(ddof=0)
    fac = tot - sub['resid'].var(ddof=0)  # variance explained by LORO pred
    s2b_raw, msw_raw, N, C, srep = anova_sigma_b(sub, 'y_dm')
    s2b_res, msw_res, _, _, _ = anova_sigma_b(sub, 'resid')
    stoch = tot - fac - s2b_res
    print(f"\n--- {label}: N={len(sub)}, repeated-config episodes N={N}, configs C={C} ---")
    print(f"var(y_dm) total          = {tot:.4f}")
    print(f"factors (LORO model)     = {fac:.4f}  ({100*fac/tot:.1f}%)")
    print(f"config-idiosyncratic     = {s2b_res:.4f}  ({100*s2b_res/tot:.1f}%)   [sigma2_B of residuals]")
    print(f"stochastic (remainder)   = {stoch:.4f}  ({100*stoch/tot:.1f}%)")
    print(f"raw sigma2_B(y_dm)={s2b_raw:.4f} ({100*s2b_raw/tot:.1f}% of var) vs factors+idio={fac+s2b_res:.4f} (consistency check)")
    print(f"ICC(resid)={s2b_res/(s2b_res+msw_res):.4f}  ICC(y_dm)={s2b_raw/(s2b_raw+msw_raw):.4f}")
    print(f"targetable share of UNEXPLAINED variance = {100*s2b_res/(tot-fac):.1f}%")
    print(f"deterministic-given-config ceiling R2 = {(fac+s2b_res)/tot:.3f} (model achieves {fac/tot:.3f})")
    return srep, s2b_res, msw_res

srep, s2b_res, msw_res = report_split(dfm, 'ALL runs (27 pi0-family + 1 groot)')
report_split(dfm[dfm['policy'] == 'pi0'], 'pi0-family runs only')

# ---------- permutation null for sigma2_B of residuals (shuffle resid within run) ----------
sub = dfm.reset_index(drop=True)
obs_s2b = s2b_res
perm_s2b = []
run_pos = {r: np.where(sub['run'].to_numpy() == r)[0] for r in runs}
base_resid = sub['resid'].to_numpy()
tmp = sub.copy()
for _ in range(500):
    p = base_resid.copy()
    for r, rows in run_pos.items():
        if len(rows) > 1:
            p[rows] = p[rng.permutation(rows)]
    tmp['resid'] = p
    s2b_p, _, _, _, _ = anova_sigma_b(tmp, 'resid')
    perm_s2b.append(s2b_p)
perm_s2b = np.array(perm_s2b)
print(f"\npermutation null for sigma2_B(resid): obs={obs_s2b:.4f}, null mean={perm_s2b.mean():.4f}, "
      f"null 97.5pct={np.percentile(perm_s2b,97.5):.4f}, p={(np.sum(perm_s2b>=obs_s2b)+1)/(len(perm_s2b)+1):.4f}")

# ---------- split-half config correlation (residuals), bootstrap CI ----------
rep = srep  # episodes of configs with >=2 appearances (all runs)
def split_half_corr(rep, col, n_draw=100, seed=2):
    r = np.random.default_rng(seed)
    groups = rep.groupby('sig')[col].apply(list)
    cors = []
    for _ in range(n_draw):
        a, b = [], []
        for vals in groups:
            pick = r.choice(len(vals), size=2, replace=False)
            a.append(vals[pick[0]]); b.append(vals[pick[1]])
        cors.append(np.corrcoef(a, b)[0, 1])
    return np.mean(cors)

sh_res = split_half_corr(rep, 'resid')
sh_raw = split_half_corr(rep, 'y_dm')
# bootstrap over configs
groups = rep.groupby('sig')['resid'].apply(np.array)
def one_corr(gr, r):
    a = np.empty(len(gr)); b = np.empty(len(gr))
    for i, vals in enumerate(gr):
        p = r.choice(len(vals), 2, replace=False)
        a[i] = vals[p[0]]; b[i] = vals[p[1]]
    return a, b
bo = []
garr = list(groups)
for _ in range(300):
    r2 = np.random.default_rng(rng.integers(1e9))
    a, b = one_corr(garr, r2)
    idx = r2.choice(len(a), size=len(a), replace=True)
    bo.append(np.corrcoef(a[idx], b[idx])[0, 1])
print(f"\nsplit-half same-config correlation ACROSS runs:")
print(f"  residuals (beyond factor model): r={sh_res:.4f}  bootstrap 95% CI [{np.percentile(bo,2.5):.4f}, {np.percentile(bo,97.5):.4f}]")
print(f"  raw y_dm: r={sh_raw:.4f}")

# ---------- object_split ----------
print("\n=== object_split ===")
print(df.groupby('run')['object_split'].nunique().value_counts().to_dict(),
      "-> unique values per run (1 everywhere = no within-run variation)")
print(df['object_split'].value_counts(dropna=False).to_dict())
