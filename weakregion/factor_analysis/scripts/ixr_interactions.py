"""Angle: interactions. Part 1 — 2x2 interaction contrasts (run-demeaned, Freedman-Lane
run-stratified permutation) for tall x rim, tall x hard-style, rim x hard-style, with
hard-style defined on held-out half of runs (cross-fitted). Plus held-out-run R2/AUC
comparison of quadratic shape+location+style model with vs without interaction terms."""
import pandas as pd, numpy as np

rng = np.random.default_rng(0)
SP = '/data/xinyua11/tmp/factor_analysis_scratch'
df = pd.read_csv(f'{SP}/pooled_episodes.csv')
df['y_dm'] = df['success'] - df.groupby('run')['success'].transform('mean')
df['tall'] = (df['obj_height'] > 0.15).astype(float)
df['rimr'] = df[['obj_x_rel', 'obj_y_rel']].abs().max(axis=1)
df['rim'] = (df['rimr'] > 0.65).astype(float)
runs = sorted(df['run'].unique())

# ---------- deterministic half-split of runs (alternate over sorted list) ----------
halfA = set(runs[0::2]); halfB = set(runs[1::2])
df['half'] = np.where(df['run'].isin(halfA), 'A', 'B')

def hard_styles(sub, n_bottom=15, min_n=30):
    g = sub.groupby('style_id')['y_dm'].agg(['mean', 'size'])
    g = g[g['size'] >= min_n]
    return set(g.sort_values('mean').head(n_bottom).index)

hsA = hard_styles(df[df['half'] == 'A'])   # defined on A -> test on B
hsB = hard_styles(df[df['half'] == 'B'])   # defined on B -> test on A
print(f"hard styles from A (n={len(hsA)}): {sorted(hsA)}")
print(f"hard styles from B (n={len(hsB)}): {sorted(hsB)}")
print(f"overlap: {len(hsA & hsB)}/15")
# assign hard-style flag using the OPPOSITE half's definition (cross-fitted, no selection bias)
df['hardstyle'] = 0.0
df.loc[(df['half'] == 'B') & df['style_id'].isin(hsA), 'hardstyle'] = 1.0
df.loc[(df['half'] == 'A') & df['style_id'].isin(hsB), 'hardstyle'] = 1.0

# ---------- Freedman-Lane interaction test, run-stratified ----------
def ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]

def interaction_test(sub, a, b, n_perm=5000, seed=1):
    r = np.random.default_rng(seed)
    y = sub['y_dm'].to_numpy()
    A = sub[a].to_numpy(); B = sub[b].to_numpy()
    Xr = np.column_stack([np.ones_like(y), A, B])
    Xf = np.column_stack([Xr, A * B])
    beta_f = ols(Xf, y); obs = beta_f[3]
    beta_r = ols(Xr, y); fit_r = Xr @ beta_r; e = y - fit_r
    run_codes = pd.factorize(sub['run'])[0]
    order = np.argsort(run_codes, kind='stable')
    bounds = np.searchsorted(run_codes[order], np.arange(run_codes.max() + 2))
    cnt = 0
    e_sorted_idx = order
    for _ in range(n_perm):
        e_perm = e.copy()
        for i in range(len(bounds) - 1):
            idx = e_sorted_idx[bounds[i]:bounds[i + 1]]
            e_perm[idx] = e[r.permutation(idx)]
        ystar = fit_r + e_perm
        bstar = ols(Xf, ystar)[3]
        if abs(bstar) >= abs(obs):
            cnt += 1
    p = (cnt + 1) / (n_perm + 1)
    # 2x2 demeaned cell means
    cells = {}
    for va in [0, 1]:
        for vb in [0, 1]:
            m = (A == va) & (B == vb)
            cells[(va, vb)] = (m.sum(), y[m].mean() if m.sum() else np.nan)
    contrast = (cells[(1, 1)][1] - cells[(1, 0)][1]) - (cells[(0, 1)][1] - cells[(0, 0)][1])
    return obs, contrast, p, cells

print("\n=== 2x2 interaction tests (run-demeaned success) ===")
res = {}
# tall x rim: full data (no cross-fit needed, thresholds pre-registered)
for name, sub, a, b in [
    ('tall_x_rim (all runs)', df, 'tall', 'rim'),
    ('tall_x_hardstyle (cross-fitted)', df, 'tall', 'hardstyle'),
    ('rim_x_hardstyle (cross-fitted)', df, 'rim', 'hardstyle'),
]:
    obs, contrast, p, cells = interaction_test(sub, a, b)
    res[name] = (obs, contrast, p, cells)
    print(f"\n{name}: OLS interaction beta={obs:+.4f}  cell contrast={contrast:+.4f}  perm p={p:.4f}")
    for k, (n, m) in sorted(cells.items()):
        print(f"   {a}={k[0]} {b}={k[1]}: n={n:5d}  demeaned SR={m:+.4f}")
    c = cells
    add_pred = c[(0, 0)][1] + (c[(1, 0)][1] - c[(0, 0)][1]) + (c[(0, 1)][1] - c[(0, 0)][1])
    print(f"   additive prediction for (1,1): {add_pred:+.4f}  observed: {c[(1,1)][1]:+.4f}")

# hard-style main effect on test halves (sanity: does the cross-fitted flag replicate?)
hs_eff = df.groupby('hardstyle')['y_dm'].agg(['mean', 'size'])
print("\ncross-fitted hardstyle main effect (demeaned):\n", hs_eff)

# ---------- held-out-run R2 / AUC: base quadratic model vs +interactions ----------
def style_scores(train, k=20):
    g = train.groupby('style_id')['y_dm'].agg(['mean', 'size'])
    return (g['mean'] * g['size'] / (g['size'] + k)).to_dict()

def cat_scores(train, k=20):
    g = train.groupby('object_category')['y_dm'].agg(['mean', 'size'])
    return (g['mean'] * g['size'] / (g['size'] + k)).to_dict()

def build_X(sub, sscore, cscore, hardset, variant):
    h = sub['obj_height'].to_numpy(); w = sub['obj_width'].to_numpy()
    r = sub['rimr'].to_numpy()
    ss = sub['style_id'].map(sscore).fillna(0.0).to_numpy()
    cs = sub['object_category'].map(cscore).fillna(0.0).to_numpy()
    tall = sub['tall'].to_numpy(); rim = sub['rim'].to_numpy()
    hs = sub['style_id'].isin(hardset).astype(float).to_numpy()
    base = [np.ones(len(sub)), h, h**2, w, w**2, h*w, r, r**2, ss, cs]
    if variant == 'base':
        cols = base
    elif variant == 'base+mains':
        cols = base + [tall, rim, hs]
    elif variant == 'base+mains+int':
        cols = base + [tall, rim, hs, tall*rim, tall*hs, rim*hs]
    elif variant == 'base+contint':
        cols = base + [h*r, h*ss, r*ss]
    return np.column_stack(cols)

def ridge(X, y, lam=1e-3):
    XtX = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(XtX, X.T @ y)

def auc_manual(y, s):
    # Mann-Whitney AUC
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    # midranks for ties
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2
        i = j + 1
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

variants = ['base', 'base+mains', 'base+mains+int', 'base+contint']
per_run = {v: [] for v in variants}
sse = {v: 0.0 for v in variants}; sst = 0.0
dfm = df.dropna(subset=['obj_height', 'obj_width']).copy()  # 50 rows in one run lack h/w
print(f"\nmodel rows: {len(dfm)} (dropped {len(df)-len(dfm)} NaN h/w)")
for held in runs:
    tr = dfm[dfm['run'] != held]; te = dfm[dfm['run'] == held]
    ss_ = style_scores(tr); cs_ = cat_scores(tr); hset = hard_styles(tr)
    sst += ((te['y_dm'] - 0) ** 2).sum()  # predictions target y_dm; baseline 0
    for v in variants:
        Xtr = build_X(tr, ss_, cs_, hset, v); Xte = build_X(te, ss_, cs_, hset, v)
        b = ridge(Xtr, tr['y_dm'].to_numpy())
        pred = Xte @ b
        sse[v] += ((te['y_dm'].to_numpy() - pred) ** 2).sum()
        a = auc_manual(te['success'].to_numpy(), pred)
        per_run[v].append((held, len(te), a, ((te['y_dm'].to_numpy() - pred) ** 2).sum(),
                           (te['y_dm'].to_numpy() ** 2).sum()))

print("\n=== held-out-run (LORO) model comparison ===")
summary = {}
for v in variants:
    r2 = 1 - sse[v] / sst
    aucs = np.array([a for (_, n, a, _, _) in per_run[v] if not np.isnan(a)])
    ns = np.array([n for (_, n, a, _, _) in per_run[v] if not np.isnan(a)])
    auc_w = (aucs * ns).sum() / ns.sum()
    summary[v] = (r2, auc_w)
    print(f"{v:16s}: pooled held-out R2={r2:.4f}  n-weighted mean per-run AUC={auc_w:.4f}")

# paired per-run comparison base+mains vs base+mains+int (sign-flip test over runs)
d_auc = np.array([per_run['base+mains+int'][i][2] - per_run['base+mains'][i][2] for i in range(len(runs))])
d_auc = d_auc[~np.isnan(d_auc)]
nflip = 20000; cnt = 0
obs = d_auc.mean()
for _ in range(nflip):
    s = rng.choice([-1, 1], size=len(d_auc))
    if abs((d_auc * s).mean()) >= abs(obs):
        cnt += 1
print(f"\nper-run dAUC (int - mains): mean={obs:+.4f}, sign-flip p={(cnt+1)/(nflip+1):.4f}")
d_r2 = np.array([per_run['base+mains'][i][3] - per_run['base+mains+int'][i][3] for i in range(len(runs))])
obs2 = d_r2.sum() / sst; cnt = 0
for _ in range(nflip):
    s = rng.choice([-1, 1], size=len(d_r2))
    if abs((d_r2 * s).sum() / sst) >= abs(obs2):
        cnt += 1
print(f"pooled dR2 (int - mains): {obs2:+.5f}, sign-flip p={(cnt+1)/(nflip+1):.4f}")

np.save(f'{SP}/ixr_interaction_summary.npy', np.array([summary[v] for v in variants]))
