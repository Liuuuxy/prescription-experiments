"""Angle: category-signal power analysis.
(1) per-category n + binomial CI widths
(2) observed vs null (within-run label permutation) spread of per-category SR
(3) split-half reliability of per-category SR (run-split and config-split halves)
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RNG = np.random.default_rng(0)
SP = '/data/xinyua11/tmp/factor_analysis_scratch'
df = pd.read_csv(f'{SP}/pooled_episodes.csv')
df['success'] = df['success'].astype(int)
df['y'] = df['success'] - df.groupby('run')['success'].transform('mean')  # run-demeaned

cats = np.sort(df['object_category'].unique())
cat_idx = {c: i for i, c in enumerate(cats)}
ci = df['object_category'].map(cat_idx).to_numpy()
y = df['y'].to_numpy()
suc = df['success'].to_numpy()
runs = df['run'].to_numpy()
run_names = np.sort(df['run'].unique())
n_cat = len(cats)

# ---------- (1) POWER ----------
counts = np.bincount(ci, minlength=n_cat)
p_hat = np.bincount(ci, weights=suc, minlength=n_cat) / counts
pbar = suc.mean()


def wilson_halfwidth(p, n, z=1.96):
    denom = 1 + z**2 / n
    halfw = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return halfw

hw = wilson_halfwidth(np.clip(p_hat, .05, .95), counts)
print("=== (1) POWER ===")
print(f"n categories: {n_cat}; episode counts: median {np.median(counts):.0f}, "
      f"IQR [{np.percentile(counts,25):.0f},{np.percentile(counts,75):.0f}], min {counts.min()}, max {counts.max()}")
print(f"Wilson 95% CI half-width at observed p: median {np.median(hw):.3f}, "
      f"IQR [{np.percentile(hw,25):.3f},{np.percentile(hw,75):.3f}], max {hw.max():.3f}")
for thr in (0.05, 0.10, 0.15):
    print(f"  categories with CI half-width > {thr}: {(hw>thr).sum()}/{n_cat}")
# naive pooled SR spread (what the user eyeballed)
print(f"naive pooled per-category SR: SD {p_hat.std(ddof=1):.3f}, range [{p_hat.min():.2f},{p_hat.max():.2f}]")

# ---------- (2) OBSERVED vs NULL SPREAD (run-demeaned) ----------
def cat_stats(ci_arr, yv, min_n=20):
    cnt = np.bincount(ci_arr, minlength=n_cat)
    s = np.bincount(ci_arr, weights=yv, minlength=n_cat)
    m = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    mask = cnt >= min_n
    sd_unw = np.nanstd(m[mask], ddof=1)
    # weighted between-category sum of squares (chi2-like)
    T = np.nansum(cnt[mask] * m[mask] ** 2)
    # count-weighted SD
    w = cnt[mask]
    mu_w = np.nansum(w * m[mask]) / w.sum()
    sd_w = np.sqrt(np.nansum(w * (m[mask] - mu_w) ** 2) / w.sum())
    return sd_unw, sd_w, T, m, mask

MIN_N = 20
obs_sd, obs_sdw, obs_T, obs_m, obs_mask = cat_stats(ci, y, MIN_N)

B = 2000
null_sd = np.empty(B); null_sdw = np.empty(B); null_T = np.empty(B)
# pre-split indices by run for fast within-run permutation
run_groups = [np.flatnonzero(runs == r) for r in run_names]
ci_perm = ci.copy()
for b in range(B):
    for g in run_groups:
        ci_perm[g] = ci[g][RNG.permutation(len(g))]
    null_sd[b], null_sdw[b], null_T[b], _, _ = cat_stats(ci_perm, y, MIN_N)

p_sd = (1 + (null_sd >= obs_sd).sum()) / (1 + B)
p_T = (1 + (null_T >= obs_T).sum()) / (1 + B)
excess_var = obs_sd**2 - null_sd.mean()**2
sig_sd = np.sqrt(max(excess_var, 0))
print("\n=== (2) OBSERVED vs NULL SPREAD (run-demeaned SR, cats n>=%d: %d cats) ===" % (MIN_N, obs_mask.sum()))
print(f"observed SD of per-category demeaned SR: {obs_sd:.4f} (weighted {obs_sdw:.4f})")
print(f"null SD: mean {null_sd.mean():.4f}, 95% range [{np.percentile(null_sd,2.5):.4f},{np.percentile(null_sd,97.5):.4f}]")
print(f"p(perm, SD stat) = {p_sd:.4f};  p(perm, weighted-SSq stat) = {p_T:.4f}")
print(f"implied TRUE category-signal SD = sqrt(obs^2 - E[null]^2) = {sig_sd:.4f}")
print(f"variance ratio signal/(observed spread): {excess_var/max(obs_sd**2,1e-12):.2f}")

# ---------- (3) SPLIT-HALF RELIABILITY ----------
def split_half_r(assign_half, min_n=10, n_splits=200, by='run'):
    rs = []
    for s in range(n_splits):
        halfA = assign_half(s)
        mA = np.zeros(n_cat); mB = np.zeros(n_cat)
        cA = np.bincount(ci[halfA], minlength=n_cat)
        cB = np.bincount(ci[~halfA], minlength=n_cat)
        sA = np.bincount(ci[halfA], weights=y[halfA], minlength=n_cat)
        sB = np.bincount(ci[~halfA], weights=y[~halfA], minlength=n_cat)
        ok = (cA >= min_n) & (cB >= min_n)
        if ok.sum() < 10:
            continue
        a = sA[ok] / cA[ok]; b = sB[ok] / cB[ok]
        r = np.corrcoef(a, b)[0, 1]
        rs.append((r, ok.sum()))
    rs, ns = np.array([x[0] for x in rs]), np.array([x[1] for x in rs])
    return rs, ns

# run-split: random half of runs
run_arr = run_names.copy()
def run_split(seed):
    r = np.random.default_rng(1000 + seed)
    half = r.permutation(len(run_arr))[:len(run_arr)//2]
    half_set = set(run_arr[half])
    return np.isin(runs, list(half_set))

rs_run, ns_run = split_half_r(run_split)
# config-split: assign unique config signature to halves (avoids shared-config inflation)
sig = (df['object_category'].astype(str) + '|' + df['layout_id'].astype(str) + '|' +
       df['style_id'].astype(str) + '|' + df['obj_x_abs'].round(6).astype(str) + '|' +
       df['obj_y_abs'].round(6).astype(str))
usig, sig_inv = np.unique(sig.to_numpy(), return_inverse=True)
def cfg_split(seed):
    r = np.random.default_rng(2000 + seed)
    half_sig = r.random(len(usig)) < 0.5
    return half_sig[sig_inv]

rs_cfg, ns_cfg = split_half_r(cfg_split)

def sb(r):  # Spearman-Brown to full length
    return 2 * r / (1 + r)

print("\n=== (3) SPLIT-HALF RELIABILITY of per-category (demeaned) SR ===")
for name, rs, ns in [('run-split (14/14 runs)', rs_run, ns_run), ('config-split (disjoint eval configs)', rs_cfg, ns_cfg)]:
    print(f"{name}: mean r = {rs.mean():.3f}, 95% range over {len(rs)} splits "
          f"[{np.percentile(rs,2.5):.3f},{np.percentile(rs,97.5):.3f}], "
          f"median #cats used {np.median(ns):.0f}; Spearman-Brown full-data reliability = {sb(rs.mean()):.3f}")

# ---------- PLOT ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
ax.hist(null_sd, bins=40, color='0.7', label=f'null (within-run permuted), B={B}')
ax.axvline(obs_sd, color='crimson', lw=2, label=f'observed = {obs_sd:.3f}')
ax.set_xlabel('SD of per-category run-demeaned SR (cats n>=20)')
ax.set_ylabel('count')
ax.set_title(f'Category-SR spread: observed vs null\nperm p={p_sd:.3f}; implied true signal SD={sig_sd:.3f}')
ax.legend(fontsize=8)

ax = axes[1]
order = np.argsort(p_hat)
sel = order[np.isin(order, np.flatnonzero(counts >= MIN_N))]
ax.errorbar(np.arange(len(sel)), p_hat[sel], yerr=hw[sel], fmt='.', ms=3, lw=0.6, color='steelblue', alpha=0.8)
ax.axhline(pbar, color='k', ls='--', lw=1, label=f'pooled mean {pbar:.2f}')
ax.set_xlabel('categories sorted by naive SR (n>=20)')
ax.set_ylabel('naive SR with Wilson 95% CI')
ax.set_title('Per-category SR: CI half-width median %.2f' % np.median(hw[counts >= MIN_N]))
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f'{SP}/catsignal_null_spread_hist.png', dpi=140)
print('\nsaved plot catsignal_null_spread_hist.png')
