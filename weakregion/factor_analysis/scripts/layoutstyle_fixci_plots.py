"""Bias-corrected bootstrap CIs for sd_true (episode bootstrap adds ~1 extra unit of
sampling variance to level means, so subtract 2x samp var in replicates) + plots."""
import numpy as np, pandas as pd, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(1)
SP = '/data/xinyua11/tmp/factor_analysis_scratch'
df = pd.read_csv(f'{SP}/pooled_episodes.csv')
df['success'] = df['success'].astype(int)
run_codes, run_levels = pd.factorize(df['run'])
lay_codes, lay_levels = pd.factorize(df['layout_id'])
sty_codes, sty_levels = pd.factorize(df['style_id'])
NRUN, NLAY, NSTY = len(run_levels), len(lay_levels), len(sty_levels)
run_idx = [np.where(run_codes == r)[0] for r in range(NRUN)]
y = df['success'].to_numpy(float)
run_mean = np.bincount(run_codes, weights=y) / np.bincount(run_codes)
resid = y - run_mean[run_codes]

def level_means(codes, vals, K):
    cnt = np.bincount(codes, minlength=K).astype(float)
    s = np.bincount(codes, weights=vals, minlength=K)
    return s / np.maximum(cnt, 1), cnt

def sd_true_bc_ci(codes, K, nboot=1000):
    out = []
    for _ in range(nboot):
        bidx = np.concatenate([idx[rng.integers(0, len(idx), len(idx))] for idx in run_idx])
        c = codes[bidx]; v = resid[bidx]
        mm, cc = level_means(c, v, K)
        ok = cc > 1
        sq = np.bincount(c, weights=v**2, minlength=K) / np.maximum(cc, 1)
        s2 = (sq - mm**2) * cc / np.maximum(cc - 1, 1)
        vt = mm[ok].var(ddof=1) - 2 * np.mean(s2[ok] / cc[ok])   # 2x: original + resample noise
        out.append(np.sqrt(max(vt, 0)))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

ciL = sd_true_bc_ci(lay_codes, NLAY)
ciS = sd_true_bc_ci(sty_codes, NSTY)
print('layout sd_true bias-corrected boot CI:', ciL)
print('style  sd_true bias-corrected boot CI:', ciS)

res = json.load(open(f'{SP}/layoutstyle_results.json'))
res['layout_success']['sd_true_bc_ci'] = ciL
res['style_success']['sd_true_bc_ci'] = ciS
json.dump(res, open(f'{SP}/layoutstyle_results.json', 'w'), indent=1)

# ---------------- plots ----------------
d = np.load(f'{SP}/layoutstyle_plotdata.npz')
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
for i, (nm, m, nullsd, nullsort, r) in enumerate([
        ('Layout', d['mL'], d['nullsdL'], d['nullsortL'], res['layout_success']),
        ('Style', d['mS'], d['nullsdS'], d['nullsortS'], res['style_success'])]):
    a = ax[i, 0]
    a.hist(nullsd, bins=40, color='0.7', label='null (shuffle within run)')
    a.axvline(m.std(ddof=1), color='crimson', lw=2, label=f'observed SD={m.std(ddof=1):.3f}')
    a.set_title(f'{nm}: SD of per-id run-demeaned SR vs null (p={r["p_sd"]:.4f})')
    a.set_xlabel('SD across 50 ids'); a.legend(fontsize=8)
    a2 = ax[i, 1]
    srt = np.sort(m)
    lo = np.percentile(nullsort, 2.5, axis=0); hi = np.percentile(nullsort, 97.5, axis=0)
    a2.fill_between(range(len(srt)), lo, hi, color='0.8', label='null 95% band')
    a2.plot(range(len(srt)), srt, 'o-', ms=3, color='crimson', label='observed')
    a2.axhline(0, color='k', lw=0.5)
    a2.set_title(f'{nm}: sorted per-id demeaned SR (sd_true={r["sd_true"]:.3f})')
    a2.set_xlabel('rank'); a2.set_ylabel('demeaned SR'); a2.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f'{SP}/layoutstyle_sr_vs_null.png', dpi=130)
plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
for a, (nm, x1, x2, rr) in zip(ax, [
        ('Layout', d['splitL1'], d['splitL2'], res['layout_splithalf']),
        ('Style', d['splitS1'], d['splitS2'], res['style_splithalf'])]):
    a.scatter(x1, x2, s=22, alpha=0.75)
    r1 = stats.pearsonr(x1, x2)[0]
    lim = max(np.abs(np.r_[x1, x2])) * 1.15
    a.plot([-lim, lim], [-lim, lim], 'k--', lw=0.6)
    a.axhline(0, color='0.6', lw=0.5); a.axvline(0, color='0.6', lw=0.5)
    a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
    a.set_xlabel('demeaned SR, run-half A'); a.set_ylabel('demeaned SR, run-half B')
    a.set_title(f'{nm} split-half: this split r={r1:.2f}\n'
                f'mean over 300 splits r={rr["r_mean"]:.2f} (null {rr["null_r_mean"]:.2f}'
                f'+-{rr["null_r_sd"]:.2f}, p={rr["p"]:.3f})', fontsize=10)
plt.tight_layout()
plt.savefig(f'{SP}/layoutstyle_splithalf.png', dpi=130)
plt.close()
print('plots saved')
