"""Layout/style effect analysis for pooled RoboCasa eval episodes.

Angles:
 1. Per-level run-demeaned SR spread vs within-run permutation null (layout, style)
 2. Effect size: SD of true effects (obs var - sampling var), best-worst gap (raw,
    null-expected, EB-shrunk) with bootstrap CIs
 3. Split-half reliability across runs (checkpoints)
 4. Phase resolution: grasp-fail vs late-fail spread per factor; conditional mixes
 5. Layout x region interaction (5 hardest vs 5 easiest layouts) + sink-distance proxy
Plus: two-way control (layout effect after removing style, and vice versa).
"""
import numpy as np, pandas as pd, json
from scipy import stats

rng = np.random.default_rng(0)
SP = '/data/xinyua11/tmp/factor_analysis_scratch'
df = pd.read_csv(f'{SP}/pooled_episodes.csv')
df['success'] = df['success'].astype(int)
n = len(df)

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

def shuffle_within_run(codes):
    out = codes.copy()
    for idx in run_idx:
        out[idx] = out[idx][rng.permutation(len(idx))]
    return out

NPERM = 2000
NBOOT = 1000

results = {}

# ---------------- Angle 1 + 2: spread vs null, effect size ----------------
def spread_analysis(codes, K, vals, name):
    m, cnt = level_means(codes, vals, K)
    sd_obs = m.std(ddof=1)
    # sampling variance of each level mean
    sq = np.bincount(codes, weights=vals**2, minlength=K) / np.maximum(cnt, 1)
    s2 = (sq - m**2) * cnt / np.maximum(cnt - 1, 1)          # within-level var
    samp_var = np.mean(s2 / cnt)                              # avg SE^2
    var_true = m.var(ddof=1) - samp_var
    sd_true = np.sqrt(max(var_true, 0))
    # permutation null
    null_sd, null_gap, null_sorted = [], [], []
    for _ in range(NPERM):
        c = shuffle_within_run(codes)
        mm, _ = level_means(c, vals, K)
        null_sd.append(mm.std(ddof=1))
        null_gap.append(mm.max() - mm.min())
        null_sorted.append(np.sort(mm))
    null_sd = np.array(null_sd); null_gap = np.array(null_gap)
    null_sorted = np.array(null_sorted)
    p_sd = (np.sum(null_sd >= sd_obs) + 1) / (NPERM + 1)
    gap_obs = m.max() - m.min()
    p_gap = (np.sum(null_gap >= gap_obs) + 1) / (NPERM + 1)
    # EB shrinkage
    shrink = max(var_true, 0) / (max(var_true, 0) + s2 / cnt)
    m_shrunk = shrink * m
    gap_shrunk = m_shrunk.max() - m_shrunk.min()
    # bootstrap (episodes within run)
    bt_sd_true, bt_gap_shrunk, bt_sd_obs = [], [], []
    for _ in range(NBOOT):
        bidx = np.concatenate([idx[rng.integers(0, len(idx), len(idx))] for idx in run_idx])
        c = codes[bidx]; v = vals[bidx]
        mm, cc = level_means(c, v, K)
        ok = cc > 1
        sqb = np.bincount(c, weights=v**2, minlength=K) / np.maximum(cc, 1)
        s2b = (sqb - mm**2) * cc / np.maximum(cc - 1, 1)
        vt = mm[ok].var(ddof=1) - np.mean(s2b[ok] / cc[ok])
        bt_sd_true.append(np.sqrt(max(vt, 0)))
        bt_sd_obs.append(mm[ok].std(ddof=1))
        sh = max(vt, 0) / (max(vt, 0) + s2b[ok] / cc[ok])
        ms = sh * mm[ok]
        bt_gap_shrunk.append(ms.max() - ms.min())
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    res = dict(
        sd_obs=float(sd_obs), null_sd_mean=float(null_sd.mean()), null_sd_ci=q(null_sd),
        p_sd=float(p_sd), sd_true=float(sd_true), sd_true_boot_ci=q(bt_sd_true),
        gap_obs=float(gap_obs), null_gap_mean=float(null_gap.mean()), null_gap_ci=q(null_gap),
        p_gap=float(p_gap), gap_shrunk=float(gap_shrunk), gap_shrunk_boot_ci=q(bt_gap_shrunk),
        mean_n_per_level=float(cnt.mean()))
    results[name] = res
    return m, cnt, null_sd, null_sorted, res

mL, cntL, nullsdL, nullsortL, resL = spread_analysis(lay_codes, NLAY, resid, 'layout_success')
mS, cntS, nullsdS, nullsortS, resS = spread_analysis(sty_codes, NSTY, resid, 'style_success')

# ---------------- Two-way control ----------------
def twoway_resid(vals, codes_other, K_other, n_iter=60):
    """Remove run and other-factor means by alternating projections."""
    r = vals.copy()
    for _ in range(n_iter):
        rm = np.bincount(run_codes, weights=r) / np.bincount(run_codes)
        r = r - rm[run_codes]
        om = np.bincount(codes_other, weights=r, minlength=K_other) / np.maximum(
            np.bincount(codes_other, minlength=K_other), 1)
        r = r - om[codes_other]
    return r

r_ctrl_sty = twoway_resid(y, sty_codes, NSTY)   # for layout effect controlling style
r_ctrl_lay = twoway_resid(y, lay_codes, NLAY)   # for style effect controlling layout
_, _, _, _, resL2 = spread_analysis(lay_codes, NLAY, r_ctrl_sty, 'layout_success_ctrl_style')
_, _, _, _, resS2 = spread_analysis(sty_codes, NSTY, r_ctrl_lay, 'style_success_ctrl_layout')

# ---------------- Angle 3: split-half reliability ----------------
def split_half(codes, K, vals, nsplit=300, min_n=8):
    obs_r, obs_rho, null_r = [], [], []
    first_split = None
    for b in range(nsplit):
        perm = rng.permutation(NRUN)
        h1 = np.isin(run_codes, perm[:NRUN // 2])
        h2 = ~h1
        m1, c1 = level_means(codes[h1], vals[h1], K)
        m2, c2 = level_means(codes[h2], vals[h2], K)
        ok = (c1 >= min_n) & (c2 >= min_n)
        if ok.sum() < 10:
            continue
        obs_r.append(stats.pearsonr(m1[ok], m2[ok])[0])
        obs_rho.append(stats.spearmanr(m1[ok], m2[ok])[0])
        if first_split is None:
            first_split = (m1[ok], m2[ok], ok.sum())
        # null: shuffle labels within run, same split
        c = shuffle_within_run(codes)
        m1n, c1n = level_means(c[h1], vals[h1], K)
        m2n, c2n = level_means(c[h2], vals[h2], K)
        okn = (c1n >= min_n) & (c2n >= min_n)
        null_r.append(stats.pearsonr(m1n[okn], m2n[okn])[0])
    obs_r = np.array(obs_r); null_r = np.array(null_r)
    r_mean = obs_r.mean()
    p = (np.sum(null_r >= r_mean) + 1) / (len(null_r) + 1)
    sb = 2 * r_mean / (1 + r_mean) if r_mean > -1 else np.nan
    return dict(r_mean=float(r_mean), r_ci=(float(np.percentile(obs_r, 2.5)),
                float(np.percentile(obs_r, 97.5))), rho_mean=float(np.mean(obs_rho)),
                null_r_mean=float(null_r.mean()), null_r_sd=float(null_r.std()),
                p=float(p), spearman_brown=float(sb)), first_split

results['layout_splithalf'], splitL = split_half(lay_codes, NLAY, resid)
results['style_splithalf'], splitS = split_half(sty_codes, NSTY, resid)

# ---------------- Angle 4: phase resolution ----------------
grasp_fail = (df['failure_phase'] == 'fail_no_grasp').to_numpy(float)
late_fail = df['failure_phase'].isin(['fail_grasped_no_transport',
                                      'fail_reached_sink_no_place']).to_numpy(float)
rm_g = np.bincount(run_codes, weights=grasp_fail) / np.bincount(run_codes)
rm_l = np.bincount(run_codes, weights=late_fail) / np.bincount(run_codes)
resid_g = grasp_fail - rm_g[run_codes]
resid_l = late_fail - rm_l[run_codes]

_, _, _, _, _ = spread_analysis(lay_codes, NLAY, resid_g, 'layout_graspfail')
_, _, _, _, _ = spread_analysis(lay_codes, NLAY, resid_l, 'layout_latefail')
_, _, _, _, _ = spread_analysis(sty_codes, NSTY, resid_g, 'style_graspfail')
_, _, _, _, _ = spread_analysis(sty_codes, NSTY, resid_l, 'style_latefail')

# conditional mixes for hardest/easiest 10 levels of each factor
def phase_mix(codes, m, name):
    order = np.argsort(m)
    hard, easy = order[:10], order[-10:]
    out = {}
    for lbl, grp in [('hard10', hard), ('easy10', easy)]:
        mask = np.isin(codes, grp)
        sub = df[mask]
        fails = sub[sub['success'] == 0]
        grasped = sub[sub['failure_phase'] != 'fail_no_grasp']
        out[lbl] = dict(
            n=int(len(sub)), sr=float(sub['success'].mean()),
            p_late_given_fail=float((fails['failure_phase'] != 'fail_no_grasp').mean()),
            grasp_rate=float((sub['failure_phase'] != 'fail_no_grasp').mean()),
            p_succ_given_grasp=float(grasped['success'].mean()), n_fail=int(len(fails)))
    # permutation p for difference in p_late_given_fail (shuffle level labels within run)
    obs_diff = out['hard10']['p_late_given_fail'] - out['easy10']['p_late_given_fail']
    fail_mask = df['success'].to_numpy() == 0
    late = late_fail
    diffs = []
    for _ in range(NPERM):
        c = shuffle_within_run(codes)
        mh = np.isin(c, hard) & fail_mask
        me = np.isin(c, easy) & fail_mask
        diffs.append(late[mh].mean() - late[me].mean())
    diffs = np.array(diffs)
    p = (np.sum(np.abs(diffs) >= abs(obs_diff)) + 1) / (NPERM + 1)
    out['late_given_fail_diff'] = float(obs_diff)
    out['late_given_fail_diff_p'] = float(p)
    results[name] = out

phase_mix(lay_codes, mL, 'layout_phase_mix')
phase_mix(sty_codes, mS, 'style_phase_mix')

# correlation between per-level grasp-fail effect and late-fail effect
mLg, _ = level_means(lay_codes, resid_g, NLAY)
mLl, _ = level_means(lay_codes, resid_l, NLAY)
mSg, _ = level_means(sty_codes, resid_g, NSTY)
mSl, _ = level_means(sty_codes, resid_l, NSTY)
results['layout_srcorr_with_grasp'] = float(stats.pearsonr(mL, -mLg)[0])
results['layout_srcorr_with_late'] = float(stats.pearsonr(mL, -mLl)[0])
results['style_srcorr_with_grasp'] = float(stats.pearsonr(mS, -mSg)[0])
results['style_srcorr_with_late'] = float(stats.pearsonr(mS, -mSl)[0])

# ---------------- Angle 5: layout x region interaction ----------------
order = np.argsort(mL)
hard5, easy5 = order[:5], order[-5:]
results['hard5_layouts'] = [str(lay_levels[i]) for i in hard5]
results['easy5_layouts'] = [str(lay_levels[i]) for i in easy5]
results['hard5_demeaned_sr'] = [float(mL[i]) for i in hard5]
results['easy5_demeaned_sr'] = [float(mL[i]) for i in easy5]

has_reg = df['region'].notna().to_numpy()
band = df['region'].str.split('-').str[0]          # near / mid / far
lat = df['region'].str.split('-').str[1]           # left / center / right

def band_sr(mask):
    out = {}
    for b in ['near', 'mid', 'far']:
        sel = mask & has_reg & (band == b).to_numpy()
        out[b] = (float(resid[sel].mean()), int(sel.sum()))
    return out

in_hard = np.isin(lay_codes, hard5)
in_easy = np.isin(lay_codes, easy5)
bh, be = band_sr(in_hard), band_sr(in_easy)
results['region_band_sr_hard5'] = bh
results['region_band_sr_easy5'] = be
obs_int = (bh['far'][0] - bh['near'][0]) - (be['far'][0] - be['near'][0])

# permutation: shuffle region labels within run x group (preserves group + run SR)
band_arr = band.to_numpy(object)
grp = np.where(in_hard, 1, np.where(in_easy, 2, 0))
cells = {}
for r in range(NRUN):
    for g in (1, 2):
        idx = np.where((run_codes == r) & (grp == g) & has_reg)[0]
        if len(idx) > 1:
            cells[(r, g)] = idx
ints = []
for _ in range(NPERM):
    bshuf = band_arr.copy()
    for idx in cells.values():
        bshuf[idx] = bshuf[idx][rng.permutation(len(idx))]
    def bs(mask, b):
        sel = mask & has_reg & (bshuf == b)
        return resid[sel].mean()
    ints.append((bs(in_hard, 'far') - bs(in_hard, 'near')) -
                (bs(in_easy, 'far') - bs(in_easy, 'near')))
ints = np.array(ints)
p_int = (np.sum(np.abs(ints) >= abs(obs_int)) + 1) / (NPERM + 1)
results['region_interaction_farnear'] = dict(obs=float(obs_int), p=float(p_int),
                                             null_sd=float(ints.std()))

# overall region main effect for context
results['region_main_effect'] = {b: (float(resid[has_reg & (band == b).to_numpy()].mean()),
                                     int((has_reg & (band == b).to_numpy()).sum()))
                                 for b in ['near', 'mid', 'far']}
results['region_lat_effect'] = {b: (float(resid[has_reg & (lat == b).to_numpy()].mean()),
                                    int((has_reg & (lat == b).to_numpy()).sum()))
                                for b in ['left', 'center', 'right']}

# geometry proxy: initial obj-sink distance ~ min_sink_dist on no-grasp failures
ng = (df['failure_phase'] == 'fail_no_grasp') & (df['max_lift'] < 0.02) & df['min_sink_dist'].notna()
proxy = df[ng].groupby('layout_id')['min_sink_dist'].median()
proxy_full = np.full(NLAY, np.nan)
for k, v in proxy.items():
    proxy_full[np.where(lay_levels == k)[0][0]] = v
okp = ~np.isnan(proxy_full)
r_geo, p_geo = stats.spearmanr(proxy_full[okp], mL[okp])
results['layout_sinkdist_proxy_vs_sr'] = dict(spearman=float(r_geo), p=float(p_geo),
                                              n_layouts=int(okp.sum()))
# does the proxy differ between hard5 and easy5?
results['sinkdist_proxy_hard5'] = float(np.nanmean(proxy_full[hard5]))
results['sinkdist_proxy_easy5'] = float(np.nanmean(proxy_full[easy5]))

# save per-level tables + split data + nulls for plotting
np.savez(f'{SP}/layoutstyle_plotdata.npz',
         mL=mL, cntL=cntL, nullsdL=nullsdL, nullsortL=nullsortL,
         mS=mS, cntS=cntS, nullsdS=nullsdS, nullsortS=nullsortS,
         splitL1=splitL[0], splitL2=splitL[1], splitS1=splitS[0], splitS2=splitS[1])
with open(f'{SP}/layoutstyle_results.json', 'w') as f:
    json.dump(results, f, indent=1, default=float)
print(json.dumps(results, indent=1, default=float))
