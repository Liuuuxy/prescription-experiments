"""Robustness: style/layout spread + split-half after controlling run + object_category
+ the other factor (guards against paired-eval-config leakage of category difficulty).
Output (recorded):
  style ctrl run+cat+layout: obs_sd=0.0853 null=0.0392 p=0.0010 ; split-half r=0.668 [0.57,0.75]
  layout ctrl run+cat+style: obs_sd=0.0432 null=0.0379 p=0.0989 ; split-half r=0.160 [-0.06,0.38]
  hardest styles: 49,43,42,21,22 ; easiest: 27,36,31,30,17 ; hardest layouts: 19,41,45,50,13
"""
import numpy as np, pandas as pd
from scipy import stats
rng = np.random.default_rng(2)
SP = '/data/xinyua11/tmp/factor_analysis_scratch'
df = pd.read_csv(f'{SP}/pooled_episodes.csv'); df['success'] = df['success'].astype(int)
run_codes, _ = pd.factorize(df['run']); lay_codes, lay_lv = pd.factorize(df['layout_id'])
sty_codes, sty_lv = pd.factorize(df['style_id']); cat_codes, _ = pd.factorize(df['object_category'])
NRUN = run_codes.max() + 1; NLAY = 50; NSTY = 50
run_idx = [np.where(run_codes == r)[0] for r in range(NRUN)]
y = df['success'].to_numpy(float)

def demean_multi(vals, code_list, iters=80):
    r = vals.copy()
    for _ in range(iters):
        for c in code_list:
            K = c.max() + 1
            m = np.bincount(c, weights=r, minlength=K) / np.maximum(np.bincount(c, minlength=K), 1)
            r = r - m[c]
    return r

def lm(codes, vals, K):
    cnt = np.bincount(codes, minlength=K).astype(float)
    return np.bincount(codes, weights=vals, minlength=K) / np.maximum(cnt, 1), cnt

def spread(codes, K, vals, nperm=1000):
    m, cnt = lm(codes, vals, K); sd = m.std(ddof=1); null = []
    for _ in range(nperm):
        c = codes.copy()
        for idx in run_idx: c[idx] = c[idx][rng.permutation(len(idx))]
        null.append(lm(c, vals, K)[0].std(ddof=1))
    null = np.array(null)
    return sd, null.mean(), (np.sum(null >= sd) + 1) / (nperm + 1), m

def split_half(codes, K, vals, nsplit=200, min_n=8):
    rs = []
    for _ in range(nsplit):
        perm = rng.permutation(NRUN); h1 = np.isin(run_codes, perm[:NRUN // 2]); h2 = ~h1
        m1, c1 = lm(codes[h1], vals[h1], K); m2, c2 = lm(codes[h2], vals[h2], K)
        ok = (c1 >= min_n) & (c2 >= min_n)
        rs.append(stats.pearsonr(m1[ok], m2[ok])[0])
    return np.mean(rs), np.percentile(rs, [2.5, 97.5])

r3 = demean_multi(y, [run_codes, cat_codes, lay_codes])
print('style ctrl run+cat+layout:', spread(sty_codes, NSTY, r3)[:3],
      split_half(sty_codes, NSTY, r3))
r4 = demean_multi(y, [run_codes, cat_codes, sty_codes])
print('layout ctrl run+cat+style:', spread(lay_codes, NLAY, r4)[:3],
      split_half(lay_codes, NLAY, r4))
