#!/usr/bin/env python
"""ADPIL Stage 0: design-ceiling vs detection-floor computation (protocol amendment 14.2a).

Answers, BEFORE any evidence-producing run: can the Stage-2 oracle gate detect the
allocation opportunity that natural Can-PH actually contains, and if not, what designed
heterogeneity level would make the study informative?

All inputs are measured quantities from prior experiments (paths recorded below).
No new training. Output: stage0/ceiling_floor_result.json + printed report.

Model
-----
Per-demo response (C9, confirmed out-of-sample): adding one demo of zone z raises
zone-z sealed success by beta_own and other-zone success by beta_cross.
Deployment-weighted gain of an allocation {dn_z}:
    dJ = beta_cross * dn_tot + (beta_own - beta_cross) * sum_z pi_z * dn_z
Only the second term is allocation-controllable. Broad random matches pool shares in
expectation; the best allocator concentrates on argmax pi_z up to the pool cap.

Detection floor: paired active-vs-broad histories, Delta-AULC over budgets 20..80,
per-point noise = seed (sd_s^2/3, independent across budgets, new reporting seeds
per row) + draw (sd_d^2, correlation rho across nested budgets), two arms additive
(pairing shares pool+initial-20 but purchased sets differ; overlap ignored =>
conservative, floor is an upper bracket at rho=1 / lower at rho=0.5).
MDE = (t_{.975,n-1} + t_{.80,n-1}) * sd_pair / sqrt(n).
"""
import json, hashlib
import numpy as np
import h5py
from scipy import stats

RESULTS_CAP = '/data/xinyua11/robomimic_runs/capability/results.json'
DEPLOY_SHARES = '/data/xinyua11/robomimic_runs/prescribe_ph/deploy_shares.json'
POOL_H5 = '/data/xinyua11/robomimic_runs/prescribe_ph/can_ph_work.hdf5'

# --- measured inputs -------------------------------------------------------
BETA_OWN = 0.657      # pp/demo, C9_CONFIRMATION_RESULT.txt (clustered SE 0.116, t=5.66)
BETA_OWN_SE = 0.116
OWN_OTHER_RATIO = 1.84  # square screen: own/other response ratio (below its 2x gate)
SIGMA_DRAW = 3.6      # pp, vardecomp2 pooled sigma_draw under v2 recipe (BC-MLP; RNN unknown)
BUDGETS = np.arange(20, 81, 10)   # 20..80, 7 points
N_ACQ = 60            # purchased demos per history

# --- BC-RNN seed sd from CAP results (sealed scores, pooled over cells) ----
cap = json.load(open(RESULTS_CAP))
rnn = [v for v in cap.values() if v['config'] == 'rnn']
cells = {}
for v in rnn:
    cells.setdefault(v['mask'], []).append(v)
# find the sealed/test success key
sample = rnn[0]
score_key = None
for k in ('test', 'sealed', 'E_test'):
    if k in sample:
        score_key = k
        break
def sealed_J(v):
    d = v[score_key]
    if isinstance(d, dict):
        for kk in ('J_uniform', 'J', 'success', 'sr'):
            if kk in d:
                return 100.0 * d[kk] if d[kk] <= 1.0 else d[kk]
        raise KeyError(f'no J key in {list(d)}')
    return 100.0 * d if d <= 1.0 else d
ss_within, df_within = 0.0, 0
cell_means = {}
for m, vs in cells.items():
    js = np.array([sealed_J(v) for v in vs])
    cell_means[m] = js.mean()
    ss_within += ((js - js.mean()) ** 2).sum()
    df_within += len(js) - 1
SIGMA_SEED_RNN = float(np.sqrt(ss_within / df_within))
print(f'BC-RNN pooled sigma_seed (sealed, {df_within} df, {len(cells)} cells): '
      f'{SIGMA_SEED_RNN:.2f} pp   cell means: '
      + ', '.join(f'{m}={mu:.1f}' for m, mu in sorted(cell_means.items())))

# --- prevalence + pool composition ----------------------------------------
shares = json.load(open(DEPLOY_SHARES))['natural_reset_shares']
zones = sorted(shares)
pi = np.array([shares[z] for z in zones])
with h5py.File(POOL_H5, 'r') as f:
    pool_zone = {}
    for z in zones:
        mk = f'region_{z}'
        pool_zone[z] = len(f[f'mask/{mk}']) if f'mask/{mk}' in f else None
pool_n = np.array([pool_zone[z] for z in zones], dtype=float)
print(f'zones: {zones}\ndeploy pi: {pi}\npool counts (of 200): {pool_n}')
# candidate pool after removing initial 20 (drawn broadly, expectation):
cand_n = pool_n * (1 - 20 / 200)

# --- detection floor -------------------------------------------------------
w = np.array([0.5, 1, 1, 1, 1, 1, 0.5]); w = w / w.sum()   # trapezoid weights
def floor_pp(n_hist, n_seeds, sigma_d, rho):
    var_seed = 2 * (w ** 2).sum() * SIGMA_SEED_RNN ** 2 / n_seeds
    # draw term: equicorrelated rho across the 7 nested budget points
    R = np.full((7, 7), rho) + np.eye(7) * (1 - rho)
    var_draw = 2 * sigma_d ** 2 * float(w @ R @ w)
    sd_pair = np.sqrt(var_seed + var_draw)
    tcrit = stats.t.ppf(0.975, n_hist - 1) + stats.t.ppf(0.80, n_hist - 1)
    return sd_pair, tcrit * sd_pair / np.sqrt(n_hist)

floor_grid = {}
for n_hist in (12, 24):
    for n_seeds in (3, 5):
        for rho in (0.5, 1.0):
            sd_pair, mde = floor_pp(n_hist, n_seeds, SIGMA_DRAW, rho)
            floor_grid[f'n{n_hist}_s{n_seeds}_rho{rho}'] = dict(
                sd_pair=round(sd_pair, 2), mde_aulc=round(mde, 2))
print('\nDetection floor (Delta-AULC, pp):')
for k, v in floor_grid.items():
    print(f'  {k}: sd_pair={v["sd_pair"]}  MDE={v["mde_aulc"]}')

# --- ceiling: natural Can --------------------------------------------------
beta_cross = BETA_OWN / OWN_OTHER_RATIO
beta_diff = BETA_OWN - beta_cross
def alloc_advantage(pi_vec, dn_oracle, dn_broad):
    return beta_diff * float(pi_vec @ (dn_oracle - dn_broad))
p_pool = cand_n / cand_n.sum()
dn_broad = N_ACQ * p_pool
# best allocation: fill argmax-pi zone to its candidate cap, then next, greedy
order = np.argsort(-pi)
dn_oracle = np.zeros(4); left = N_ACQ
for i in order:
    take = min(left, cand_n[i]); dn_oracle[i] = take; left -= take
    if left <= 0: break
adv_final_allin = alloc_advantage(pi, dn_oracle, dn_broad)          # upper bound, ignores concavity penalty
adv_final_bc0 = (BETA_OWN) * float(pi @ (dn_oracle - dn_broad))     # if beta_cross were 0
# AULC advantage: linear accrual over rounds => weighted mean of partial doses
frac = np.array([(b - 20) / 60 for b in BUDGETS])
aulc_factor = float(w @ frac)
res_natural = dict(
    beta_own=BETA_OWN, beta_cross=round(beta_cross, 3), beta_diff=round(beta_diff, 3),
    dn_oracle=[round(x, 1) for x in dn_oracle], dn_broad=[round(x, 1) for x in dn_broad],
    ceiling_finalN_pp=round(adv_final_allin, 2),
    ceiling_finalN_if_beta_cross_zero_pp=round(adv_final_bc0, 2),
    ceiling_aulc_pp=round(adv_final_allin * aulc_factor, 2),
    ceiling_aulc_bc0_pp=round(adv_final_bc0 * aulc_factor, 2))
print(f'\nNatural-Can ceiling: final-N {res_natural["ceiling_finalN_pp"]} pp '
      f'(beta_cross=0 bound: {res_natural["ceiling_finalN_if_beta_cross_zero_pp"]}); '
      f'AULC {res_natural["ceiling_aulc_pp"]} pp (bound {res_natural["ceiling_aulc_bc0_pp"]})')
ref_floor = floor_grid['n12_s3_rho1.0']['mde_aulc']
n_needed = None
sd_ref = floor_grid['n12_s3_rho1.0']['sd_pair']
if res_natural['ceiling_aulc_bc0_pp'] > 0:
    tapprox = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)
    n_needed = int(np.ceil((tapprox * sd_ref / res_natural['ceiling_aulc_bc0_pp']) ** 2))
print(f'histories needed to detect even the beta_cross=0 bound at 80% power: ~{n_needed}')

# --- designed-prevalence dial ---------------------------------------------
print('\nDesigned-prevalence dial (exam mass q on the concentrated zone, rest uniform):')
dial = []
for q in (0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
    pi_q = np.full(4, (1 - q) / 3); pi_q[order[0]] = q
    dn_o = np.zeros(4); left = N_ACQ
    for i in np.argsort(-pi_q):
        take = min(left, cand_n[i]); dn_o[i] = take; left -= take
        if left <= 0: break
    adv = alloc_advantage(pi_q, dn_o, dn_broad)
    adv_aulc = adv * aulc_factor
    dial.append(dict(q=q, adv_finalN_pp=round(adv, 2), adv_aulc_pp=round(adv_aulc, 2),
                     x_floor=round(adv_aulc / ref_floor, 2)))
    print(f'  q={q:.2f}: final-N +{adv:5.2f} pp   AULC +{adv_aulc:5.2f} pp '
          f'({adv_aulc/ref_floor:4.2f}x the n12/s3/rho1 floor)')

out = dict(inputs=dict(beta_own=BETA_OWN, beta_own_se=BETA_OWN_SE,
                       own_other_ratio=OWN_OTHER_RATIO, sigma_draw_mlp_v2=SIGMA_DRAW,
                       sigma_seed_rnn=round(SIGMA_SEED_RNN, 2), df_seed=df_within,
                       pi_natural=list(map(float, pi)), pool_counts=list(map(float, pool_n)),
                       budgets=list(map(int, BUDGETS)), n_acq=N_ACQ, aulc_factor=round(aulc_factor, 3)),
           floor_grid=floor_grid, natural_can=res_natural,
           histories_needed_natural=n_needed, designed_prevalence_dial=dial)
p = '/data/xinyua11/robomimic_runs/adpil/stage0/ceiling_floor_result.json'
json.dump(out, open(p, 'w'), indent=1)
print(f'\nwrote {p}\nsha256 {hashlib.sha256(open(p,"rb").read()).hexdigest()[:16]}')
