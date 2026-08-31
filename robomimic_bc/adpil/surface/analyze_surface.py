#!/usr/bin/env python
"""ADPIL A1.3 frozen analysis — written BEFORE unblinding (2026-08-31), per
PREREG_SURFACE.md. Extracts exactly the five registered quantities from
surface/results.json (E_test h500). Anything else printed is labeled exploratory.
"""
import json
import numpy as np

P = '/data/xinyua11/robomimic_runs/adpil/surface'
ZONES = ['xhi_yhi', 'xhi_ylo', 'xlo_yhi', 'xlo_ylo']
ADD_N = 24

res = json.load(open(f'{P}/results.json'))
def cell(mask):
    """per-seed E_test h500 J_region dicts (pp) for one mask"""
    rows = [r for r in res.values() if r['mask'] == mask]
    return [{z: 100 * r['test_h500']['J_region'][z] for z in ZONES} for r in rows]
def dep(mask):
    return np.array([100 * r['test_h500']['J_deploy'] for r in res.values() if r['mask'] == mask])

print(f'runs loaded: {len(res)} (expect 48)')

# 1-2. balanced beta_own / beta_cross
d0 = cell('balanced_D0')
d0_mean = {z: np.mean([r[z] for r in d0]) for z in ZONES}
own, cross = [], []
for z in ZONES:
    add = cell(f'balanced_add_{z}')
    own.append((np.mean([r[z] for r in add]) - d0_mean[z]) / ADD_N)
    for zo in ZONES:
        if zo != z:
            cross.append((np.mean([r[zo] for r in add]) - d0_mean[zo]) / ADD_N)
beta_own_bal, beta_cross_bal = float(np.mean(own)), float(np.mean(cross))
print(f'\n[1] beta_own(balanced)   = {beta_own_bal:+.3f} pp/demo  (zones: '
      + ', '.join(f'{z}:{s:+.2f}' for z, s in zip(ZONES, own)) + ')')
print(f'[2] beta_cross(balanced) = {beta_cross_bal:+.3f} pp/demo  '
      f'(own/other ratio {beta_own_bal / beta_cross_bal:.2f})' if beta_cross_bal != 0 else
      f'[2] beta_cross(balanced) = {beta_cross_bal:+.3f} pp/demo')

# 3. starved slopes + concavity ratio
st_own, st_cross = [], []
for z, zc in (('xhi_ylo', 'xlo_ylo'), ('xlo_yhi', 'xhi_yhi')):
    sd0 = cell(f'starved_{z}_D0')
    sd0_z = np.mean([r[z] for r in sd0])
    a_own = cell(f'starved_{z}_add_{z}')
    a_cr = cell(f'starved_{z}_add_{zc}')
    st_own.append((np.mean([r[z] for r in a_own]) - sd0_z) / ADD_N)
    st_cross.append((np.mean([r[z] for r in a_cr]) - sd0_z) / ADD_N)
    print(f'    starved {z}: D0[{z}]={sd0_z:.1f}  own-add {st_own[-1]:+.3f}  '
          f'cross-add({zc}) {st_cross[-1]:+.3f}')
beta_own_st = float(np.mean(st_own))
print(f'[3] beta_own(starved)    = {beta_own_st:+.3f} pp/demo   '
      f'CONCAVITY RATIO starved/balanced = {beta_own_st / beta_own_bal:.2f}')

# 4. sigma_seed pooled over all 16 cells (J_deploy)
masks = sorted({r['mask'] for r in res.values()})
ss, df = 0.0, 0
for m in masks:
    js = dep(m)
    ss += ((js - js.mean()) ** 2).sum(); df += len(js) - 1
sigma_seed = float(np.sqrt(ss / df))
print(f'[4] sigma_seed(RNN,v2)   = {sigma_seed:.2f} pp  ({df} df, {len(masks)} cells)')

# 5. sigma_draw from vd_N80_d0..4 (nested ANOVA, 4 df on draws)
draws = [dep(f'vd_N80_d{i}') for i in range(5)]
k = np.mean([len(d) for d in draws])
mns = np.array([d.mean() for d in draws])
ms_between = k * mns.var(ddof=1)
ms_within = np.mean([d.var(ddof=1) for d in draws])
sigma_draw2 = max(0.0, (ms_between - ms_within) / k)
print(f'[5] sigma_draw(RNN,v2,N80) = {np.sqrt(sigma_draw2):.2f} pp  '
      f'(4 df on draws — reported with caveat, does not replace MLP 3.6)')
print(f'    vd_N80 draw means: ' + ', '.join(f'{m:.1f}' for m in mns)
      + f'   grand mean {mns.mean():.1f}')

out = dict(beta_own_balanced=round(beta_own_bal, 3), beta_cross_balanced=round(beta_cross_bal, 3),
           beta_own_starved=round(beta_own_st, 3),
           concavity_ratio=round(beta_own_st / beta_own_bal, 3) if beta_own_bal else None,
           sigma_seed_rnn_v2=round(sigma_seed, 2), df_seed=df,
           sigma_draw_rnn_v2_n80=round(float(np.sqrt(sigma_draw2)), 2),
           n_runs=len(res))
json.dump(out, open(f'{P}/surface_summary.json', 'w'), indent=1)
print(f'\nwrote {P}/surface_summary.json')
