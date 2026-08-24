import pandas as pd, numpy as np

SP='/data/xinyua11/tmp/factor_analysis_scratch/'
df = pd.read_csv(SP+'pooled_episodes.csv')
TARGETED10 = {'juice','spray','pitcher','canned_food','soap_dispenser','tupperware',
              'cheese_grater','ice_cube','cream_cheese_stick','jar'}
CORE_ALLOC = {'juice':29,'spray':28,'pitcher':24,'canned_food':21,'soap_dispenser':19,
              'tupperware':17,'cheese_grater':16,'ice_cube':16,'cream_cheese_stick':15,'jar':15}
fam = {
 'eval_baseline':('E','baseline'),'eval_core':('E','core'),'eval_random':('E','random'),
 'eval_coverage':('E','coverage'),'eval_influence':('E','influence'),'eval_value':('E','value'),
 'eval_rc':('E','rc'),'eval_rc2':('E','rc2'),
 'eval_balcat_baseline':('BALCAT','baseline'),'eval_balcat_core':('BALCAT','core'),'eval_balcat_random':('BALCAT','random'),
 'eval_strat_baseline':('STRAT','baseline'),'eval_strat_core':('STRAT','core'),'eval_strat_random':('STRAT','random'),
 'eval_strat_failretr':('STRAT','failretr'),'eval_strat_saturate':('STRAT','saturate'),'eval_strat_whiten':('STRAT','whiten'),
 'eval_strat_paired_baseline':('SP','baseline'),'eval_strat_paired_core':('SP','core'),
 'eval_strat_paired_random':('SP','random'),'eval_strat_paired_influence':('SP','influence'),
}
df['family'] = df['run'].map(lambda r: fam.get(r,(None,None))[0])
df['arm']    = df['run'].map(lambda r: fam.get(r,(None,None))[1])
df['targeted'] = df['object_category'].isin(TARGETED10)
df['tall'] = df['obj_height'] > 0.21
df['rim']  = np.maximum(df['obj_x_rel'].abs(), df['obj_y_rel'].abs()) > 0.65
df['cfg']  = (df['object_category'].astype(str)+'|'+df['layout_id'].astype(str)+'|'+df['style_id'].astype(str)
              +'|'+df['obj_x_abs'].round(4).astype(str)+'|'+df['obj_y_abs'].round(4).astype(str))

print("===== (A) DECOMPOSITION of overall gain, family E (share of gain from targeted 13% of eval)")
E = df[df['family']=='E']
def decomp(a, b):
    A = E[E['arm']==a]; B = E[E['arm']==b]
    tot = A['success'].mean()-B['success'].mean()
    pt = 0.5*(A['targeted'].mean()+B['targeted'].mean())
    dt = A.loc[A['targeted'],'success'].mean()-B.loc[B['targeted'],'success'].mean()
    dn = A.loc[~A['targeted'],'success'].mean()-B.loc[~B['targeted'],'success'].mean()
    print(f"  {a}-{b}: total={tot*100:+.1f}pp; targeted-cells contribute {pt*dt*100:+.1f}pp "
          f"({pt*100:.0f}% of eval x {dt*100:+.1f}pp), non-targeted contribute {(1-pt)*dn*100:+.1f}pp")
decomp('core','baseline'); decomp('core','random'); decomp('random','baseline')

print("\n===== (B) DOSE-RESPONSE within STRAT: core demos/category vs per-category gain (core-baseline)")
S = df[df['family']=='STRAT']
pc = S.pivot_table(index='object_category', columns='arm', values='success', aggfunc='mean')
pn = S.pivot_table(index='object_category', columns='arm', values='success', aggfunc='size')
pc = pc.loc[[c for c in CORE_ALLOC if c in pc.index]]
pc['n_demos'] = [CORE_ALLOC[c] for c in pc.index]
pc['gain_core'] = pc['core']-pc['baseline']
pc['gain_rand'] = pc['random']-pc['baseline']
print(pc[['n_demos','baseline','core','random','gain_core','gain_rand']].round(2).to_string())
for col in ['gain_core','gain_rand']:
    x = pc['n_demos'].values.astype(float); y = pc[col].values
    r = np.corrcoef(x,y)[0,1]
    # permutation p
    rng = np.random.default_rng(3); cnt=0
    for _ in range(20000):
        if abs(np.corrcoef(rng.permutation(x),y)[0,1])>=abs(r)-1e-12: cnt+=1
    print(f"  corr(n_demos, {col}) = {r:+.2f}, perm p={(cnt+1)/20001:.3f}  (10 categories)")

print("\n===== (C) POOLED powered interaction in STRAT: all 5 selection arms vs baseline, paired configs")
# for each arm vs baseline paired configs, d_i; pool; permute weak labels within arm-stratum
def pooled_paired(wcol, arms=('core','random','failretr','saturate','whiten'), nperm=20000):
    B = S[S['arm']=='baseline']
    bg = B.groupby('cfg').agg(yb=('success','mean'), w=(wcol,'first'))
    ds, ws, strat = [], [], []
    for i,a in enumerate(arms):
        A = S[S['arm']==a].groupby('cfg').agg(ya=('success','mean'))
        j = A.join(bg, how='inner')
        ds.append((j['ya']-j['yb']).values); ws.append(j['w'].values.astype(bool))
        strat.append(np.full(len(j), i))
    d = np.concatenate(ds); w = np.concatenate(ws); st = np.concatenate(strat)
    T = d[w].mean()-d[~w].mean()
    rng = np.random.default_rng(4); cnt=0
    for _ in range(nperm):
        wp = np.empty_like(w)
        for i in range(len(arms)):
            m = st==i; wp[m] = rng.permutation(w[m])
        if abs(d[wp].mean()-d[~wp].mean())>=abs(T)-1e-12: cnt+=1
    print(f"  pooled [{wcol}] arms={arms}: n_pairs={len(d)}, weak={int(w.sum())}, "
          f"gain_weak={d[w].mean()*100:+.1f}pp gain_strong={d[~w].mean()*100:+.1f}pp "
          f"DD={T*100:+.1f}pp perm p={(cnt+1)/(nperm+1):.4f}")
pooled_paired('tall'); pooled_paired('rim')
# targeted-selection arms only (exclude random)
pooled_paired('tall', arms=('core','failretr','saturate','whiten'))
pooled_paired('rim', arms=('core','failretr','saturate','whiten'))

print("\n===== (D) FAILURE-PHASE shift per arm (STRAT): where did failures go?")
ph = S.groupby(['arm','failure_phase']).size().unstack(fill_value=0)
ph = ph.div(ph.sum(axis=1), axis=0)
print((ph*100).round(1).to_string())
print("\n  no-grasp rate on TALL episodes only:")
pht = S[S['tall']].groupby(['arm','failure_phase']).size().unstack(fill_value=0)
pht = pht.div(pht.sum(axis=1), axis=0)
print((pht*100).round(1).to_string())

print("\n===== (E) TALL ceiling: best any arm achieved on h>0.21 (with Wilson CI), incl saturate=610 targeted demos")
def wilson(k,n,z=1.96):
    if n==0: return (np.nan,np.nan)
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return ((c-h)/d,(c+h)/d)
for famname in ['STRAT','BALCAT','E']:
    sub = df[(df['family']==famname)&df['tall']]
    for a,g in sub.groupby('arm'):
        k,n=g['success'].sum(),len(g); lo,hi=wilson(k,n)
        print(f"  {famname:7s} {a:9s} tall SR={k/n:.2f} ({k}/{n}) CI[{lo:.2f},{hi:.2f}]")

print("\n===== (F) progress metrics on tall failures (STRAT): max_lift and min_sink_dist among failures")
ft = S[S['tall'] & (S['success']==0)]
print(ft.groupby('arm').agg(n=('success','size'), lift=('max_lift','mean'),
                            sinkd=('min_sink_dist','mean')).round(3).to_string())

print("\n===== (G) power note: minimal detectable DD (approx, 80% power) for key contrasts")
def mdd(n1a,n0a,n1b,n0b,p=0.4):
    se = np.sqrt(p*(1-p)*(1/n1a+1/n0a+1/n1b+1/n0b))
    return 2.8*se
print(f"  family E targeted (38/262 per arm): MDD ~{mdd(38,262,38,262)*100:.0f}pp")
print(f"  STRAT tall (103/177 per arm):       MDD ~{mdd(103,177,103,177)*100:.0f}pp")
print(f"  STRAT pooled 4 targeted arms (~450/700): MDD ~{mdd(450,700,103,177)*100:.0f}pp")
