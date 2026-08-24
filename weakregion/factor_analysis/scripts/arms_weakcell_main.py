import pandas as pd, numpy as np
rng = np.random.default_rng(0)

SP='/data/xinyua11/tmp/factor_analysis_scratch/'
df = pd.read_csv(SP+'pooled_episodes.csv')

TARGETED10 = {'juice','spray','pitcher','canned_food','soap_dispenser','tupperware',
              'cheese_grater','ice_cube','cream_cheese_stick','jar'}
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

# ---- hard styles from INDEPENDENT base-policy runs (no arm runs -> no circularity)
base_runs = ['pi0_student_n500','pi0_PickPlaceCounterToSink','pi0_PickPlaceCounterToSink_n150',
             'eval_bal_A','eval_bal_B','failstates_eval']
b = df[df['run'].isin(base_runs)].copy()
b['y_dm'] = b['success'] - b.groupby('run')['success'].transform('mean')
st = b.groupby('style_id')['y_dm'].agg(['mean','size'])
st_ok = st[st['size']>=15]
thr = st_ok['mean'].quantile(1/3)
hard_styles = set(st_ok[st_ok['mean']<=thr].index)
print(f"hard styles: {len(hard_styles)} of {len(st_ok)} rated (n>=15 each), threshold demeaned SR <= {thr:.3f}")
df['hardstyle'] = df['style_id'].isin(hard_styles)

# ---------- test machinery ----------
def paired_interaction(dfA, dfB, wcol, nperm=20000, seed=1):
    """configs present in both arms; d = SR_A - SR_B per config; test mean(d|w)-mean(d|~w) by permuting w."""
    a = dfA.groupby('cfg').agg(y=('success','mean'), w=(wcol,'first'))
    c = dfB.groupby('cfg').agg(y=('success','mean'), w=(wcol,'first'))
    j = a.join(c, lsuffix='_A', rsuffix='_B', how='inner')
    if len(j)==0: return None
    # ensure consistent w (config-level attribute); use A's
    d = (j['y_A']-j['y_B']).values; w = j['w_A'].values.astype(bool)
    n1, n0 = w.sum(), (~w).sum()
    if n1<5 or n0<5: return dict(npair=len(j), n_w=int(n1), stat=np.nan, p=np.nan,
                                 d_w=np.nan, d_s=np.nan)
    T = d[w].mean()-d[~w].mean()
    r = np.random.default_rng(seed)
    cnt=0
    for _ in range(nperm):
        wp = r.permutation(w)
        if abs(d[wp].mean()-d[~wp].mean()) >= abs(T)-1e-12: cnt+=1
    return dict(npair=len(j), n_w=int(n1), d_w=d[w].mean(), d_s=d[~w].mean(), stat=T, p=(cnt+1)/(nperm+1))

def unpaired_dd_boot(dfA, dfB, wcol, nboot=10000, seed=2):
    """diff-in-diff (SR_A-SR_B in weak) - (SR_A-SR_B in strong), stratified bootstrap CI."""
    r = np.random.default_rng(seed)
    def cellsr(d):
        yw = d.loc[d[wcol],'success'].values; ys = d.loc[~d[wcol],'success'].values
        return yw, ys
    ywA, ysA = cellsr(dfA); ywB, ysB = cellsr(dfB)
    if min(len(ywA),len(ysA),len(ywB),len(ysB))<5: return None
    T = (ywA.mean()-ywB.mean()) - (ysA.mean()-ysB.mean())
    stats=[]
    for _ in range(nboot):
        t = (r.choice(ywA,len(ywA)).mean()-r.choice(ywB,len(ywB)).mean()) - \
            (r.choice(ysA,len(ysA)).mean()-r.choice(ysB,len(ysB)).mean())
        stats.append(t)
    lo,hi = np.percentile(stats,[2.5,97.5])
    return dict(nA=len(dfA), nB=len(dfB), n_wA=len(ywA), n_wB=len(ywB),
                gain_w=ywA.mean()-ywB.mean(), gain_s=ysA.mean()-ysB.mean(), dd=T, lo=lo, hi=hi)

def report(famname, armA, armB, wcol):
    A = df[(df['family']==famname)&(df['arm']==armA)]
    B = df[(df['family']==famname)&(df['arm']==armB)]
    u = unpaired_dd_boot(A,B,wcol)
    p = paired_interaction(A,B,wcol)
    if u is None: 
        print(f"  {armA} vs {armB} [{wcol}]: underpowered (cell n<5)"); return
    line = (f"  {armA}-{armB} [{wcol}]: gain_weak={u['gain_w']*100:+.1f}pp (n={u['n_wA']}/{u['n_wB']}) "
            f"gain_strong={u['gain_s']*100:+.1f}pp  DD={u['dd']*100:+.1f}pp CI[{u['lo']*100:+.1f},{u['hi']*100:+.1f}]")
    if p and not np.isnan(p.get('stat',np.nan)):
        line += f"  paired: DD={p['stat']*100:+.1f}pp p={p['p']:.4f} (npair={p['npair']}, weak={p['n_w']})"
    elif p:
        line += f"  paired: underpowered (npair={p['npair']}, weak={p['n_w']})"
    print(line)

# ---------- 0) targeted-category concentration, family E ----------
print("\n===== (0) FAMILY E (general eval): does the gain concentrate on the 10 TARGETED categories?")
for a in ['core','random','coverage','influence','rc2','rc','value']:
    report('E', a, 'baseline', 'targeted')
print("  --- core vs random head-to-head:")
report('E','core','random','targeted')

print("\n===== (0b) BALCAT family (category-balanced eval): targeted concentration")
for a in ['core','random']:
    report('BALCAT', a, 'baseline', 'targeted')
report('BALCAT','core','random','targeted')

# ---------- 1) height ----------
print("\n===== (1) HEIGHT: tall (h>0.21) interaction")
for famname, arms in [('E',['core','random','coverage','influence','rc2']),
                      ('BALCAT',['core','random']),
                      ('STRAT',['core','random','failretr','saturate','whiten'])]:
    print(f" family {famname}:")
    for a in arms:
        report(famname, a, 'baseline', 'tall')
    if famname!='E' or True:
        report(famname,'core','random','tall')

# height-bin profile per arm (STRAT + BALCAT pooled view, and E)
print("\n height-bin SR per arm:")
bins=[0,0.04,0.08,0.14,0.21,1.0]; labels=['<4cm','4-8','8-14','14-21','>21cm']
for famname in ['E','STRAT','BALCAT']:
    sub=df[df['family']==famname].copy()
    sub['hb']=pd.cut(sub['obj_height'],bins,labels=labels)
    piv=sub.pivot_table(index='arm',columns='hb',values='success',aggfunc='mean',observed=True)
    cnt=sub.pivot_table(index='arm',columns='hb',values='success',aggfunc='size',observed=True)
    print(f" -- {famname} (SR, n in parens)")
    out=piv.copy()
    for cix in piv.columns:
        out[cix]=[f"{piv.loc[i,cix]:.2f}({int(cnt.loc[i,cix])})" if pd.notna(piv.loc[i,cix]) else '-' for i in piv.index]
    print(out.to_string())

# ---------- 2) rim ----------
print("\n===== (2) RIM (max|rel|>0.65) interaction")
for famname, arms in [('E',['core','random','coverage','influence','rc2']),
                      ('BALCAT',['core','random']),
                      ('STRAT',['core','random','failretr','saturate','whiten'])]:
    print(f" family {famname}:")
    for a in arms:
        report(famname, a, 'baseline', 'rim')
    report(famname,'core','random','rim')

# ---------- 3) style ----------
print("\n===== (3) HARD-STYLE (bottom-tercile styles from independent base runs) interaction")
for famname, arms in [('E',['core','random','coverage','influence','rc2']),
                      ('BALCAT',['core','random']),
                      ('STRAT',['core','random','failretr','saturate','whiten'])]:
    print(f" family {famname}:")
    for a in arms:
        report(famname, a, 'baseline', 'hardstyle')
    report(famname,'core','random','hardstyle')
