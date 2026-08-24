import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
}
df['family'] = df['run'].map(lambda r: fam.get(r,(None,None))[0])
df['arm']    = df['run'].map(lambda r: fam.get(r,(None,None))[1])
df['tall'] = df['obj_height'] > 0.21
df['cfg']  = (df['object_category'].astype(str)+'|'+df['layout_id'].astype(str)+'|'+df['style_id'].astype(str)
              +'|'+df['obj_x_abs'].round(4).astype(str)+'|'+df['obj_y_abs'].round(4).astype(str))
S = df[df['family']=='STRAT']

print("=== STRAT overall arm effects (targeted-region SR), two-prop z + paired-config permutation")
def z2(pa,na,pb,nb):
    se=np.sqrt(pa*(1-pa)/na+pb*(1-pb)/nb); return (pa-pb)/se
def paired_overall(a,b,nperm=20000):
    A=S[S['arm']==a].groupby('cfg')['success'].mean(); B=S[S['arm']==b].groupby('cfg')['success'].mean()
    j=pd.concat([A,B],axis=1,keys=['a','b']).dropna(); d=(j['a']-j['b']).values
    T=d.mean(); rng=np.random.default_rng(5); cnt=0
    for _ in range(nperm):
        s=rng.choice([-1,1],len(d))
        if abs((d*s).mean())>=abs(T)-1e-12: cnt+=1
    return len(d), T, (cnt+1)/(nperm+1)
for a,b in [('core','baseline'),('random','baseline'),('core','random'),
            ('failretr','baseline'),('saturate','baseline'),('whiten','baseline')]:
    A=S[S['arm']==a]; B=S[S['arm']==b]
    pa,na=A['success'].mean(),len(A); pb,nb=B['success'].mean(),len(B)
    npair,T,p = paired_overall(a,b)
    print(f"  {a:8s}-{b:8s}: {pa:.3f} vs {pb:.3f} diff={100*(pa-pb):+.1f}pp z={z2(pa,na,pb,nb):+.2f} | paired n={npair} d={T*100:+.1f}pp p={p:.4f}")

print("\n=== Tall penalty per arm in STRAT: SR(>21cm) - SR(4-14cm) [did anyone FLATTEN the cliff?]")
for a,g in S.groupby('arm'):
    tall=g[g['obj_height']>0.21]['success']; mid=g[(g['obj_height']>=0.04)&(g['obj_height']<=0.14)]['success']
    pen=tall.mean()-mid.mean()
    se=np.sqrt(tall.var()/len(tall)+mid.var()/len(mid))
    print(f"  {a:9s}: tall {tall.mean():.2f}(n={len(tall)}) mid {mid.mean():.2f}(n={len(mid)}) penalty={pen*100:+.0f}pp +-{se*196:.0f}")

# ---- figure
fig,axes=plt.subplots(1,3,figsize=(15,4.5))
bins=[0,0.04,0.08,0.14,0.21,1.0]; labels=['<4','4-8','8-14','14-21','>21']
Sx=S.copy(); Sx['hb']=pd.cut(Sx['obj_height'],bins,labels=labels)
order=['baseline','random','core','failretr','saturate','whiten']
colors={'baseline':'k','random':'tab:blue','core':'tab:red','failretr':'tab:green','saturate':'tab:orange','whiten':'tab:purple'}
ax=axes[0]
for a in order:
    g=Sx[Sx['arm']==a].groupby('hb',observed=True)['success'].mean()
    ax.plot(range(len(g)), g.values, marker='o', label=a, color=colors[a], lw=2 if a in('baseline','core','random') else 1, alpha=1 if a in('baseline','core','random') else 0.5)
ax.set_xticks(range(5)); ax.set_xticklabels(labels); ax.set_xlabel('object height (cm)'); ax.set_ylabel('success rate')
ax.set_title('STRAT (targeted-10 eval): SR vs height by arm\ntall cliff persists in every arm'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax=axes[1]
cells=['tall\n(h>21cm)','rim\n(>0.65)','hard\nstyle','all']
b_=S[S['arm']=='baseline']
Sx['rim']=np.maximum(Sx['obj_x_rel'].abs(),Sx['obj_y_rel'].abs())>0.65
base_runs = ['pi0_student_n500','pi0_PickPlaceCounterToSink','pi0_PickPlaceCounterToSink_n150','eval_bal_A','eval_bal_B','failstates_eval']
bb=df[df['run'].isin(base_runs)].copy(); bb['ydm']=bb['success']-bb.groupby('run')['success'].transform('mean')
stm=bb.groupby('style_id')['ydm'].agg(['mean','size']); stm=stm[stm['size']>=15]
hard=set(stm[stm['mean']<=stm['mean'].quantile(1/3)].index)
Sx['hardstyle']=Sx['style_id'].isin(hard)
w=0.25
for i,a in enumerate(['random','core']):
    gains=[]
    for cell in ['tall','rim','hardstyle',None]:
        A=Sx[Sx['arm']==a]; B=Sx[Sx['arm']=='baseline']
        if cell: gains.append((A.loc[A[cell],'success'].mean()-B.loc[B[cell],'success'].mean())*100)
        else: gains.append((A['success'].mean()-B['success'].mean())*100)
    ax.bar(np.arange(4)+(i-0.5)*w, gains, w, label=f'{a} - baseline', color=colors[a])
ax.set_xticks(range(4)); ax.set_xticklabels(cells); ax.set_ylabel('gain vs baseline (pp)')
ax.set_title('STRAT: gain by weak cell\n(uniform lift, not weak-cell-specific)'); ax.legend(); ax.grid(alpha=0.3,axis='y')
ax=axes[2]
E=df[df['family']=='E']
arms_e=['random','core','coverage','influence','rc2']
tg=[]; ntg=[]
E_t = E['object_category'].isin(TARGETED10)
for a in arms_e:
    A=E[E['arm']==a]; B=E[E['arm']=='baseline']
    At=A['object_category'].isin(TARGETED10); Bt=B['object_category'].isin(TARGETED10)
    tg.append((A.loc[At,'success'].mean()-B.loc[Bt,'success'].mean())*100)
    ntg.append((A.loc[~At,'success'].mean()-B.loc[~Bt,'success'].mean())*100)
x=np.arange(len(arms_e))
ax.bar(x-0.2,tg,0.4,label='targeted 10 cats (n~38)',color='tab:red')
ax.bar(x+0.2,ntg,0.4,label='other ~70 cats (n~262)',color='tab:gray')
ax.set_xticks(x); ax.set_xticklabels(arms_e,rotation=20); ax.set_ylabel('gain vs baseline (pp)')
ax.set_title('Family E (general eval): gain vs baseline\ntargeted vs non-targeted categories'); ax.legend(fontsize=8); ax.grid(alpha=0.3,axis='y')
plt.tight_layout()
plt.savefig(SP+'arms_weakcell_summary.png',dpi=130)
print("\nsaved figure arms_weakcell_summary.png")
