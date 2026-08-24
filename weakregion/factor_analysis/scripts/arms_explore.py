import pandas as pd, numpy as np

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
df['arm'] = df['run'].map(lambda r: fam.get(r,(None,None))[1])
df['targeted'] = df['object_category'].isin(TARGETED10)
df['tall'] = df['obj_height'] > 0.21
df['rim'] = np.maximum(df['obj_x_rel'].abs(), df['obj_y_rel'].abs()) > 0.65
df['cfg'] = (df['object_category'].astype(str)+'|'+df['layout_id'].astype(str)+'|'+df['style_id'].astype(str)
             +'|'+df['obj_x_abs'].round(4).astype(str)+'|'+df['obj_y_abs'].round(4).astype(str))

for f in ['E','BALCAT','STRAT','SP']:
    sub = df[df['family']==f]
    print('=== family', f)
    g = sub.groupby('arm').agg(n=('success','size'), sr=('success','mean'),
                               tall=('tall','mean'), rim=('rim','mean'), targ=('targeted','mean'))
    print(g.round(3).to_string())
    # pairing: how many configs shared across all arms in family
    piv = sub.groupby(['cfg','arm']).size().unstack(fill_value=0)
    arms = piv.columns.tolist()
    shared_all = (piv>0).all(axis=1).sum()
    print('configs:', len(piv), 'shared by ALL arms:', shared_all)
    # pairwise sharing with baseline
    if 'baseline' in arms:
        for a in arms:
            if a=='baseline': continue
            both = ((piv[a]>0)&(piv['baseline']>0)).sum()
            print(f'  shared baseline~{a}: {both}')

# cross-family sharing (e.g., STRAT vs SP)
print()
print('episode field as config? check if same episode idx = same cfg within family E')
e = df[df['family']=='E']
chk = e.groupby('episode')['cfg'].nunique()
print('family E: episodes with >1 distinct cfg across arms:', (chk>1).sum(), 'of', len(chk))
s = df[df['family']=='STRAT']
chk2 = s.groupby('episode')['cfg'].nunique()
print('STRAT: episodes with >1 distinct cfg:', (chk2>1).sum(), 'of', len(chk2))
