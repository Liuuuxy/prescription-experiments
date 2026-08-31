import json, itertools
import numpy as np
from scipy import stats
P="/data/xinyua11/robomimic_runs/prescribe_ph"
PROF=["balanced","starved_xlo_yhi","starved_xhi_ylo"]
REG=["xlo_ylo","xlo_yhi","xhi_ylo","xhi_yhi"]
ADD=[f"add_{r}" for r in REG]+["add_cov","add_pfail","add_div1","add_div2"]
D={}
for p in PROF:
    for a in ADD+["D0"]:
        for s in (0,1):
            D[(p,a,s)]={k:json.load(open(f"{P}/out/{p}_{a}_s{s}/fixed_eval_{k}.json")) for k in ("test","probe")}
Ereg=np.load(f"{P}/E_test.npz")["region"]
q=json.load(open(f"{P}/deploy_shares.json"))["natural_reset_shares"]

# ---- A. eval-side vs training-side noise: probe/test correlation (independent start sets, SAME policy)
xs=[D[k]["probe"]["J_deploy"] for k in D]; ys=[D[k]["test"]["J_deploy"] for k in D]
r=np.corrcoef(xs,ys)[0,1]
print(f"A. corr(J_probe,J_test) over {len(xs)} runs = {r:.3f}   sd_test={np.std(ys,ddof=1):.4f} sd_probe={np.std(xs,ddof=1):.4f}")
# implied eval noise on test: var_test = var_policy + var_eval_test ; cov = var_policy*(scaling)
covpt=np.cov(xs,ys,ddof=1)[0,1]
print(f"   cov={covpt:.5f} -> var_policy~{covpt:.5f}, var_eval_test~{np.var(ys,ddof=1)-covpt:.5f} => eval share {100*(np.var(ys,ddof=1)-covpt)/np.var(ys,ddof=1):.1f}%")
# binomial expectation for eval noise on J_deploy with 50/region
ev=sum(q[rr]**2*0.25/50 for rr in REG); print(f"   binomial max eval var (p=.5) = {ev:.5f} (sd {np.sqrt(ev):.4f})")

# ---- B. paired start resampling of null-pair d
rng=np.random.default_rng(0)
ds=[]
for p in PROF:
    for s in (0,1):
        ds.append((np.array(D[(p,'add_div1',s)]["test"]["successes"]),np.array(D[(p,'add_div2',s)]["test"]["successes"])))
def Jd_from(succ,idx):
    sr=succ[idx]; rg=Ereg[idx]
    return sum(q[rr]*sr[rg==rr].mean() for rr in REG)
boot=[]
for b in range(4000):
    idx=np.concatenate([rng.choice(np.where(Ereg==rr)[0],50,replace=True) for rr in REG])
    for a1,a2 in ds: boot.append(Jd_from(a1,idx)-Jd_from(a2,idx))
boot=np.array(boot).reshape(4000,6)
resid=boot-boot.mean(axis=0)
print(f"B. eval-resampling sd of d (pooled) = {resid.std(ddof=1):.4f}")
obs_d=np.array([D[(p,'add_div1',s)]["test"]["J_deploy"]-D[(p,'add_div2',s)]["test"]["J_deploy"] for p in PROF for s in (0,1)])
print(f"   sd(obs d)={obs_d.std(ddof=1):.4f} var={obs_d.var(ddof=1):.5f}; eval var={resid.var(ddof=1):.5f} -> eval share {100*resid.var(ddof=1)/obs_d.var(ddof=1):.1f}%")
# episode agreement between div1/div2
ag=[np.mean(a1==a2) for a1,a2 in ds]; print(f"   div1/div2 episode agreement: {['%.3f'%a for a in ag]} mean {np.mean(ag):.3f}")

# ---- C. ANOVA arm main effect + arm x profile interaction (48 addition runs)
Y=np.array([[ [D[(p,a,s)]["test"]["J_deploy"] for s in (0,1)] for a in ADD] for p in PROF]) # 3x8x2
y=Y.reshape(-1)
cell=np.array([[[i*2+s for s in (0,1)] for a in ADD] for i in range(3)]).reshape(-1)
arm=np.array([[[j for s in (0,1)] for j in range(8)] for i in range(3)]).reshape(-1)
prof=np.array([[[i for s in (0,1)] for j in range(8)] for i in range(3)]).reshape(-1)
gm=y.mean()
ss_tot=((y-gm)**2).sum()
def ss_of(fac):
    return sum(len(y[fac==l])*(y[fac==l].mean()-gm)**2 for l in np.unique(fac))
ss_cell=ss_of(cell); ss_arm=ss_of(arm)
# interaction arm x profile
ap=arm*10+prof
ss_ap=ss_of(ap)-ss_arm-ss_of(prof)
ss_res=ss_tot-ss_cell-ss_arm
df_res=48-1-5-7
F=(ss_arm/7)/(ss_res/df_res)
print(f"C. SS tot={ss_tot:.4f} cell(profxseed)={ss_cell:.4f}({100*ss_cell/ss_tot:.0f}%) arm={ss_arm:.4f}({100*ss_arm/ss_tot:.0f}%) resid={ss_res:.4f}({100*ss_res/ss_tot:.0f}%)")
print(f"   arm main effect F(7,{df_res})={F:.3f} p={1-stats.f.cdf(F,7,df_res):.4f}")
ss_res2=ss_tot-ss_cell-ss_arm-ss_ap; df2=48-1-5-7-14
F2=(ss_ap/14)/(ss_res2/df2)
print(f"   arm x profile interaction F(14,{df2})={F2:.3f} p={1-stats.f.cdf(F2,14,df2):.4f}  (Gate-3 style)")

# ---- D. THE HIGH-POWER TEST the analyst skipped:
# within-run difference-in-differences: does add_r raise region r RELATIVE to the run's own other regions?
print("\nD. within-run DiD: (J_r - mean J_other) for add_r vs same quantity for div runs")
def did(run,rr):
    jr=run["test"]["J_region"]
    return jr[rr]-np.mean([jr[o] for o in REG if o!=rr])
rows=[]
for p in PROF:
    for s in (0,1):
        for rr in REG:
            t=did(D[(p,f"add_{rr}",s)],rr)
            b=np.mean([did(D[(p,f"add_div{k}",s)],rr) for k in (1,2)])
            b0=did(D[(p,"D0",s)],rr)
            rows.append((p,s,rr,t,b,b0,t-b,t-b0))
arr=np.array([[x[6],x[7]] for x in rows])
print(f"   n={len(rows)} paired cells")
for nm,col in (("vs div",0),("vs D0",1)):
    v=arr[:,col]
    t_,p_=stats.ttest_1samp(v,0)
    print(f"   {nm}: mean {v.mean()*100:+.2f}pp  sd {v.std(ddof=1)*100:.2f}  t={t_:.3f} p(2s)={p_:.4f} p(1s)={p_/2 if t_>0 else 1-p_/2:.4f}")
# also per-region breakdown vs div
print("   per-region (vs div):")
for rr in REG:
    v=np.array([x[6] for x in rows if x[2]==rr]); print(f"     {rr:8s} n={len(v)} mean {v.mean()*100:+6.2f}pp  vals "+" ".join(f"{z*100:+.0f}" for z in v))
# raw absolute region gain (no DiD)
raw=[]
for p in PROF:
    for s in (0,1):
        for rr in REG:
            raw.append(D[(p,f"add_{rr}",s)]["test"]["J_region"][rr]-np.mean([D[(p,f"add_div{k}",s)]["test"]["J_region"][rr] for k in (1,2)]))
raw=np.array(raw); t_,p_=stats.ttest_1samp(raw,0)
print(f"   raw J_region[r] add_r - div: mean {raw.mean()*100:+.2f}pp sd {raw.std(ddof=1)*100:.2f} t={t_:.3f} p={p_:.4f}")
