import json, os, itertools
import numpy as np
P="/data/xinyua11/robomimic_runs/prescribe_ph"
PROF=["balanced","starved_xlo_yhi","starved_xhi_ylo"]
REG=["xlo_ylo","xlo_yhi","xhi_ylo","xhi_yhi"]
ADD=[f"add_{r}" for r in REG]+["add_cov","add_pfail","add_div1","add_div2"]
V=np.zeros((3,8,2))
for i,p in enumerate(PROF):
    for j,a in enumerate(ADD):
        for s in (0,1):
            V[i,j,s]=json.load(open(f"{P}/out/{p}_{a}_s{s}/fixed_eval_test.json"))["J_deploy"]
Xobs=json.load(open(f"{P}/MDE.json"))["X"]
TMULT=3.9806457521353384
pairs=[(i,j) for i in range(8) for j in range(8) if i!=j]   # ordered pairs = 56
assert len(pairs)==56

def stats_for(labeling):
    # labeling: tuple of 3 ordered pairs (d1,d2) index into arm axis
    dvals=np.empty(6); prof_mean=np.empty(3)
    hits=0; sadv=0.0; madv=-9; hitlist=[]
    per=[]
    for i,(a1,a2) in enumerate(labeling):
        d=V[i,a1,:]-V[i,a2,:]
        prof_mean[i]=d.mean()
    sdp=prof_mean.std(ddof=1)
    Xr=TMULT*sdp/np.sqrt(3)
    advs=np.empty((3,8,2)); advs[:]=np.nan
    for i,(a1,a2) in enumerate(labeling):
        base=0.5*(V[i,a1,:]+V[i,a2,:])
        for j in range(8):
            if j in (a1,a2): continue
            advs[i,j,:]=V[i,j,:]-base
    return advs, Xr

def count_hits(advs, X):
    m=np.nanmean(advs,axis=2)
    ok=(~np.isnan(m)) & (m>X) & (np.sign(advs[:,:,0])==np.sign(advs[:,:,1]))
    return int(ok.sum()), float(m[ok].sum()), (float(m[ok].max()) if ok.any() else -9)

# observed
obs_lab=(( ADD.index("add_div1"),ADD.index("add_div2")),)*3
advs,Xr=stats_for(obs_lab)
print("observed:",count_hits(advs,Xobs),"X recomputed",Xr)

N=0; hf=np.zeros(20); ge7_frozen=0; ge1_frozen=0
ge7_pipe=0; ge1_pipe=0; Xle=0; hits_pipe_sum=0; hits_frozen_sum=0
ge7_sum_frozen=0; ge7_max_frozen=0
cond_le_n=0; cond_le_ge7=0; cond_gt_n=0; cond_gt_ge7=0
hits_pipe_all=[]; X_all=[]
for lab in itertools.product(pairs,repeat=3):
    advs,Xr=stats_for(lab)
    hF,sF,mF=count_hits(advs,Xobs)
    hP,sP,mP=count_hits(advs,Xr)
    N+=1
    hits_frozen_sum+=hF; hits_pipe_sum+=hP
    if hF>=7: ge7_frozen+=1
    if hF>=1: ge1_frozen+=1
    if hF>=7 and sF>=1.3366375: ge7_sum_frozen+=1
    if hF>=7 and mF>=0.2485575: ge7_max_frozen+=1
    if hP>=7: ge7_pipe+=1
    if hP>=1: ge1_pipe+=1
    if Xr<=Xobs+1e-12:
        Xle+=1; cond_le_n+=1
        if hP>=7: cond_le_ge7+=1
    else:
        cond_gt_n+=1
        if hP>=7: cond_gt_ge7+=1
    X_all.append(Xr)
X_all=np.array(X_all)
print(f"N={N}")
print(f"FROZEN X: E[hits]={hits_frozen_sum/N:.3f}  P(hits>=7)={ge7_frozen/N:.4f}  P(hits>=1)={ge1_frozen/N:.4f}")
print(f"          P(>=7 & sum>=133.7pp)={ge7_sum_frozen/N:.4f}  P(>=7 & max>=24.86pp)={ge7_max_frozen/N:.4f}")
print(f"FULL PIPE: E[hits]={hits_pipe_sum/N:.3f}  P(hits>=7)={ge7_pipe/N:.5f}  P(hits>=1)={ge1_pipe/N:.4f}")
print(f"P(X<=Xobs)={Xle/N:.5f}  P(>=7|X<=obs)={cond_le_ge7/max(cond_le_n,1):.4f}  P(>=7|X>obs)={cond_gt_ge7/max(cond_gt_n,1):.5f}")
print(f"X: median={np.median(X_all):.4f} mean={X_all.mean():.4f} obs={Xobs:.4f} pct(obs)={100*(X_all<=Xobs).mean():.3f}%")
