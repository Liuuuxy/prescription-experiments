#!/usr/bin/env python
"""Robustness: (1) random control across MANY seeds (no seed cherry-pick),
(2) direction-AUC of class is non-flipped, (3) nuisance signal magnitude vs class."""
import json, numpy as np, torchvision
G="/data/xinyua11/xgradtest/gradlog/"; DATA="/data/xinyua11/xgradtest/data"
def unit(x,ax=-1): n=np.linalg.norm(x,axis=ax,keepdims=True); return x/(n+1e-12)
def auc(s,hot):
    o=np.argsort(s,kind='mergesort'); r=np.empty(len(s)); r[o]=np.arange(len(s))
    nh=hot.sum(); return (r[hot].sum()-nh*(nh-1)/2)/(nh*(~hot).sum())
def sym_auc(s,hot): a=auc(s,hot); return max(a,1-a)
cand=np.load(G+"cand_raw.npy",mmap_mode="r"); pl=np.load(G+"pool_labels.npy")
gfail=np.load(G+"gfail.npy"); meta=json.load(open(G+"meta.json")); ckpts=meta["ckpts"]
pool=np.array(meta["pool"]); N=cand.shape[1]
rare=pl<20

# random control over 10 different seeds for the *defined* attribute (not best-of)
print("--- random-attribute best-mode across 10 seeds (implementer used seed 0) ---")
for step in [3000,6000]:
    ti=ckpts.index(step)
    Xu=unit(np.asarray(cand[ti],dtype=np.float64),ax=1)
    U,_,_=np.linalg.svd(Xu,full_matrices=False); U=U[:,:40]
    vals=[]
    for seed in range(10):
        hot=np.random.RandomState(seed).rand(N)>0.5
        vals.append(max(sym_auc(U[:,i],hot) for i in range(40)))
    print(f"  step{step}: seed0={vals[0]:.3f}  10-seed mean={np.mean(vals):.3f}  range[{min(vals):.3f},{max(vals):.3f}]")

# direction AUC for class, signed (must be >0.5 = rare ranked high, non-flipped)
print("\n--- class direction-AUC sign check (must be >0.5, rare ranked high) ---")
for step in [3000,6000]:
    ti=ckpts.index(step)
    Xu=unit(np.asarray(cand[ti],dtype=np.float64),ax=1)
    g=unit(Xu[rare].mean(0)-Xu[~rare].mean(0))
    a=auc(Xu@g,rare)
    cg=auc(Xu@unit(gfail[ti].astype(np.float64)),rare)
    print(f"  step{step}: class dir-AUC={a:.3f} (signed, >0.5 OK)   cos-to-gfail AUC={cg:.3f}")
