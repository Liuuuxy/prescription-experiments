#!/usr/bin/env python
"""Null distribution of best-mode-AUC (max over 40 modes) under RANDOM splits,
and class-stratified nuisance splits to kill any residual class confound."""
import json, numpy as np, torchvision
G="/data/xinyua11/xgradtest/gradlog/"; DATA="/data/xinyua11/xgradtest/data"
def unit(x,ax=-1): n=np.linalg.norm(x,axis=ax,keepdims=True); return x/(n+1e-12)
def auc(s,hot):
    o=np.argsort(s,kind='mergesort'); r=np.empty(len(s)); r[o]=np.arange(len(s))
    nh=hot.sum(); return (r[hot].sum()-nh*(nh-1)/2)/(nh*(~hot).sum())
def sym_auc(s,hot): a=auc(s,hot); return max(a,1-a)

cand=np.load(G+"cand_raw.npy",mmap_mode="r"); pl=np.load(G+"pool_labels.npy")
meta=json.load(open(G+"meta.json")); ckpts=meta["ckpts"]; pool=np.array(meta["pool"]); N=cand.shape[1]
ds=torchvision.datasets.CIFAR100(root=DATA,train=True,download=False)
imgs=ds.data[pool].astype(np.float32)
bright=imgs.mean(axis=(1,2,3)); red=imgs[...,0].mean(axis=(1,2))
mx=imgs.max(3); mn=imgs.min(3); sat=((mx-mn)/(mx+1e-6)).mean(axis=(1,2))

# Precompute U (top-40 modes) once per ckpt
Umap={}
for step in [3000,6000]:
    ti=ckpts.index(step)
    Xu=unit(np.asarray(cand[ti],dtype=np.float64),ax=1)
    U,S,_=np.linalg.svd(Xu,full_matrices=False)
    Umap[step]=U[:,:40].copy()
    print(f"step{step}: top-5 singular values {np.round(S[:5],2)}  (energy in mode0 frac={S[0]**2/np.sum(S**2):.3f})")

def bestmode(U,hot,k=40): return max(sym_auc(U[:,i],hot) for i in range(k))

# ---- NULL: 200 random 50/50 splits, distribution of best-mode AUC ----
print("\n--- NULL best-mode-AUC over 200 random 50/50 splits (max over 40 modes) ---")
rng=np.random.RandomState(12345)
for step in [3000,6000]:
    U=Umap[step]; vals=[]
    for _ in range(200):
        hot=rng.rand(N)>0.5
        vals.append(bestmode(U,hot))
    vals=np.array(vals)
    print(f"step{step}: null best-mode AUC  mean={vals.mean():.3f}  p50={np.percentile(vals,50):.3f}  p95={np.percentile(vals,95):.3f}  max={vals.max():.3f}")

# ---- compare each attribute to the null p95 ----
print("\n--- attribute best-mode vs NULL p95 (is it ABOVE the noise ceiling?) ---")
def top_half(v): return v>np.median(v)
ATTRS={"class(rare)":pl<20,"brightness":top_half(bright),"color(red)":top_half(red),"saturation":top_half(sat)}
rng2=np.random.RandomState(12345)
for step in [3000,6000]:
    U=Umap[step]
    nullv=np.array([bestmode(U,rng2.rand(N)>0.5) for _ in range(200)])
    p95=np.percentile(nullv,95)
    print(f"\nstep{step}: null p95={p95:.3f}")
    for nm,hot in ATTRS.items():
        bm=bestmode(U,hot)
        flag="ABOVE null p95" if bm>p95 else "within noise"
        print(f"  {nm:<12} best-mode={bm:.3f}   [{flag}]")

# ---- CLASS-STRATIFIED nuisance: split brightness within each class, balanced ----
print("\n--- class-stratified brightness (hot=brighter-than-class-median) ---")
strat_bright=np.zeros(N,bool)
for c in np.unique(pl):
    m=pl==c
    strat_bright[m]=bright[m]>np.median(bright[m])
for step in [3000,6000]:
    print(f"  step{step}: strat-brightness best-mode={bestmode(Umap[step],strat_bright):.3f}")
