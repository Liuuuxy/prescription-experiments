#!/usr/bin/env python
"""INDEPENDENT adversarial re-verification of cifar_control finding.
Written from scratch by the verifier. Does NOT import implementer's code."""
import json, numpy as np, torchvision

G="/data/xinyua11/xgradtest/gradlog/"
DATA="/data/xinyua11/xgradtest/data"

def unit(x,ax=-1):
    n=np.linalg.norm(x,axis=ax,keepdims=True); return x/(n+1e-12)

def auc(s,hot):
    # rank-sum AUC: prob a random hot scores above a random cold (ties=0.5)
    # use scipy-free average-rank to handle ties correctly
    order=np.argsort(s,kind='mergesort')
    ranks=np.empty(len(s),float); ranks[order]=np.arange(1,len(s)+1)
    # average ranks for ties
    s_sorted=s[order]
    i=0
    while i<len(s):
        j=i
        while j+1<len(s) and s_sorted[j+1]==s_sorted[i]: j+=1
        if j>i:
            avg=(ranks[order[i]]+ranks[order[j]])/2
            for k in range(i,j+1): ranks[order[k]]=avg
        i=j+1
    nh=hot.sum(); nc=(~hot).sum()
    return (ranks[hot].sum()-nh*(nh+1)/2)/(nh*nc)

def sym_auc(s,hot):
    a=auc(s,hot); return max(a,1.0-a)

cand=np.load(G+"cand_raw.npy",mmap_mode="r")
pl=np.load(G+"pool_labels.npy")
gfail=np.load(G+"gfail.npy"); gval=np.load(G+"gval.npy")
meta=json.load(open(G+"meta.json")); ckpts=meta["ckpts"]; pool=np.array(meta["pool"])
N=cand.shape[1]

ds=torchvision.datasets.CIFAR100(root=DATA,train=True,download=False)
imgs=ds.data[pool].astype(np.float32); targets=np.array(ds.targets)[pool]
assert np.array_equal(targets,pl), "pool/image mismatch!"
print("pool/image correspondence OK; N=",N)

bright=imgs.mean(axis=(1,2,3))
red=imgs[...,0].mean(axis=(1,2))
mx=imgs.max(3); mn=imgs.min(3); sat=((mx-mn)/(mx+1e-6)).mean(axis=(1,2))
def top_half(v): return v>np.median(v)
rng=np.random.RandomState(0); rhot=rng.rand(N)>0.5

ATTRS={"class(rare)":pl<20,"brightness":top_half(bright),
       "color(red)":top_half(red),"saturation":top_half(sat),"random":rhot}

# ---- CONFOUND CHECK: is brightness/sat split correlated with class? ----
print("\n--- CONFOUND CHECK (does nuisance hot-half align with rare class?) ---")
rare=pl<20
for nm in ["brightness","color(red)","saturation"]:
    h=ATTRS[nm]
    # fraction rare among hot vs cold
    print(f"{nm:<12} P(rare|hot)={rare[h].mean():.3f}  P(rare|cold)={rare[~h].mean():.3f}  (balanced if ~0.30)")

# also: is rare class systematically brighter? (i.e. could class-mode 'accidentally' read brightness)
print(f"\nmean brightness rare={bright[rare].mean():.1f} vs common={bright[~rare].mean():.1f}")
print(f"mean red       rare={red[rare].mean():.1f} vs common={red[~rare].mean():.1f}")
print(f"mean sat       rare={sat[rare].mean():.3f} vs common={sat[~rare].mean():.3f}")

print("\n--- BEST-MODE SVD AUC (max over top-40 modes, symmetric) ---")
print(f"{'attr':<14}"+ "".join(f"step{s:<7}" for s in [3000,6000]))
res={}
for step in [3000,6000]:
    ti=ckpts.index(step)
    X=np.asarray(cand[ti],dtype=np.float64)
    Xu=unit(X,ax=1)
    U,S,_=np.linalg.svd(Xu,full_matrices=False)
    res[step]={}
    for nm,hot in ATTRS.items():
        best=max(sym_auc(U[:,i],hot) for i in range(min(40,U.shape[1])))
        res[step][nm]=best
for nm in ATTRS:
    print(f"{nm:<14}"+"".join(f"{res[s][nm]:<11.3f}" for s in [3000,6000]))

cls=np.mean([res[s]["class(rare)"] for s in [3000,6000]])
nuis=np.mean([res[s][n] for s in [3000,6000] for n in ["brightness","color(red)","saturation"]])
rnd=np.mean([res[s]["random"] for s in [3000,6000]])
print(f"\nmean best-mode: class={cls:.3f} nuisances={nuis:.3f} random={rnd:.3f}")
print(f"class-nuisance gap={cls-nuis:.3f}")

# ---- cos-to-gfail sanity (reference says ~0.96) ----
print("\n--- cos(unit(cand[ti]), unit(gfail[ti])) AUC for rare class ---")
for step in [3000,6000]:
    ti=ckpts.index(step)
    Xu=unit(np.asarray(cand[ti],dtype=np.float64),ax=1)
    g=unit(gfail[ti].astype(np.float64))
    s=Xu@g
    print(f"  step{step}: cos-to-gfail AUC(rare>common)={auc(s,rare):.3f}")
