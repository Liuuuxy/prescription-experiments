#!/usr/bin/env python3
"""Visualize RC-LESS D_val gradient clusters — show they carry ~no object/semantic structure."""
import numpy as np, json
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
NAVY=(0.07,0.137,0.247)
D=np.load('/data/xinyua11/robocasa/weakregion/rcless/dval_sketch.npy').astype(np.float64)
cats=json.load(open('/data/xinyua11/robocasa/weakregion/rcless/dval_meta.json'))['categories']
def unit(x): return x/(np.linalg.norm(x,axis=-1,keepdims=True)+1e-12)
U=unit(D)
def kmeans(X,k,iters=50,seed=0):
    rng=np.random.RandomState(seed); C=X[rng.choice(len(X),k,replace=False)].copy()
    for _ in range(iters):
        d=((X[:,None,:]-C[None,:,:])**2).sum(-1); a=d.argmin(1)
        newC=np.stack([X[a==j].mean(0) if (a==j).any() else C[j] for j in range(k)])
        if np.allclose(newC,C): break
        C=newC
    return a
lab=kmeans(U,14,seed=0)
# 2D PCA on centered unit sketches
Dc=U-U.mean(0); Uu,S,Vt=np.linalg.svd(Dc,full_matrices=False); XY=Dc@Vt[:2].T
var=(S**2/(S**2).sum())[:2]
# category -> mean height
eb=json.load(open('/data/xinyua11/robocasa/weakregion/eval_baseline/weakregion.json'))['episodes']
hb=defaultdict(list)
for e in eb:
    if e.get('obj_height') is not None: hb[e['object_category']].append(e['obj_height'])
catH={c:np.mean(v) for c,v in hb.items()}
H=np.array([catH.get(c,np.nan)*100 for c in cats])

fig,(axA,axB)=plt.subplots(1,2,figsize=(12.4,5.2))
axA.scatter(XY[:,0],XY[:,1],c=lab,cmap='tab20',s=26,alpha=0.85,edgecolors='none')
axA.set_title("Colored by k-means cluster (m=14)\ndemos barely co-cluster by object (0.41 vs 0.36 chance)",
              fontsize=12.5,fontweight='bold',color=NAVY,loc='left')
sc=axB.scatter(XY[:,0],XY[:,1],c=H,cmap='viridis',s=26,alpha=0.9,edgecolors='none')
axB.set_title("Colored by object height\nno height separation (η²=0.026)",
              fontsize=12.5,fontweight='bold',color=NAVY,loc='left')
plt.colorbar(sc,ax=axB,label='object height (cm)')
for ax in (axA,axB):
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)"); ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ('top','right'): ax.spines[s].set_visible(False)
fig.suptitle("RC-LESS D_val gradient clusters carry ~no object/semantic structure "
             "(a diffuse 4096-D cloud: top-2 PCs = 6% var)",
             fontsize=13.5,fontweight='bold',color=NAVY,y=1.02,x=0.02,ha='left')
plt.tight_layout(); plt.savefig('/data/xinyua11/robocasa/talk/figs/fig_rcless_clusters.png',dpi=150,bbox_inches='tight'); plt.close()
print("wrote fig_rcless_clusters.png")
