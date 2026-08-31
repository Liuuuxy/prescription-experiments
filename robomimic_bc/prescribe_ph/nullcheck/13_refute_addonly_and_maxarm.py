import json, itertools
import numpy as np, h5py
from scipy import stats
P="/data/xinyua11/robomimic_runs/prescribe_ph"
PROF=["balanced","starved_xlo_yhi","starved_xhi_ylo"]
REG=["xlo_ylo","xlo_yhi","xhi_ylo","xhi_yhi"]
ADD=[f"add_{r}" for r in REG]+["add_cov","add_pfail","add_div1","add_div2"]
meta=json.load(open(f"{P}/regions.json")); ro=meta["region_of"]
h=h5py.File(f"{P}/can_ph_work.hdf5","r")
rows=[]
for p in PROF:
    for a in ADD:
        d=[x.decode() for x in h["mask"][f"{p}_{a}"][:]]; c={r:sum(1 for x in d if ro[x]==r) for r in REG}
        assert len(d)==104
        for s in (0,1):
            jr=json.load(open(f"{P}/out/{p}_{a}_s{s}/fixed_eval_test.json"))["J_region"]
            for r in REG: rows.append((f"{p}_{a}_s{s}",r,c[r],jr[r]))
h.close()
runs=sorted({x[0] for x in rows}); ridx={r:i for i,r in enumerate(runs)}; gidx={r:i for i,r in enumerate(REG)}
y=np.array([x[3] for x in rows]); A=np.zeros((len(rows),len(runs)+4+1))
for i,x in enumerate(rows): A[i,ridx[x[0]]]=1; A[i,len(runs)+gidx[x[1]]]=1; A[i,-1]=x[2]
b,_,_,_=np.linalg.lstsq(A,y,rcond=None); res=y-A@b; dof=len(y)-np.linalg.matrix_rank(A)
se=np.sqrt((res@res/dof)*np.linalg.pinv(A.T@A)[-1,-1])
print(f"ADD-ARMS ONLY (all ntot=104, 48 runs, 192 obs): beta_n={b[-1]:.5f} se={se:.5f} t={b[-1]/se:.2f} -> +24 demos {b[-1]*24*100:+.1f}pp  p={2*(1-stats.t.cdf(abs(b[-1]/se),dof)):.3g}")

# reproduce analyst's max-arm statistic
V=np.zeros((3,8,2))
for i,p in enumerate(PROF):
    for j,a in enumerate(ADD):
        for s in (0,1): V[i,j,s]=json.load(open(f"{P}/out/{p}_{a}_s{s}/fixed_eval_test.json"))["J_deploy"]
d1,d2=ADD.index("add_div1"),ADD.index("add_div2")
base=0.5*(V[:,d1,:]+V[:,d2,:])
means={ADD[j]:float((V[:,j,:]-base).mean()) for j in range(8) if j not in (d1,d2)}
print("per-arm mean adv vs div:", {k:round(v*100,1) for k,v in sorted(means.items(),key=lambda kv:-kv[1])})
obsmax=max(means.values())
pairs=[(i,j) for i in range(8) for j in range(8) if i!=j]
rng=np.random.default_rng(1); N=200000; ge=0; nulls=[]
for _ in range(N):
    lab=[pairs[rng.integers(56)] for _ in range(3)]
    bs=np.array([0.5*(V[i,lab[i][0],:]+V[i,lab[i][1],:]) for i in range(3)])
    mm=[]
    for j in range(8):
        vals=[]
        for i in range(3):
            if j in lab[i]: vals=None;break
            vals.append(V[i,j,:]-bs[i])
        if vals is not None: mm.append(np.mean(vals))
    if mm:
        nulls.append(max(mm))
        if max(mm)>=obsmax-1e-12: ge+=1
nulls=np.array(nulls)
print(f"max-arm obs {obsmax*100:+.2f}pp  perm p={ge/len(nulls):.4f}  null median {np.median(nulls)*100:+.1f} q95 {np.quantile(nulls,.95)*100:+.1f}")
