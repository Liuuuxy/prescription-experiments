import json, itertools
import numpy as np, h5py
from scipy import stats
P="/data/xinyua11/robomimic_runs/prescribe_ph"
PROF=["balanced","starved_xlo_yhi","starved_xhi_ylo"]
REG=["xlo_ylo","xlo_yhi","xhi_ylo","xhi_yhi"]
ARMS=["D0"]+[f"add_{r}" for r in REG]+["add_cov","add_pfail","add_div1","add_div2"]+[f"rm_{r}" for r in REG]
# eval set composition
for f_ in ("E_test","E_probe"):
    z=np.load(f"{P}/{f_}.npz"); import collections
    print(f_, collections.Counter(z["region"].tolist()), "states shape", z["states"].shape)
# demo counts per region per mask
meta=json.load(open(f"{P}/regions.json")); ro=meta["region_of"]
h=h5py.File(f"{P}/can_ph_work.hdf5","r")
def comp(mask):
    d=[x.decode() for x in h["mask"][mask][:]]
    return {r:sum(1 for x in d if ro[x]==r) for r in REG}, len(d)
rows=[]
for p in PROF:
    for a in ARMS:
        c,n=comp(f"{p}_{a}")
        for s in (0,1):
            jr=json.load(open(f"{P}/out/{p}_{a}_s{s}/fixed_eval_test.json"))["J_region"]
            jrp=json.load(open(f"{P}/out/{p}_{a}_s{s}/fixed_eval_probe.json"))["J_region"]
            for r in REG:
                rows.append(dict(run=f"{p}_{a}_s{s}",reg=r,n=c[r],share=c[r]/n,ntot=n,J=jr[r],Jp=jrp[r],arm=a,prof=p,seed=s))
h.close()
print("n obs",len(rows),"runs",len({x['run'] for x in rows}))
runs=sorted({x["run"] for x in rows}); ridx={r:i for i,r in enumerate(runs)}
gidx={r:i for i,r in enumerate(REG)}
def twoway(yk,xk):
    y=np.array([x[yk] for x in rows])
    Xf=np.zeros((len(rows),len(runs)+len(REG)))
    for i,x in enumerate(rows):
        Xf[i,ridx[x["run"]]]=1; Xf[i,len(runs)+gidx[x["reg"]]]=1
    xv=np.array([x[xk] for x in rows],dtype=float)
    A=np.column_stack([Xf,xv])
    beta,_,_,_=np.linalg.lstsq(A,y,rcond=None)
    res=y-A@beta
    dof=len(y)-np.linalg.matrix_rank(A)
    s2=res@res/dof
    XtXi=np.linalg.pinv(A.T@A)
    se=np.sqrt(s2*XtXi[-1,-1])
    t=beta[-1]/se
    return beta[-1],se,t,dof,2*(1-stats.t.cdf(abs(t),dof))
for yk,lab in (("J","E_test"),("Jp","E_probe")):
    for xk in ("n","share"):
        b,se,t,dof,pv=twoway(yk,xk)
        print(f"{lab}: J_region ~ run FE + region FE + {xk}:  beta={b:.5f} se={se:.5f} t={t:.2f} dof={dof} p={pv:.3g}"
              + (f"   (=> +{b*24*100:.1f}pp per +24 demos)" if xk=="n" else f"   (=> +{b*0.2*100:.1f}pp per +0.2 share)"))
# cluster-robust by run
def twoway_cl(yk,xk):
    y=np.array([x[yk] for x in rows])
    Xf=np.zeros((len(rows),len(runs)+len(REG)))
    for i,x in enumerate(rows):
        Xf[i,ridx[x["run"]]]=1; Xf[i,len(runs)+gidx[x["reg"]]]=1
    xv=np.array([x[xk] for x in rows],dtype=float)
    A=np.column_stack([Xf,xv]); beta,_,_,_=np.linalg.lstsq(A,y,rcond=None); res=y-A@beta
    XtXi=np.linalg.pinv(A.T@A); meat=np.zeros_like(XtXi)
    for r in runs:
        idx=[i for i,x in enumerate(rows) if x["run"]==r]
        u=A[idx].T@res[idx]; meat+=np.outer(u,u)
    V=XtXi@meat@XtXi; se=np.sqrt(V[-1,-1]); 
    G=len(runs); adj=np.sqrt(G/(G-1))
    return beta[-1],se*adj,beta[-1]/(se*adj)
b,se,t=twoway_cl("J","n"); print(f"cluster-robust(by run, G=54) E_test beta_n={b:.5f} se={se:.5f} t={t:.2f} -> +24 demos = {b*24*100:+.1f}pp, p~{2*(1-stats.norm.cdf(abs(t))):.3g}")
