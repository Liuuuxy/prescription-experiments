"""Square heterogeneity verdict, per PREREG_SQUARE_HET.md (frozen spec, conditional verdicts).
M0: J_r ~ C(run)+C(region)+C(dose)   [dose in {removed, base, augmented}]
M1: + region x dose interaction.
Test: run-clustered Wald on the 6 interaction terms (4 regions x 3 doses -> (4-1)x(3-1)=6 df).
PASS needs p<0.05 AND >=1 region own-dose slope differing from pooled by >2x with |t|>=2.
Placebo check: DiD on untreated regions must be small relative to claimed heterogeneity."""
import json, glob
import numpy as np
import h5py
from scipy import stats
P="/data/xinyua11/robomimic_runs/prescribe_sq"
meta=json.load(open(f"{P}/regions.json")); REG=meta["grid"]["labels"]; rof=meta["region_of"]
f=h5py.File(f"{P}/square_ph_work.hdf5","r")
def counts(mask):
    ds=[x.decode() for x in f["mask"][mask][:]]; c={r:0 for r in REG}
    for d in ds: c[rof[d]]+=1
    return c
rows=[]
for fp in glob.glob(f"{P}/out/screen_*/fixed_eval_test.json"):
    d=json.load(open(fp)); c=counts(d["mask"])
    for i,r in enumerate(REG):
        dose=0 if c[r]==0 else (1 if c[r]<=20 else 2)   # removed / base(20 or div 26) / augmented(44)
        # div adds 6/region -> 26: still 'base-ish'; classify by prereg bins [0,1,N/4,N/4+24+1): 26 falls in bin2? N/4=20; bins {0},[1,20],[21,44]. div=26 -> augmented-lite. Use raw n_r as continuous too.
        rows.append((d["name"],i,c[r],d["J_region"][r]))
f.close()
runs=sorted(set(r[0] for r in rows)); ridx={n:i for i,n in enumerate(runs)}
G=len(runs)
print(f"{G} screen runs x 4 regions = {len(rows)} obs")
if G<33: raise SystemExit("[wait] screen incomplete")
def build(interact):
    X=[]; Y=[]; CL=[]
    for n,reg,nr,j in rows:
        dose=0 if nr==0 else (1 if nr<=20 else 2)
        x=np.zeros(G+3+2+(6 if interact else 0))
        x[ridx[n]]=1
        if reg>0: x[G+reg-1]=1
        if dose>0: x[G+3+dose-1]=1
        if interact and reg>0 and dose>0:
            x[G+5+(reg-1)*2+(dose-1)]=1
        X.append(x); Y.append(j); CL.append(ridx[n])
    return np.array(X),np.array(Y),np.array(CL)
def cluster_fit(X,Y,CL):
    b,_,_,_=np.linalg.lstsq(X,Y,rcond=None)
    res=Y-X@b; XtXi=np.linalg.pinv(X.T@X)
    meat=np.zeros((X.shape[1],)*2)
    for g in np.unique(CL):
        s=X[CL==g].T@res[CL==g]; meat+=np.outer(s,s)
    V=XtXi@meat@XtXi*G/(G-1)
    return b,V
X1,Y,CL=build(True)
b1,V1=cluster_fit(X1,Y,CL)
idx=list(range(G+5,G+11))
bb=b1[idx]; VV=V1[np.ix_(idx,idx)]
W=float(bb@np.linalg.pinv(VV)@bb)
p=1-stats.chi2.cdf(W,6)
print(f"\nHETEROGENEITY WALD: chi2(6) = {W:.2f}, p = {p:.4f}  (run-clustered)")
# per-region own-dose slope (continuous n_r), pooled slope
def slope_fit(region=None):
    X=[]; Yv=[]; C=[]
    for n,reg,nr,j in rows:
        if region is not None and reg!=region: continue
        x=np.zeros(G+3+1); x[ridx[n]]=1
        if reg>0: x[G+reg-1]=1
        x[-1]=nr
        X.append(x); Yv.append(j); C.append(ridx[n])
    X=np.array(X); Yv=np.array(Yv); C=np.array(C)
    b,V=cluster_fit(X,Yv,C)
    return b[-1],np.sqrt(V[-1,-1])
sp,sep=slope_fit()
print(f"pooled slope: {sp*100:+.3f} pp/demo (SE {sep*100:.3f}, t {sp/sep:.2f})")
notable=[]
for i,r in enumerate(REG):
    s,se=slope_fit(i)
    flag=""
    if se>0 and abs(s/se)>=2 and (s>2*sp or s<sp/2):
        flag=" <-- differs >2x from pooled with |t|>=2"; notable.append(r)
    print(f"  {r:10s} slope {s*100:+.3f} (SE {se*100:.3f}, t {s/se:+.2f}){flag}")
if p<0.05 and notable:
    print(f"\nVERDICT: HETEROGENEITY DETECTED (p={p:.4f}; regions {notable}). Pending placebo check "
          "and adversarial verification before any claim; if it survives, Square becomes the Gate-2 benchmark.")
elif p<0.05:
    print(f"\nVERDICT: interaction significant (p={p:.4f}) but no region clears the 2x+t>=2 slope bar "
          "-> does NOT meet the pre-registered PASS; report as suggestive only.")
else:
    print(f"\nVERDICT: NO detectable heterogeneity (p={p:.4f}) -> two-domain negative control; "
          "allocation program moves to heterogeneous-COST settings per prereg.")
