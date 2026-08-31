"""C9 out-of-sample slope confirmation (spec frozen in PREREG_CORRECTION_2.md before launch).
Uses ONLY the 24 confirmation runs (seeds 2,3). Model: J_r ~ C(run)+C(region)+n_r (continuous
per-demo slope), run-clustered SE. Pre-registered read: slope > 0, clustered t >= 2, and
point estimate within a factor of 2 of the in-sample +0.44pp/demo; else C9's finding is
retracted. All verdicts conditional."""
import json, glob
import numpy as np
import h5py
P="/data/xinyua11/robomimic_runs/prescribe_ph"
meta=json.load(open(f"{P}/regions.json")); REG=meta["grid"]["labels"]; rof=meta["region_of"]
f=h5py.File(f"{P}/can_ph_work.hdf5","r")
def counts(mask):
    ds=[x.decode() for x in f["mask"][mask][:]]; c={r:0 for r in REG}
    for d in ds: c[rof[d]]+=1
    return c
conf=[r["name"] for r in json.load(open(f"{P}/confirm_manifest.json"))]
rows=[]
for n in conf:
    fp=f"{P}/out/{n}/fixed_eval_test.json"
    try: d=json.load(open(fp))
    except FileNotFoundError: print("[wait] missing",n); continue
    c=counts(d["mask"])
    for i,r in enumerate(REG): rows.append((n,i,c[r],d["J_region"][r]))
f.close()
print(f"{len(rows)//4}/{len(conf)} confirmation runs present")
if len(rows)<len(conf)*4: raise SystemExit("incomplete — rerun when pool finishes")
runs=sorted(set(r[0] for r in rows)); ridx={n:i for i,n in enumerate(runs)}
X=[]; Y=[]; CL=[]
for n,reg,nr,j in rows:
    x=np.zeros(len(runs)+3+1); x[ridx[n]]=1
    if reg>0: x[len(runs)+reg-1]=1
    x[-1]=nr
    X.append(x); Y.append(j); CL.append(ridx[n])
X=np.array(X); Y=np.array(Y); CL=np.array(CL)
b,_,_,_=np.linalg.lstsq(X,Y,rcond=None)
res=Y-X@b
XtXi=np.linalg.pinv(X.T@X)
meat=np.zeros((X.shape[1],X.shape[1]))
for g in np.unique(CL):
    Xg=X[CL==g]; ug=res[CL==g]
    meat+=np.outer(Xg.T@ug,Xg.T@ug)
V=XtXi@meat@XtXi
G=len(np.unique(CL)); V*=G/(G-1)
slope=b[-1]; se=np.sqrt(V[-1,-1]); t=slope/se
print(f"out-of-sample per-demo slope = {slope*100:+.3f} pp/demo  clustered SE {se*100:.3f}  t = {t:.2f}  (G={G} runs)")
IN=0.0044
lo,hi=IN/2,IN*2
if slope>0 and t>=2 and lo<=slope<=hi:
    print(f"VERDICT: CONFIRMED — slope in [{lo*100:.2f},{hi*100:.2f}] pp/demo with t>=2; the C9 regional-response finding stands.")
elif slope>0 and t>=2:
    print(f"VERDICT: PARTIAL — positive and significant but outside 2x of in-sample {IN*100:.2f}; report both numbers, claim direction only.")
else:
    print("VERDICT: NOT CONFIRMED — the pre-registered criterion failed; C9's regional-response finding is retracted per its own rule.")
