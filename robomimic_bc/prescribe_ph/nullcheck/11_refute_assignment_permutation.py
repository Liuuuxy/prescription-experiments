import json, itertools
import numpy as np
from scipy import stats
P="/data/xinyua11/robomimic_runs/prescribe_ph"
PROF=["balanced","starved_xlo_yhi","starved_xhi_ylo"]
REG=["xlo_ylo","xlo_yhi","xhi_ylo","xhi_yhi"]
def JR(n,w="test"): return json.load(open(f"{P}/out/{n}/fixed_eval_{w}.json"))["J_region"]
def JU(n,w="test"): return json.load(open(f"{P}/out/{n}/fixed_eval_{w}.json"))["J_uniform"]
q=json.load(open(f"{P}/deploy_shares.json"))["natural_reset_shares"]
CELLS=[(p,s) for p in PROF for s in (0,1)]

for w in ("test","probe"):
  print(f"\n===================== {w.upper()} =====================")
  for FAM,pref in (("add","add_"),("rm","rm_")):
    # M[cell][j][r] = J_region[r] of the run that manipulated region j
    M=np.array([[[JR(f"{p}_{pref}{REG[j]}_s{s}",w)[REG[r]] for r in range(4)] for j in range(4)] for p,s in CELLS])
    # centre each RUN by its own mean over regions (removes run-level quality)
    C=M-M.mean(axis=2,keepdims=True)
    diag=np.array([[C[c,j,j] for j in range(4)] for c in range(len(CELLS))])
    obs=diag.mean()
    # own-region vs OTHER RUNS' same region (controls region difficulty explicitly)
    off=np.array([[np.mean([C[c,j2,j] for j2 in range(4) if j2!=j]) for j in range(4)] for c in range(len(CELLS))])
    contrast=(diag-off)
    cellm=contrast.mean(axis=1)
    t,pp=stats.ttest_1samp(cellm,0)
    print(f" {FAM}: own-region centred effect {obs*100:+.2f}pp ; own-vs-otherruns-same-region {contrast.mean()*100:+.2f}pp")
    print(f"      clustered n=6 cells: mean {cellm.mean()*100:+.2f}pp sd {cellm.std(ddof=1)*100:.2f} t={t:+.3f} p2={pp:.4f} cells "+" ".join(f"{z*100:+.1f}" for z in cellm))
    # EXACT assignment permutation: per cell enumerate 4!=24 bijections, convolve
    percell=[]
    for c in range(len(CELLS)):
        vals=[sum(C[c,j,pi[j]] for j in range(4)) for pi in itertools.permutations(range(4))]
        percell.append(np.array(vals))
    tot=percell[0]
    for k in range(1,len(percell)):
        tot=(tot[:,None]+percell[k][None,:]).ravel()
    obs_sum=diag.sum()
    p=float((tot>=obs_sum-1e-12).mean())
    print(f"      EXACT assignment-permutation (24^6={len(tot)}): obs sum {obs_sum:.4f}, null mean {tot.mean():.4f} sd {tot.std():.4f}, p={p:.6g}")
    # net effect on overall performance
    ju_t=np.array([[JU(f"{p}_{pref}{REG[j]}_s{s}",w) for j in range(4)] for p,s in CELLS])
    ju_div=np.array([np.mean([JU(f"{p}_add_div{k}_s{s}",w) for k in (1,2)]) for p,s in CELLS])
    d=ju_t.mean(axis=1)-ju_div
    t2,p2=stats.ttest_1samp(d,0)
    print(f"      NET J_uniform: mean({FAM} arms) - div = {d.mean()*100:+.2f}pp (cells "+" ".join(f"{z*100:+.1f}" for z in d)+f") t={t2:+.2f} p={p2:.3f}")
  # decompose add: own region up? others down?
  own=[];oth=[]
  for ci,(p,s) in enumerate(CELLS):
    for j in range(4):
        jr=JR(f"{p}_add_{REG[j]}_s{s}",w); dv={r:np.mean([JR(f"{p}_add_div{k}_s{s}",w)[r] for k in (1,2)]) for r in REG}
        own.append(jr[REG[j]]-dv[REG[j]]); oth.append(np.mean([jr[REG[r]]-dv[REG[r]] for r in range(4) if r!=j]))
  own=np.array(own);oth=np.array(oth)
  print(f" decomposition vs div: own region {own.mean()*100:+.2f}pp   other regions {oth.mean()*100:+.2f}pp   gap {np.mean(own-oth)*100:+.2f}pp")
