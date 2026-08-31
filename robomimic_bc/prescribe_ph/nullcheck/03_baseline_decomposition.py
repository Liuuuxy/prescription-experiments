import json
import numpy as np
from scipy import stats
P="/data/xinyua11/robomimic_runs/prescribe_ph"
PROFILES=["balanced","starved_xlo_yhi","starved_xhi_ylo"]
REGIONS=["xlo_ylo","xlo_yhi","xhi_ylo","xhi_yhi"]
ARMS=[f"add_{r}" for r in REGIONS]+["add_cov","add_pfail"]
ALL8=ARMS+["add_div1","add_div2"]
X=json.load(open(f"{P}/MDE.json"))["X"]
def jt(n): return json.load(open(f"{P}/out/{n}/fixed_eval_test.json"))["J_deploy"]
J={p:{s:{a:jt(f"{p}_{a}_s{s}") for a in ALL8} for s in (0,1)} for p in PROFILES}
D0={p:{s:jt(f"{p}_D0_s{s}") for s in (0,1)} for p in PROFILES}

R=np.load("/data/xinyua11/tmp/claude-969153092/-data-xinyua11-robocasa/a6f630e5-2182-431d-8c49-04c4acb3a9d1/scratchpad/perm.npz")
sel=R["reX"]<=X
print("P(nhits>=7 | X <= observed X) = %.4f  (n=%d of %d relabelings)"
      % ((R["reX_nhits"][sel]>=7).mean(), sel.sum(), len(sel)))
print("P(nhits>=7 | X  > observed X) = %.5f" % ((R["reX_nhits"][~sel]>=7).mean()))
print("mean nhits | X<=obs : %.2f ;  | X>obs : %.2f"
      % (R["reX_nhits"][sel].mean(), R["reX_nhits"][~sel].mean()))

print("\n================ Q2: is the 'hit' pattern a low DIVERSE baseline? ================")
print("%-16s %4s %8s %8s %8s %9s %9s" % ("profile","seed","J(D0)","J(div)","mean8","div-D0","div-mean8"))
lucky={}
for p in PROFILES:
    for s in (0,1):
        m8=np.mean([J[p][s][a] for a in ALL8])
        dv=0.5*(J[p][s]["add_div1"]+J[p][s]["add_div2"])
        lucky[(p,s)]=(D0[p][s],dv,m8)
        print("%-16s %4d %8.3f %8.3f %8.3f %+9.3f %+9.3f"%(p,s,D0[p][s],dv,m8,dv-D0[p][s],dv-m8))

def table(basefn,label):
    print("\n--- advantages vs %s ---"%label)
    hits=[]
    for p in PROFILES:
        for a in ARMS:
            advs=[J[p][s][a]-basefn(p,s) for s in (0,1)]
            m=float(np.mean(advs)); rep=np.sign(advs[0])==np.sign(advs[1])
            hit=(m>X) and rep
            if hit: hits.append((p,a,m))
            print("  %-16s %-12s %+6.1f %+6.1f  mean %+6.1fpp%s"
                  %(p,a,advs[0]*100,advs[1]*100,m*100,"  <-- HIT" if hit else ""))
    print("  => %d hits / 18"%len(hits))
    return hits

h_div=table(lambda p,s: 0.5*(J[p][s]["add_div1"]+J[p][s]["add_div2"]), "div pair (as pre-registered)")
h_d0 =table(lambda p,s: D0[p][s], "J(D0) at same profile+seed")
h_m8 =table(lambda p,s: np.mean([J[p][s][a] for a in ALL8]), "mean of all 8 additions at same profile+seed")
h_m6 =table(lambda p,s: np.mean([J[p][s][a] for a in ARMS]), "mean of the 6 non-div arms")

print("\n--- decomposition of each pre-registered advantage ---")
print("  adv_vs_div = (J_arm - mean8)  +  (mean8 - J_div)      [arm term + baseline-luck term]")
print("%-16s %-12s %9s %9s %9s"%("profile","arm","adv","arm term","luck term"))
tot_a=tot_l=0
for p,a,m in h_div:
    at=np.mean([J[p][s][a]-np.mean([J[p][s][x] for x in ALL8]) for s in (0,1)])
    lt=np.mean([np.mean([J[p][s][x] for x in ALL8])-0.5*(J[p][s]["add_div1"]+J[p][s]["add_div2"]) for s in (0,1)])
    tot_a+=at; tot_l+=lt
    print("%-16s %-12s %+8.1f %+8.1f %+8.1f"%(p,a,m*100,at*100,lt*100))
print("  mean over the 7 hits: adv %+.1fpp = arm %+.1fpp + luck %+.1fpp   (luck share %.0f%%)"
      %( (tot_a+tot_l)/7*100, tot_a/7*100, tot_l/7*100, 100*tot_l/(tot_a+tot_l)))

print("\n--- variance decomposition of J_deploy across the 18 addition runs ---")
Y=np.array([[J[p][s][a] for a in ALL8] for p in PROFILES for s in (0,1)])  # 6 x 8
gm=Y.mean()
ss_tot=((Y-gm)**2).sum()
cell=Y.mean(axis=1,keepdims=True)
ss_cell=(8*(cell-gm)**2).sum()          # profile x seed (i.e. seed/draw block) effect
ss_within=((Y-cell)**2).sum()           # arm + noise, within profile-seed
armmean=Y.mean(axis=0)
ss_arm=(6*(armmean-gm)**2).sum()
ss_resid=ss_tot-ss_cell-ss_arm
print("  SS total %.4f | profile x seed block %.4f (%.0f%%) | arm main effect %.4f (%.0f%%) | residual %.4f (%.0f%%)"
      %(ss_tot,ss_cell,100*ss_cell/ss_tot,ss_arm,100*ss_arm/ss_tot,ss_resid,100*ss_resid/ss_tot))
F=(ss_arm/7)/(ss_resid/(len(Y.ravel())-1-7-5))
print("  arm main-effect F(7,35) = %.2f  p = %.3f"%(F,1-stats.f.cdf(F,7,35)))
print("  sd within profile-seed across 8 arms (pooled) = %.4f"%np.sqrt(ss_within/(6*7)))
