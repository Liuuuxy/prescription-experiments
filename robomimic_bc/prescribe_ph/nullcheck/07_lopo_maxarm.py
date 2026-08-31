import json
import numpy as np
rng=np.random.default_rng(3)
P="/data/xinyua11/robomimic_runs/prescribe_ph"
PROFILES=["balanced","starved_xlo_yhi","starved_xhi_ylo"]
REGIONS=["xlo_ylo","xlo_yhi","xhi_ylo","xhi_yhi"]
ARMS=[f"add_{r}" for r in REGIONS]+["add_cov","add_pfail"]
ALL8=ARMS+["add_div1","add_div2"]
def jt(n): return json.load(open(f"{P}/out/{n}/fixed_eval_test.json"))["J_deploy"]
V={p:np.array([[jt(f"{p}_{a}_s{s}") for s in (0,1)] for a in ALL8]) for p in PROFILES}
def statset(profs,assign):
    A=np.zeros(6)
    for p in profs:
        v=V[p][assign[p]]; div=0.5*(v[6]+v[7]); A+=(v[:6]-div).mean(axis=1)
    return A/len(profs)
ident={p:np.arange(8) for p in PROFILES}
print("leave-one-profile-out: max-arm mean adv and its permutation p")
for drop in [None]+PROFILES:
    profs=[p for p in PROFILES if p!=drop]
    o=statset(profs,ident); om=o.max()
    N=100000
    cnt=0
    for _ in range(N):
        asg={p:rng.permutation(8) for p in profs}
        if statset(profs,asg).max()>=om: cnt+=1
    print("  drop=%-16s best arm %-12s  %+6.2fpp   p=%.4f   (per-arm: %s)"
          %(str(drop),ARMS[int(o.argmax())],om*100,(cnt+1)/(N+1),
            ", ".join("%s %+0.1f"%(a.replace('add_',''),x*100) for a,x in zip(ARMS,o))))
