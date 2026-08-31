import json, os, itertools
import numpy as np
P="/data/xinyua11/robomimic_runs/prescribe_ph"
PROF=["balanced","starved_xlo_yhi","starved_xhi_ylo"]
REG=["xlo_ylo","xlo_yhi","xhi_ylo","xhi_yhi"]
ADD=[f"add_{r}" for r in REG]+["add_cov","add_pfail","add_div1","add_div2"]
def J(n,f="J_deploy"):
    return json.load(open(f"{P}/out/{n}/fixed_eval_test.json"))[f]
Jd={}; Ju={}; S={}
for p in PROF:
    for a in ADD+["D0"]:
        for s in (0,1):
            n=f"{p}_{a}_s{s}"
            d=json.load(open(f"{P}/out/{n}/fixed_eval_test.json"))
            Jd[(p,a,s)]=d["J_deploy"]; Ju[(p,a,s)]=d["J_uniform"]
            S[(p,a,s)]=np.array(d["successes"])
print("=== raw J_deploy (E_test) ===")
print(f"{'arm':12s} " + "  ".join(f"{p[:9]:>9s}s{s}" for p in PROF for s in (0,1)))
for a in ["D0"]+ADD:
    print(f"{a:12s} " + "  ".join(f"{Jd[(p,a,s)]:11.3f}" for p in PROF for s in (0,1)))

# --- 1. reproduce gate1
X=json.load(open(f"{P}/MDE.json"))["X"]
def gate(Jd, divnames=("add_div1","add_div2"), Xv=None):
    per=[]
    dvals=[]
    for p in PROF:
        for s in (0,1):
            dvals.append(Jd[(p,divnames[0],s)]-Jd[(p,divnames[1],s)])
    pm=[np.mean(dvals[2*i:2*i+2]) for i in range(3)]
    sdp=np.std(pm,ddof=1)
    Xc=3.9806457521353384*sdp/np.sqrt(3)
    use = X if Xv is None else Xv
    hits=[]
    for p in PROF:
        for a in ADD:
            if a in divnames: continue
            advs=[Jd[(p,a,s)]-0.5*(Jd[(p,divnames[0],s)]+Jd[(p,divnames[1],s)]) for s in (0,1)]
            m=np.mean(advs)
            if m>use and np.sign(advs[0])==np.sign(advs[1]):
                hits.append((p,a,m))
    return hits,Xc,sdp
h,Xc,sdp=gate(Jd)
print(f"\nreproduced: X={X:.6f} recomputed={Xc:.6f} sd_prof={sdp:.6f} hits={len(h)}")
for p,a,m in h: print(f"   {p}/{a} {m*100:+.1f}pp")
print("sum adv", sum(m for _,_,m in h)*100, "max", max(m for _,_,m in h)*100, "min",min(m for _,_,m in h)*100)

# --- 2. pure seed-to-seed (data IDENTICAL) spread
print("\n=== seed0 vs seed1 within arm (data identical -> pure training stochasticity) ===")
diffs=[Jd[(p,a,0)]-Jd[(p,a,1)] for p in PROF for a in ADD+["D0"]]
print("n cells",len(diffs))
print("mean |diff| %.4f  sd(diff) %.4f  -> per-run sd %.4f"%(np.mean(np.abs(diffs)),np.std(diffs,ddof=1),np.std(diffs,ddof=1)/np.sqrt(2)))
print("range", min(diffs), max(diffs))
# same for J_uniform
du=[Ju[(p,a,0)]-Ju[(p,a,1)] for p in PROF for a in ADD+["D0"]]
print("J_uniform: sd(diff) %.4f -> per-run sd %.4f"%(np.std(du,ddof=1),np.std(du,ddof=1)/np.sqrt(2)))
