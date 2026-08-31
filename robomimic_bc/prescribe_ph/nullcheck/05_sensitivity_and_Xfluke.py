import json, itertools
import numpy as np
from scipy import stats
rng=np.random.default_rng(20260825)
P="/data/xinyua11/robomimic_runs/prescribe_ph"
PROFILES=["balanced","starved_xlo_yhi","starved_xhi_ylo"]
REGIONS=["xlo_ylo","xlo_yhi","xhi_ylo","xhi_yhi"]
ARMS=[f"add_{r}" for r in REGIONS]+["add_cov","add_pfail"]
ALL8=ARMS+["add_div1","add_div2"]
X=json.load(open(f"{P}/MDE.json"))["X"]
def load(n):
    d=json.load(open(f"{P}/out/{n}/fixed_eval_test.json")); return d
J={p:{s:{a:load(f"{p}_{a}_s{s}")["J_deploy"] for a in ALL8} for s in (0,1)} for p in PROFILES}
S={p:{s:{a:np.array(load(f"{p}_{a}_s{s}")["successes"]) for a in ALL8} for s in (0,1)} for p in PROFILES}
reg=np.load(f"{P}/E_test.npz",allow_pickle=True)["region"]
q=json.load(open(f"{P}/deploy_shares.json"))["natural_reset_shares"]; print("deploy shares:",q)
regs=np.array([str(r) for r in reg])
# per-region weight normalisation: J_deploy = sum_r q_r * mean(succ in r)
def Jdep(v):
    return sum(q[r]*v[regs==r].mean() for r in q)
print("check Jdep recompute vs stored:", Jdep(S["balanced"][0]["add_div1"]), J["balanced"][0]["add_div1"])

print("\n=== 1. how low was the ACTUAL div pair? rank of its mean among all C(8,2)=28 pairs ===")
for p in PROFILES:
    for s in (0,1):
        v=np.array([J[p][s][a] for a in ALL8])
        pm=np.array([0.5*(v[i]+v[j]) for i,j in itertools.combinations(range(8),2)])
        act=0.5*(J[p][s]["add_div1"]+J[p][s]["add_div2"])
        r=int((pm<act).sum())+1
        print("  %-16s s%d  div-pair mean %.3f -> rank %2d/28 (percentile %.0f%%)"%(p,s,act,r,100*r/28))

print("\n=== 2. eval-set-only null (resample the 200 per-start successes) — the WRONG null, for scale ===")
B=4000
ds=[]
for p in PROFILES:
    for s in (0,1):
        a=S[p][s]["add_div1"]; b=S[p][s]["add_div2"]
        idx=rng.integers(0,200,size=(B,200))
        ds.append(np.array([Jdep(a[i])-Jdep(b[i]) for i in idx[:400]]))
ds=np.concatenate(ds)
print("  sd of div1-div2 under START-resampling only: %.4f  (training-side sd from null pairs: %.4f)"
      %(ds.std(ddof=1),0.26633))
print("  => eval-set noise explains %.1f%% of the null-pair variance; %.1f%% is training-side"
      %(100*ds.var()/0.26633**2, 100*(1-ds.var()/0.26633**2)))
# episode-level agreement between the two null draws
ag=[ (S[p][s]["add_div1"]==S[p][s]["add_div2"]).mean() for p in PROFILES for s in (0,1)]
print("  per-episode agreement between the two identical-in-expectation draws: %s (mean %.3f)"
      %(np.round(ag,3),np.mean(ag)))

print("\n=== 3. sensitivity: permutation that ALSO breaks seed pairing (independent relabel per seed) ===")
V={p:np.array([[J[p][s][a] for s in (0,1)] for a in ALL8]) for p in PROFILES}
tcrit=stats.t.ppf(0.95,df=2)+stats.t.ppf(0.80,df=2)
def one_draw():
    nh_f=nh_r=0; dmeans=[]; ms=[]; reps=[]
    for p in PROFILES:
        pi0=rng.permutation(8); pi1=rng.permutation(8)
        col0=V[p][pi0,0]; col1=V[p][pi1,1]
        d=0.5*((col0[6]-col0[7])+(col1[6]-col1[7])); dmeans.append(d)
        div0=0.5*(col0[6]+col0[7]); div1=0.5*(col1[6]+col1[7])
        a0=col0[:6]-div0; a1=col1[:6]-div1
        m=(a0+a1)/2; r=np.sign(a0)==np.sign(a1)
        ms.append(m); reps.append(r)
    m=np.concatenate(ms); r=np.concatenate(reps)
    Xr=tcrit*np.std(dmeans,ddof=1)/np.sqrt(3)
    return int(((m>X)&r).sum()), int(((m>Xr)&r).sum()), Xr
N=200000
A=np.array([one_draw() for _ in range(N)])
print("  N=%d MC draws"%N)
print("  frozen X : mean nhits %.2f  P(>=7) = %.4f"%(A[:,0].mean(),(A[:,0]>=7).mean()))
print("  recomp X : mean nhits %.2f  P(>=7) = %.5f ; P(X<=obs)=%.5f"
      %(A[:,1].mean(),(A[:,1]>=7).mean(),(A[:,2]<=X).mean()))

print("\n=== 4. analytic check on the X fluke ===")
d=[0.00846,-0.14121,-0.33005,0.16812,0.28583,-0.37450]
sd_d=np.std(d,ddof=1); sig_pm=sd_d/np.sqrt(2)
sd_prof=np.std([np.mean(d[0:2]),np.mean(d[2:4]),np.mean(d[4:6])],ddof=1)
chi=2*(sd_prof/sig_pm)**2
print("  sd(per-seed d)=%.4f -> expected sd(profile mean d)=%.4f ; observed %.4f"%(sd_d,sig_pm,sd_prof))
print("  P(sd_profile <= observed | true sd = %.4f) = chi2_2 cdf(%.5f) = %.4f"%(sig_pm,chi,stats.chi2.cdf(chi,2)))
print("  EXPECTED X had the profile means not fluked: %.4f (=%.1fpp)  vs frozen X %.4f"%(tcrit*sig_pm/np.sqrt(3),100*tcrit*sig_pm/np.sqrt(3),X))
print("  median X from the enumeration = 0.3011 (30.1pp)")

print("\n=== 5. is there ANY arm-level signal? per-arm mean adv vs div over all 3 profiles ===")
for a in ARMS:
    cells=[J[p][s][a]-0.5*(J[p][s]["add_div1"]+J[p][s]["add_div2"]) for p in PROFILES for s in (0,1)]
    t=stats.ttest_1samp(cells,0)
    print("  %-12s cells %s  mean %+6.1fpp  t=%.2f p=%.3f"
          %(a,np.round(np.array(cells)*100,1),np.mean(cells)*100,t.statistic,t.pvalue))
print("\n  Gate-3 style: does the argmax arm differ across profiles?")
for p in PROFILES:
    mm={a:np.mean([J[p][s][a] for s in (0,1)]) for a in ARMS}
    b=max(mm,key=mm.get); print("   %-16s best=%-12s (%.3f)  spread %.3f"%(p,b,mm[b],max(mm.values())-min(mm.values())))
