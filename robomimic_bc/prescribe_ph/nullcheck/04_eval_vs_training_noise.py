import json
import numpy as np
rng=np.random.default_rng(7)
P="/data/xinyua11/robomimic_runs/prescribe_ph"
PROFILES=["balanced","starved_xlo_yhi","starved_xhi_ylo"]
q=json.load(open(f"{P}/deploy_shares.json"))["natural_reset_shares"]
regs=np.array([str(r) for r in np.load(f"{P}/E_test.npz",allow_pickle=True)["region"]])
MASK={r:(regs==r) for r in q}
def Jdep(v): return sum(q[r]*v[MASK[r]].mean() for r in q)
def sc(n): return np.array(json.load(open(f"{P}/out/{n}/fixed_eval_test.json"))["successes"])
B=20000
print("PAIRED start-resampling (eval-set noise ONLY) of d = J(div1)-J(div2), per profile-seed:")
vs=[]
for p in PROFILES:
    for s in (0,1):
        a=sc(f"{p}_add_div1_s{s}"); b=sc(f"{p}_add_div2_s{s}")
        idx=rng.integers(0,200,size=(B,200))
        dd=np.array([Jdep(a[i])-Jdep(b[i]) for i in idx])
        vs.append(dd.var(ddof=1))
        print("  %-16s s%d  observed d %+0.4f   bootstrap sd %.4f   agreement %.3f"
              %(p,s,Jdep(a)-Jdep(b),dd.std(ddof=1),(a==b).mean()))
ev=np.mean(vs); tot=0.26633262702242605**2
print("\n  mean eval-only variance of d      = %.5f  (sd %.4f)"%(ev,np.sqrt(ev)))
print("  total observed variance of d      = %.5f  (sd %.4f)"%(tot,np.sqrt(tot)))
print("  => eval-set (start-sampling) share = %.1f%% ; TRAINING-side share = %.1f%% (sd %.4f)"
      %(100*ev/tot,100*(1-ev/tot),np.sqrt(max(tot-ev,0))))
print("  per-run training-side sd of J_deploy = %.4f"%(np.sqrt(max(tot-ev,0)/2)))
print("\n  If one instead built the MDE from start-resampling alone, X would be about")
print("  1.645*sd_eval*? ... eval-only sd of a single paired contrast = %.4f -> naive MDE ~ %.1fpp"
      %(np.sqrt(ev),100*2.486*np.sqrt(ev)))
