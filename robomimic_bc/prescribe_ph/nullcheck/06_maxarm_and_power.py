import json, itertools
import numpy as np
from scipy import stats
rng=np.random.default_rng(11)
P="/data/xinyua11/robomimic_runs/prescribe_ph"
PROFILES=["balanced","starved_xlo_yhi","starved_xhi_ylo"]
REGIONS=["xlo_ylo","xlo_yhi","xhi_ylo","xhi_yhi"]
ARMS=[f"add_{r}" for r in REGIONS]+["add_cov","add_pfail"]
ALL8=ARMS+["add_div1","add_div2"]
def jt(n): return json.load(open(f"{P}/out/{n}/fixed_eval_test.json"))["J_deploy"]
V={p:np.array([[jt(f"{p}_{a}_s{s}") for s in (0,1)] for a in ALL8]) for p in PROFILES}  # 8x2

# observed: per-arm mean adv across 6 cells; statistic = max over the 6 arms
def stat(assign):
    """assign: dict p -> permutation of 0..7 giving which VALUE row goes to each ARM slot.
       slots 0..5 = arms (identity preserved across profiles), 6,7 = div."""
    A=np.zeros((6,))
    for k,p in enumerate(PROFILES):
        v=V[p][assign[p]]
        div=0.5*(v[6]+v[7])
        A+= (v[:6]-div).mean(axis=1)
    return A/3.0
ident={p:np.arange(8) for p in PROFILES}
obs=stat(ident)
print("observed per-arm mean adv (over 3 profiles x 2 seeds):")
for a,x in zip(ARMS,obs): print("   %-12s %+6.2fpp"%(a,x*100))
obs_max=obs.max(); print("   max = %+.2fpp (%s)"%(obs_max*100,ARMS[int(obs.argmax())]))

N=200000
mx=np.empty(N)
for t in range(N):
    asg={p:rng.permutation(8) for p in PROFILES}
    mx[t]=stat(asg).max()
print("\npermutation null for max_arm(mean adv): P(max >= observed) = %.4f  (N=%d)"%((mx>=obs_max).mean(),N))
print("   null max distribution: median %+.2fpp  q90 %+.2fpp  q95 %+.2fpp"%tuple(100*np.percentile(mx,[50,90,95])))

print("\n=== power / how many replicates would be needed ===")
sig=0.1858   # per-run training-side sd of J_deploy
sd_contrast=np.sqrt(1.5)*sig
print("  per-run training-side sd sigma = %.4f"%sig)
print("  sd of (arm - mean of 2 div) at one profile-seed = sqrt(1.5)*sigma = %.4f (%.1fpp)"%(sd_contrast,100*sd_contrast))
print("  observed screen has n=2 cells per profile-arm -> SE = %.4f (%.1fpp); honest MDE = %.1fpp"
      %(sd_contrast/np.sqrt(2),100*sd_contrast/np.sqrt(2),100*2.486*sd_contrast/np.sqrt(2)))
print("  n=6 cells (all profiles) -> SE %.1fpp, MDE %.1fpp"%(100*sd_contrast/np.sqrt(6),100*2.486*sd_contrast/np.sqrt(6)))
for D in (0.20,0.15,0.10,0.05):
    n=(2.486*sd_contrast/D)**2
    print("  to detect a TRUE effect of %4.1fpp at 80%% power / one-sided .05: n = %.0f paired training runs per arm"%(100*D,np.ceil(n)))
print("\n  For reference the pre-registered profile-clustered estimator SHOULD have given")
print("  X = 3.981 * sd(profile-mean d)/sqrt(3) with E[sd] = %.4f -> E[X] = %.1fpp"
      %(0.2663/np.sqrt(2),100*3.9806*(0.2663/np.sqrt(2))/np.sqrt(3)))
