"""Frozen analysis for the nested variance decomposition (PREREG_VARDECOMP.md).
Primary: per-N and pooled nested ANOVA -> sigma_draw vs sigma_seed with df and F-tests.
Secondary (exploratory, labelled): variance heterogeneity, composition correlations,
horizon-400 vs 500 comparability. All verdicts conditional."""
import json, glob
import numpy as np
from scipy import stats
J={}
for fp in glob.glob("out/vd_*/var_eval.json"):
    d=json.load(open(fp)); J[d["name"]]=d
NS=[10,40,80,120]
def grid(N,h="h500"):
    return np.array([[J[f"vd_N{N}_d{k}_s{s}"]["test"][h]["J_deploy"] for s in (0,1,2)] for k in range(5)])
missing=[f"vd_N{N}_d{k}_s{s}" for N in NS for k in range(5) for s in (0,1,2) if f"vd_N{N}_d{k}_s{s}" not in J]
if missing: raise SystemExit(f"[wait] {len(missing)} runs missing, e.g. {missing[:3]}")
print(f"{'N':>4s} {'mean':>6s} | {'sd_seed':>8s} {'sd_draw':>8s} | {'F(4,10)':>8s} {'p':>6s}")
SSB=SSW=0; dfB=dfW=0
for N in NS:
    g=grid(N); dm=g.mean(1); gm=g.mean()
    msb=3*np.var(dm,ddof=1); msw=np.mean(np.var(g,axis=1,ddof=1))
    ssb=msb*4; ssw=msw*10
    SSB+=ssb; SSW+=ssw; dfB+=4; dfW+=10
    s_seed=np.sqrt(msw); s_draw=np.sqrt(max(0,(msb-msw)/3))
    F=msb/msw; p=1-stats.f.cdf(F,4,10)
    print(f"{N:4d} {gm*100:5.1f}% | {s_seed*100:7.1f}pp {s_draw*100:7.1f}pp | {F:8.2f} {p:6.3f}")
MSB,MSW=SSB/dfB,SSW/dfW
Fp=MSB/MSW; pp_=1-stats.f.cdf(Fp,dfB,dfW)
s_seed=np.sqrt(MSW); s_draw=np.sqrt(max(0,(MSB-MSW)/3))
print(f"\nPOOLED ({dfB},{dfW} df): sigma_seed = {s_seed*100:.1f}pp, sigma_draw = {s_draw*100:.1f}pp, F = {Fp:.2f}, p = {pp_:.4f}")
if pp_<0.05 and s_draw>0:
    print(f"PRIMARY VERDICT: WHICH demos you draw has a detectable effect beyond the seed lottery "
          f"(draw component {s_draw*100:.1f}pp vs seed {s_seed*100:.1f}pp).")
else:
    bound=np.sqrt(max(0,(MSW*stats.f.ppf(0.95,dfB,dfW)-MSW)/3))
    print(f"PRIMARY VERDICT: NO detectable draw effect; upper bound sigma_draw <~ {bound*100:.1f}pp "
          f"(per the prereg, reported as a bound, not a null).")
# secondary
comp=json.load(open("draw_composition.json"))
resid=[]; ent=[]; ln=[]
for N in NS:
    g=grid(N); dm=g.mean(1)
    for k in range(5):
        resid.append(dm[k]-dm.mean())
        z=comp[f"vd_N{N}_d{k}"]["zones"]; p4=np.array([z.get(r,0) for r in ("xlo_ylo","xlo_yhi","xhi_ylo","xhi_yhi")],dtype=float)
        p4=p4/p4.sum(); ent.append(-(p4[p4>0]*np.log(p4[p4>0])).sum())
        ln.append(comp[f"vd_N{N}_d{k}"]["mean_len"])
for nm,x in (("zone-balance entropy",ent),("mean demo length",ln)):
    r,pv=stats.pearsonr(x,resid)
    print(f"secondary: corr(draw mean resid, {nm}) = {r:+.2f} (p={pv:.2f}, n=20) [exploratory]")
d45=[J[n]["test"]["h500"]["J_deploy"]-J[n]["test"]["h400"]["J_deploy"] for n in J if n.startswith("vd_")]
print(f"secondary: horizon 500 vs 400 adds {np.mean(d45)*100:+.2f}pp on average (max {max(d45)*100:+.1f}) — "
      f"{'conclusions unchanged' if max(d45)<0.05 else 'CHECK: horizon materially matters'}")
