"""Frozen VD2 analysis (PREREG_VD2.md). Three pre-registered readouts, conditional verdicts."""
import json, glob
import numpy as np
from scipy import stats
J2={};
for fp in glob.glob("out/vd2_*/var_eval.json"):
    d=json.load(open(fp)); J2[d["name"]]=d
J1={}
for fp in glob.glob("../vardecomp/out/vd_*/var_eval.json"):
    d=json.load(open(fp)); J1[d["name"]]=d["test"]["h500"]["J_deploy"]
NS=[10,40,80,120]
missing=[f"vd2_N{N}_d{k}_s{s}" for N in NS for k in range(5) for s in (0,1,2) if f"vd2_N{N}_d{k}_s{s}" not in J2]
if missing: raise SystemExit(f"[wait] missing {len(missing)}")
print("=== 1. nested decomposition, paper recipe (best-of-ckpt, sealed exam) ===")
print(f"{'N':>4s} {'mean':>6s} {'sd_seed':>8s} {'sd_draw':>8s} {'F':>6s} {'p':>6s}")
SSB=SSW=0;dfB=dfW=0
for N in NS:
    g=np.array([[J2[f"vd2_N{N}_d{k}_s{s}"]["test"]["h500"]["J_deploy"] for s in (0,1,2)] for k in range(5)])
    dm=g.mean(1); msb=3*np.var(dm,ddof=1); msw=np.mean(np.var(g,axis=1,ddof=1))
    SSB+=msb*4; SSW+=msw*10; dfB+=4; dfW+=10
    F=msb/msw;p=1-stats.f.cdf(F,4,10)
    print(f"{N:4d} {g.mean()*100:5.1f}% {np.sqrt(msw)*100:7.1f} {np.sqrt(max(0,(msb-msw)/3))*100:7.1f} {F:6.2f} {p:6.3f}")
MSB,MSW=SSB/dfB,SSW/dfW; Fp=MSB/MSW
print(f"POOLED: sigma_seed {np.sqrt(MSW)*100:.1f}pp  sigma_draw {np.sqrt(max(0,(MSB-MSW)/3))*100:.1f}pp  F={Fp:.2f} p={1-stats.f.cdf(Fp,dfB,dfW):.4f}")
print("\n=== 2. paired recipe effect: best-of-500 minus last-of-300, same (N,draw,seed) ===")
for N in NS:
    d=[J2[f"vd2_N{N}_d{k}_s{s}"]["test"]["h500"]["J_deploy"]-J1[f"vd_N{N}_d{k}_s{s}"] for k in range(5) for s in (0,1,2)]
    dm=[np.mean([J2[f"vd2_N{N}_d{k}_s{s}"]["test"]["h500"]["J_deploy"]-J1[f"vd_N{N}_d{k}_s{s}"] for s in (0,1,2)]) for k in range(5)]
    se=np.std(dm,ddof=1)/np.sqrt(5); m=np.mean(d)
    print(f"  N={N:3d}: {m*100:+5.1f}pp  (draw-clustered t={m/se:.2f})")
alld=[np.mean([J2[f"vd2_N{N}_d{k}_s{s}"]["test"]["h500"]["J_deploy"]-J1[f"vd_N{N}_d{k}_s{s}"] for s in (0,1,2)]) for N in NS for k in range(5)]
se=np.std(alld,ddof=1)/np.sqrt(len(alld)); m=np.mean(alld)
print(f"  POOLED (20 draw-cells): {m*100:+.2f}pp, t={m/se:.2f}, p={2*(1-stats.t.cdf(abs(m/se),19)):.3f}")
print("\n=== 3. selection inflation: 50-scene selection score minus own sealed score ===")
infl=[J2[n]["sel_best"]-J2[n]["test"]["h500"]["J_deploy"] for n in J2]
print(f"  mean {np.mean(infl)*100:+.1f}pp  sd {np.std(infl,ddof=1)*100:.1f}  positive in {sum(np.array(infl)>0)}/{len(infl)}")
be=[J2[n]["best_epoch"] for n in J2]
import collections; print("  best-epoch distribution:",dict(sorted(collections.Counter(be).items())))
