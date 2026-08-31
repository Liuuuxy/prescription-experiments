"""Frozen analysis (PREREG_CONV.md): plateau epoch vs N. Conditional verdicts."""
import json, glob
import numpy as np
from scipy import stats
R=[json.load(open(fp)) for fp in glob.glob("out/*/cv_eval.json")]
if len(R)<24: raise SystemExit(f"[wait] {len(R)}/24")
Ns=[int(d["mask"].split("_N")[1].split("_")[0]) for d in R]
pl=[d["plateau_ep"] for d in R]
ok=[(n,p) for n,p in zip(Ns,pl) if p is not None]
Ns2,pl2=zip(*ok)
rho,pv=stats.spearmanr(Ns2,pl2)
print(f"n={len(ok)} runs with a defined plateau (max>0); Spearman(plateau epoch, N) = {rho:+.2f}, p = {pv:.3f}")
for N in (10,40,80,120):
    v=[p for n,p in ok if n==N]
    print(f"  N={N:3d}: plateau epochs {sorted(v)}  median {np.median(v):.0f}")
if pv<0.05 and rho>0:
    print("VERDICT: SUPPORTED — more demos need more gradient steps to plateau.")
elif pv<0.05 and rho<0:
    print("VERDICT: REVERSED — fewer demos plateau LATER (unexpected).")
else:
    print("VERDICT: NO detectable relationship at this sample; plateau epoch does not scale with N in 10-120.")
# descriptive: mean curve per N
EPS=[10,20,30,50,75,100,150,200,250]
print("\nmean success by checkpoint (rows=N):")
for N in (10,40,80,120):
    rows=[d["scores"] for d in R if f"_N{N}_" in d["name"]]
    m=[np.mean([r.get(str(e),np.nan) for r in rows])*100 for e in EPS]
    print(f"N={N:3d} "+" ".join(f"{v:5.1f}" for v in m))
