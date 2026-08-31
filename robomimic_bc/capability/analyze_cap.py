"""CAP-4 frozen analysis. Run AFTER all 40 runs complete. Do not edit after hashing."""
import json,itertools,numpy as np
P="/data/xinyua11/robomimic_runs/capability"
R=json.load(open(f"{P}/results.json"))
REGS=["xlo_ylo","xhi_ylo","xlo_yhi","xhi_yhi"]
def cown(Jr,own):
    return 100*(Jr[own]-np.mean([Jr[r] for r in REGS if r!=own]))
rows=[]   # (cfg, own_region, seed, c_own)  from the 32 region-only runs
plac=[]   # (cfg, c_placebo) : same contrast centred on a NON-own region
for k,v in R.items():
    if not v["mask"].startswith("region_"): continue
    own=v["mask"][len("region_"):]; Jr=v["test"]["J_region"]
    rows.append((v["config"],own,v["seed"],cown(Jr,own)))
    for r in REGS:
        if r!=own: plac.append((v["config"],cown(Jr,r)))
import collections
cell=collections.defaultdict(list)
for cfg,own,s,c in rows: cell[(cfg,own)].append(c)
# pooled within-cell variance: 8 cells x 3 df = 24 df
ss=sum(sum((np.array(v)-np.mean(v))**2) for v in cell.values()); df=sum(len(v)-1 for v in cell.values())
s2=ss/df; se_cell=np.sqrt(s2/4)
print(f"pooled within-cell sd = {np.sqrt(s2):.2f} pp on {df} df")
for cfg in ["mlp","rnn"]:
    m=np.mean([np.mean(cell[(cfg,r)]) for r in REGS])
    se=np.sqrt(s2/4/4)
    print(f"c_own[{cfg}] = {m:+.2f} pp  SE {se:.2f}  t_vs_0 {m/se:.2f}  (per-region "
          +", ".join(f"{r}:{np.mean(cell[(cfg,r)]):+.1f}" for r in REGS)+")")
mM=np.mean([np.mean(cell[("mlp",r)]) for r in REGS]); mR=np.mean([np.mean(cell[("rnn",r)]) for r in REGS])
seD=np.sqrt(2*s2/4/4); D=mM-mR
print(f"TEST 1  Delta = c_own[mlp]-c_own[rnn] = {D:+.2f} pp  SE {seD:.2f}  t {D/seD:.2f}  df {df}")
print(f"TEST 2  c_own[rnn] vs 0 : {mR:+.2f}  SE {np.sqrt(s2/16):.2f}  t {mR/np.sqrt(s2/16):.2f}")
for cfg in ["mlp","rnn"]:
    pv=[c for g,c in plac if g==cfg]
    print(f"PLACEBO[{cfg}] n={len(pv)} mean {np.mean(pv):+.2f} sd {np.std(pv,ddof=1):.2f} "
          f"|max| {max(abs(np.array(pv))):.1f}")
for k,v in sorted(R.items()):
    if v["mask"]=="pilot_D0_40": print("N40 anchor",k,round(v["test"]["J_deploy"],3))
