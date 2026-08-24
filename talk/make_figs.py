#!/usr/bin/env python3
"""Figures for the revised prescribed-data talk: overall-first, per-category, height, target instability."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from collections import defaultdict

WK = "/data/xinyua11/robocasa/weakregion"
OUT = "/data/xinyua11/robocasa/talk/figs"
os.makedirs(OUT, exist_ok=True)

NAVY=(0.07,0.137,0.247); TEAL=(0.078,0.561,0.561); AMBER=(0.753,0.333,0.169)
MUTED=(0.42,0.45,0.5); LIGHT=(0.90,0.93,0.95)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":13,"axes.edgecolor":"#c4ccd2",
                     "axes.linewidth":0.9,"figure.dpi":150})

OLD=["juice","spray","pitcher","canned_food","soap_dispenser","tupperware","cheese_grater","ice_cube","cream_cheese_stick","jar"]
NEW=["cheese_grater","pitcher","juice","jar","reamer","colander","spray","jug","bottled_water","yogurt"]

def eps(a, strat=False):
    return json.load(open(f"{WK}/eval_{'strat_' if strat else ''}{a}/weakregion.json"))["episodes"]
def rate(E, cats=None, comp=False):
    S=E if cats is None else [e for e in E if (e['object_category'] in cats)!=comp]
    n=len(S); s=sum(e['success'] for e in S); return (s/n if n else float('nan')), n

# ---------- FIG 1: overall (headline) ----------
arms=["core","baseline","random","influence","coverage","value"]
lab={"core":"core (P(fail) top-10)","baseline":"baseline (no FT)","random":"random",
     "influence":"failure-influence","coverage":"coverage","value":"value-influence"}
vals={a:rate(eps(a))[0] for a in arms}
order=sorted(arms, key=lambda a:vals[a])
fig,ax=plt.subplots(figsize=(9.2,4.5))
base=vals["baseline"]
cols=[TEAL if a=="core" else (MUTED if a=="baseline" else (0.62,0.66,0.71)) for a in order]
y=np.arange(len(order))
ax.barh(y,[vals[a] for a in order],color=cols,height=0.62)
for i,a in enumerate(order):
    ax.text(vals[a]+0.004,i,f"{vals[a]:.3f}",va="center",ha="left",fontsize=13,
            fontweight="bold" if a=="core" else "normal",color=NAVY)
ax.axvline(base,ls="--",lw=1.4,color=AMBER)
ax.text(base,len(order)-0.4,f" baseline {base:.3f}",color=AMBER,fontsize=11,va="bottom")
ax.set_yticks(y); ax.set_yticklabels([lab[a] for a in order],fontsize=12)
ax.set_xlim(0.45,0.62); ax.set_xlabel("Overall success  (n=300, natural distribution)")
ax.set_title("Overall success — the trustworthy metric\ncore is the only arm above baseline",
             fontsize=15,fontweight="bold",color=NAVY,loc="left")
for s in ("top","right"): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_overall.png",bbox_inches="tight"); plt.close()

# ---------- FIG 2: per-category stratified heatmap (raw + delta) ----------
sarms=["baseline","core","random","whiten"]
pc={a:defaultdict(lambda:[0,0]) for a in sarms}
for a in sarms:
    for e in eps(a,strat=True):
        pc[a][e['object_category']][0]+=e['success']; pc[a][e['object_category']][1]+=1
def r(a,c):
    k=pc[a][c]; return k[0]/k[1] if k[1] else np.nan
cats=sorted(OLD, key=lambda c:r("baseline",c))  # hardest (lowest baseline) first
raw=np.array([[r(a,c) for a in sarms] for c in cats])
delta=np.array([[r(a,c)-r("baseline",c) for a in ["core","random","whiten"]] for c in cats])

fig,(axA,axB)=plt.subplots(1,2,figsize=(12.6,5.4),gridspec_kw={"width_ratios":[4,3]})
im=axA.imshow(raw,cmap="RdYlGn",vmin=0,vmax=1,aspect="auto")
axA.set_xticks(range(len(sarms)));axA.set_xticklabels(["baseline","core","random","whiten"],fontsize=12)
axA.set_yticks(range(len(cats)));axA.set_yticklabels(cats,fontsize=11)
for i in range(len(cats)):
    for j in range(len(sarms)):
        axA.text(j,i,f"{raw[i,j]:.2f}",ha="center",va="center",fontsize=10,
                 color="black")
axA.set_title("Per-category success (stratified, per_cat≈16–39)\nhardest categories at top",
              fontsize=13,fontweight="bold",color=NAVY,loc="left")
nrm=TwoSlopeNorm(vmin=-0.35,vcenter=0,vmax=0.35)
imB=axB.imshow(delta,cmap="RdBu",norm=nrm,aspect="auto")
axB.set_xticks(range(3));axB.set_xticklabels(["core","random","whiten"],fontsize=12)
axB.set_yticks(range(len(cats)));axB.set_yticklabels([])
for i in range(len(cats)):
    for j in range(3):
        axB.text(j,i,f"{delta[i,j]:+.2f}",ha="center",va="center",fontsize=10,color="black")
axB.set_title("Δ vs baseline\n(green = helped, red = hurt)",fontsize=13,fontweight="bold",color=NAVY,loc="left")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_percat_heatmap.png",bbox_inches="tight"); plt.close()

# ---------- FIG 3: height tertiles (the real weak axis) ----------
harms=["baseline","core","random","coverage","influence","value"]
allh=sorted(e['obj_height'] for e in eps('baseline') if e.get('obj_height') is not None)
q1,q2=allh[len(allh)//3],allh[2*len(allh)//3]
def hbin(h): return 0 if h<q1 else (2 if h>q2 else 1)
H={}
for a in harms:
    b=[[0,0],[0,0],[0,0]]
    for e in eps(a):
        h=e.get('obj_height')
        if h is None: continue
        k=hbin(h); b[k][0]+=e['success']; b[k][1]+=1
    H[a]=[b[k][0]/b[k][1] if b[k][1] else np.nan for k in range(3)]
fig,ax=plt.subplots(figsize=(9.2,4.9))
x=np.arange(3)
for a in harms:
    if a=="baseline":
        ax.plot(x,H[a],"-o",lw=3.2,color=NAVY,label="baseline",zorder=5,ms=8)
    elif a=="core":
        ax.plot(x,H[a],"-o",lw=2.6,color=TEAL,label="core",zorder=4,ms=7)
    else:
        ax.plot(x,H[a],"-o",lw=1.6,color=(0.6,0.64,0.69),label=a,ms=5,alpha=0.9)
ax.set_xticks(x);ax.set_xticklabels([f"short\n(<{q1*100:.0f}cm)","medium",f"tall\n(>{q2*100:.0f}cm)"],fontsize=12)
ax.set_ylabel("success"); ax.set_ylim(0.28,0.75)
ax.set_title("By the real weak axis (object height), every intervention REGRESSED the tall objects",
             fontsize=13.5,fontweight="bold",color=NAVY,loc="left")
ax.annotate("tall = the actual weak region;\nall arms fall below baseline here",
            xy=(2,H["baseline"][2]),xytext=(1.15,0.66),fontsize=11,color=AMBER,
            arrowprops=dict(arrowstyle="->",color=AMBER,lw=1.4))
ax.legend(ncol=3,fontsize=10,frameon=False,loc="lower left")
for s in ("top","right"): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_height.png",bbox_inches="tight"); plt.close()

# ---------- FIG 4: targeted-region verdict is MEASUREMENT-dependent ----------
# four ways to measure success on "the targeted region" -> the verdict jumps around;
# the cleanest (rightmost: balanced-select + balanced-measure) has all arms tied.
def strat_overall(a): return rate(eps(a, strat=True))[0]
def balcat(a): return rate(json.load(open(f"{WK}/eval_balcat_{a}/weakregion.json"))["episodes"])[0]
groups = [
    ("n=300 slice\nOLD-10 cats",
     {"baseline": rate(eps("baseline"), set(OLD))[0], "core": rate(eps("core"), set(OLD))[0],
      "random": rate(eps("random"), set(OLD))[0]}),
    ("n=300 slice\nBALANCED-10 cats",
     {"baseline": rate(eps("baseline"), set(NEW))[0], "core": rate(eps("core"), set(NEW))[0],
      "random": rate(eps("random"), set(NEW))[0]}),
    ("stratified\nOLD-10 (reliable)",
     {"baseline": strat_overall("baseline"), "core": strat_overall("core"),
      "random": strat_overall("random")}),
    ("stratified · BALANCED-10\n(balanced select + measure)",
     {"baseline": balcat("baseline"), "core": balcat("core"), "random": balcat("random")}),
]
barms = ["baseline", "core", "random"]
bcol = {"baseline": (0.55, 0.59, 0.64), "core": TEAL, "random": AMBER}
fig, ax = plt.subplots(figsize=(12.2, 4.9))
x = np.arange(len(groups)); w = 0.26
for j, arm in enumerate(barms):
    vals = [g[1][arm] for g in groups]
    ax.bar(x + (j - 1) * w, vals, w, label=arm, color=bcol[arm])
    for i, v in enumerate(vals):
        ax.text(x[i] + (j - 1) * w, v + 0.006, f"{v:.2f}", ha="center", fontsize=9.5, color=NAVY)
ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups], fontsize=10.5)
ax.set_ylabel("targeted-region success"); ax.set_ylim(0, 0.52)
# bracket the cleanest group
ax.annotate("do it all 'right' →\nall arms TIE (0.33–0.35)", xy=(3, 0.352), xytext=(2.35, 0.47),
            fontsize=10.5, color=NAVY, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.4))
ax.set_title("The 'targeted-region' verdict swings with how you measure it — "
             "and vanishes when you do it right\n"
             "core − baseline:  +0.10 → −0.01 → +0.11 → +0.02      core − random:  +0.15 → +0.20 → +0.02 → +0.02",
             fontsize=12, fontweight="bold", color=NAVY, loc="left")
ax.legend(fontsize=10.5, frameon=False, ncol=3, loc="upper left")
for s in ("top", "right"): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_target_instability.png", bbox_inches="tight"); plt.close()

print("wrote:", os.listdir(OUT))
