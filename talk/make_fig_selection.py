#!/usr/bin/env python3
"""Selection-level figure: which methods actually concentrate on the failure region?"""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
WK="/data/xinyua11/robocasa/weakregion"; OUT="/data/xinyua11/robocasa/talk/figs"
NAVY=(0.07,0.137,0.247); TEAL=(0.078,0.561,0.561); AMBER=(0.753,0.333,0.169); GRAY=(0.55,0.59,0.64)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":13,"figure.dpi":150})
OLD=set(["juice","spray","pitcher","canned_food","soap_dispenser","tupperware","cheese_grater","ice_cube","cream_cheese_stick","jar"])
cm=json.load(open(f"{WK}/rcless/cand_meta.json")); emap=dict(zip(cm["episodes"],cm["categories"]))
def tgt_frac(path, key="core_episodes"):
    sel=json.load(open(path))[key]; c=[emap.get(e) for e in sel]
    return 100.0*sum(1 for x in c if x in OLD)/len(sel)
rand=tgt_frac(f"{WK}/arms.json","random_episodes")
rows=[  # (label, value, family)
    ("core  (P(fail) heuristic)", 100.0, "heuristic"),
    ("failure-influence  (LESS)", tgt_frac(f"{WK}/arms_influence.json"), "gradient"),
    ("2.3  CLIP retrieval", tgt_frac(f"{WK}/arms_retrieval.json"), "visual"),
    ("2.4  DINOv2 greedy-coverage", tgt_frac(f"{WK}/arms_submod.json"), "visual"),
    ("DINOv2 contrast  (failretr)", tgt_frac(f"{WK}/arms_failretr.json"), "visual"),
    ("value-influence  (LESS)", tgt_frac(f"{WK}/arms_value.json"), "gradient"),
    ("random  (control)", rand, "control"),
]
rows.sort(key=lambda r:r[1])
col={"heuristic":TEAL,"gradient":NAVY,"visual":AMBER,"control":GRAY}
fig,ax=plt.subplots(figsize=(9.6,4.9))
y=np.arange(len(rows))
ax.barh(y,[r[1] for r in rows],color=[col[r[2]] for r in rows],height=0.64)
for i,r in enumerate(rows):
    ax.text(r[1]+1.2,i,f"{r[1]:.0f}%",va="center",fontsize=12,fontweight="bold",color=NAVY)
ax.axvline(rand,ls="--",lw=1.5,color=GRAY)
ax.text(rand+1,len(rows)-0.4,f"random / pool ≈ {rand:.0f}%",color=GRAY,fontsize=10,va="bottom")
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows],fontsize=11.5)
ax.set_xlim(0,108); ax.set_xlabel("% of the 200 selected demos that fall in the failure region")
ax.set_title("Which selectors actually concentrate on failures? (selection-level)\n"
             "visual retrieval (2.3/2.4) lands at random — scene similarity ≠ the hard object",
             fontsize=13.5,fontweight="bold",color=NAVY,loc="left")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=TEAL,label="heuristic"),Patch(color=NAVY,label="gradient influence"),
                   Patch(color=AMBER,label="visual retrieval"),Patch(color=GRAY,label="control")],
          fontsize=9.5,frameon=False,loc="lower right",ncol=2)
for s in ("top","right"): ax.spines[s].set_visible(False)
plt.tight_layout(); os.makedirs(OUT,exist_ok=True); plt.savefig(f"{OUT}/fig_selection_targeting.png",bbox_inches="tight"); plt.close()
print("wrote fig_selection_targeting.png ; values:", {r[0]:round(r[1]) for r in rows})
