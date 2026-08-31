import csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

G="/data/xinyua11/robocasa/cifar_experiments/robot_mirrors"
rows=list(csv.reader(open(f"{G}/cifar_cluster_class_table.csv")))
classes=[c.rstrip("*") for c in rows[0][1:]]
rare=[c.endswith("*") for c in rows[0][1:]]
mat={}; accv=None
for r in rows[1:]:
    if r[0]=="accuracy": accv=np.array([float(x)*100 for x in r[1:]])
    elif r[0]!="pool_total": mat[r[0]]=np.array([int(x) for x in r[1:]])
clusters=sorted(mat)

R=json.load(open("/data/xinyua11/xgradtest/armrace/cluster_results.json"))
import collections
race=collections.defaultdict(list)
for v in R.values(): race[v["arm"]].append(v)
res={}
for a,vs in race.items():
    res[a.replace("cluster","c")]={
        "dr":np.mean([v["delta_rare"] for v in vs])*100,
        "do":np.mean([v["delta_overall"] for v in vs])*100,
        "rf":np.mean([v["draw_rare_frac"] for v in vs])*100}


Rmain=json.load(open("/data/xinyua11/xgradtest/armrace/results.json"))
gm=collections.defaultdict(list)
for k,v in Rmain.items():
    if k!="base_ref": gm[v["arm"]].append(v)
for arm,cn in (("gradarm_a","c1"),("gradarm_b","c5")):
    vs=gm[arm]
    res[cn]={"dr":np.mean([v["delta_rare"] for v in vs])*100,
             "do":np.mean([v["delta_overall"] for v in vs])*100,
             "rf":np.mean([v["draw_rare_frac"] for v in vs])*100,
             "alias":arm}

fig=plt.figure(figsize=(24,7.5))
gs=GridSpec(2,2,width_ratios=[10,1.45],height_ratios=[5,1],hspace=0.02,wspace=0.01)
ax=fig.add_subplot(gs[0,0])
M=np.array([mat[c] for c in clusters])
im=ax.imshow(M,aspect="auto",cmap="Blues",vmax=130)
ax.set_yticks(range(len(clusters)))
ax.set_yticklabels([f"{c} (n={mat[c].sum()})" for c in clusters],fontsize=10)
ax.set_xticks([])
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if M[i,j]>=45:
            ax.text(j,i,str(M[i,j]),ha="center",va="center",fontsize=6,
                    color="white" if M[i,j]>90 else "#1a3b5e")
ax.set_title("CIFAR: demos per (gradient cluster x class) - orange names = rare/weak classes; bottom strip = per-class test accuracy; right panel = outcome when 200 demos from that cluster were used (mean of 4 fine-tune rounds)",fontsize=11,loc="left")

# accuracy strip
ax2=fig.add_subplot(gs[1,0])
ax2.imshow(accv[None,:],aspect="auto",cmap="RdYlGn",vmin=0,vmax=90)
ax2.set_yticks([0]); ax2.set_yticklabels(["accuracy"],fontsize=9)
for j,a in enumerate(accv):
    if np.isfinite(a): ax2.text(j,0,f"{a:.0f}",ha="center",va="center",fontsize=5.2)
ax2.set_xticks(range(len(classes)))
ax2.set_xticklabels(classes,rotation=90,fontsize=6.5)
for lbl,isr in zip(ax2.get_xticklabels(),rare):
    if isr: lbl.set_color("#d15c10"); lbl.set_fontweight("bold")

# RIGHT PANEL: race outcomes per cluster
ax3=fig.add_subplot(gs[0,1]); ax3.set_xlim(0,1); ax3.set_ylim(len(clusters)-0.5,-0.5)
ax3.axis("off")
BASE_RARE,BASE_ALL=14.55,37.30
ax3.text(0.02,-0.62,"when raced (delta, pp):",fontsize=9,fontweight="bold")
for i,c in enumerate(clusters):
    if c in res:
        r=res[c]
        w=max(min(r["dr"]/4.0,1.0),0.02)
        ax3.barh(i-0.18,w*0.9,left=0.02,height=0.3,color="#2a78d6",alpha=0.85)
        ax3.text(0.02,i+0.13,"rare %+.2f   overall %+.2f"%(r["dr"],r["do"]),fontsize=8.2,va="center")
        if "alias" in r: ax3.text(0.72,i-0.30,"raced as "+r["alias"],fontsize=6.6,color="#2a78d6",va="center")
        ax3.text(0.02,i+0.36,"drew %.0f%% rare-class images"%r["rf"],fontsize=7,va="center",color="#666")
    else:
        pass
note="before: rare 14.55%, overall 37.3%\nreference arms: pure-rare +3.79, random +0.88, null -0.47\nc1/c5 = the two most-separated clusters, raced as gradarm_a/b\nrare-delta tracks %-rare-drawn -> clusters add nothing\nbeyond their class composition"
fig.text(0.845,0.085,note,fontsize=7.6,va="top")
plt.savefig("cifar_cluster_class_race.png",dpi=140,bbox_inches="tight")
print("saved")
