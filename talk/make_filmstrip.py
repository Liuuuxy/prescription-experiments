#!/usr/bin/env python3
"""Two-row rollout filmstrip: one failed (no-grasp) + one successful pi0 rollout."""
import glob, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

FR="/data/xinyua11/robocasa/talk/rollout_frames"; OUT="/data/xinyua11/robocasa/talk/figs"
NAVY=(0.07,0.137,0.247); TEAL=(0.078,0.561,0.561); AMBER=(0.80,0.30,0.18)
N=8
def load(pat):
    f=sorted(glob.glob(f"{FR}/{pat}"))[0]; A=np.load(f)
    if A.dtype!=np.uint8:
        A=(255*(A-A.min())/(A.ptp()+1e-6)).astype(np.uint8) if A.max()<=1.01 else A.astype(np.uint8)
    obj=os.path.basename(f).split("_",1)[1].rsplit(".",1)[0]
    return A, obj
fail,fobj=load("failure_*.npy"); succ,sobj=load("success_*.npy")
def pick(A):
    idx=np.linspace(0,len(A)-1,N).round().astype(int)
    return idx,[A[i] for i in idx]
fi,fimg=pick(fail); si,simg=pick(succ)

fig,axes=plt.subplots(2,N,figsize=(N*1.85,4.9))
plt.subplots_adjust(left=0.055,right=0.995,top=0.86,bottom=0.02,wspace=0.04,hspace=0.14)
rows=[(fimg,fi,fail,AMBER,f"FAILURE  ·  no-grasp  ·  {fobj}",
       ["approach","reach","reach","hover","hover","reach","hover","never grasped"]),
      (simg,si,succ,TEAL,f"SUCCESS  ·  {sobj}",
       ["approach","reach","grasp","lift","transport","transport","over sink","placed ✓"])]
for r,(imgs,idx,full,color,label,phases) in enumerate(rows):
    for c in range(N):
        ax=axes[r,c]; ax.imshow(imgs[c]); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color(color); sp.set_linewidth(2.4)
        ax.set_title(phases[c],fontsize=8.5,color=NAVY,pad=2)
        ax.text(0.5,-0.07,f"t={idx[c]}",transform=ax.transAxes,ha="center",va="top",
                fontsize=7.5,color=(0.4,0.43,0.48))
    # row label bar on the left
    y0=axes[r,0].get_position().y0; y1=axes[r,0].get_position().y1
    fig.text(0.012,(y0+y1)/2,label,rotation=90,va="center",ha="center",
             fontsize=11,fontweight="bold",color=color)
fig.suptitle("One rollout, start → end:  pi0 fails at the GRASP (top)  vs  completes the pick-and-place (bottom)",
             fontsize=13.5,fontweight="bold",color=NAVY,y=0.965,x=0.055,ha="left")
os.makedirs(OUT,exist_ok=True)
plt.savefig(f"{OUT}/fig_rollout_filmstrip.png",dpi=150,bbox_inches="tight"); plt.close()
print("wrote fig_rollout_filmstrip.png ; fail",fail.shape,"succ",succ.shape)
