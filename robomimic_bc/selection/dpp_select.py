"""Coverage baseline: greedy MAP k-DPP over demo embeddings (mean/std/first/last state + length)."""
import h5py, json
import numpy as np
f=h5py.File("/data/xinyua11/robomimic_runs/selection/can_mh_sel.hdf5","r")
demos=sorted(f["data"].keys(),key=lambda d:int(d.split("_")[1]))
KEYS=["robot0_eef_pos","robot0_eef_quat","robot0_gripper_qpos","object"]
E=[]
for d in demos:
    s=np.concatenate([f["data"][d]["obs"][k][:] for k in KEYS],axis=1)
    E.append(np.concatenate([s.mean(0),s.std(0),s[0],s[-1],[len(s)/300]]))
f.close()
E=np.array(E); E=(E-E.mean(0))/(E.std(0)+1e-9)
d2=((E[:,None,:]-E[None,:,:])**2).sum(-1); K=np.exp(-d2/(2*np.median(d2)))
sel=[int(np.argmax(K.sum(1)))]
for _ in range(99):
    cand=[i for i in range(len(demos)) if i not in sel]
    # greedy MAP: maximize log det of K[sel+i]
    best,bv=None,-1e18
    Ks=K[np.ix_(sel,sel)]+1e-6*np.eye(len(sel)); L=np.linalg.cholesky(Ks)
    for i in cand:
        v=K[np.ix_(sel,[i])]; w=np.linalg.solve(L,v)
        gain=np.log(max(K[i,i]+1e-6-float(w.T@w),1e-12))
        if gain>bv: bv,best=gain,i
    sel.append(best)
picks=sorted([demos[i] for i in sel],key=lambda d:int(d.split("_")[1]))
json.dump(picks,open("dpp_top100.json","w"))
f=h5py.File("/data/xinyua11/robomimic_runs/selection/can_mh_sel.hdf5","r+")
if "dpp100" in f["mask"]: del f["mask"]["dpp100"]
f["mask"].create_dataset("dpp100",data=np.array(picks,dtype="S")); f.close()
print("dpp100 mask written,",len(picks),"demos")
