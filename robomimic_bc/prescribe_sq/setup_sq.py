"""Square-PH heterogeneity benchmark, phase-1 setup.
Regions: 2x2 grid over (nut initial y) x (nut initial yaw), dataset medians.
x is NOT used (5mm spread = placement jitter). Pilot D0 sizes 60/80/96
(96 = max with 24-demo wells: min region count 48 - 24 = 24)."""
import h5py, json, shutil, zlib
import numpy as np
SRC="/data/xinyua11/robomimic_datasets/square/ph/low_dim_v141.hdf5"
WORK="/data/xinyua11/robomimic_runs/prescribe_sq/square_ph_work.hdf5"
REG=["ylo_yawlo","ylo_yawhi","yhi_yawlo","yhi_yawhi"]
def rng_for(tag): return np.random.RandomState(zlib.crc32(tag.encode())%(2**31))
shutil.copy(SRC,WORK)
f=h5py.File(WORK,"r+")
demos=sorted(f["data"].keys(),key=lambda d:int(d.split("_")[1]))
O=np.array([f["data"][d]["obs"]["object"][0] for d in demos])
ypos=O[:,1]; q=O[:,3:7]
yaw=np.arctan2(2*(q[:,3]*q[:,2]+q[:,0]*q[:,1]),1-2*(q[:,1]**2+q[:,2]**2))
qy,qyaw=float(np.median(ypos)),float(np.median(yaw))
lab=(ypos>qy).astype(int)*2+(yaw>qyaw).astype(int)
region_of={d:REG[l] for d,l in zip(demos,lab)}
if "mask" in f: del f["mask"]
g=f.create_group("mask")
def wm(name,dl): g.create_dataset(name,data=np.array(sorted(dl,key=lambda d:int(d.split("_")[1])),dtype="S"))
by={r:[d for d in demos if region_of[d]==r] for r in REG}
for r in REG: wm(f"region_{r}",by[r])
for n in (60,80,96):
    picks=[]
    for r in REG:
        rng=rng_for(f"sq_pilot_D0_{n}_{r}")
        picks+=list(rng.choice(by[r],n//4,replace=False))
    wm(f"pilot_D0_{n}",picks)
meta={"grid":{"qy":qy,"qyaw":qyaw,"labels":REG,"rule":"2*(y>qy)+(yaw>qyaw); y=object[1], yaw from object[3:7] xyzw quat"},
      "region_counts":{r:len(by[r]) for r in REG},"region_of":region_of,
      "lengths":{d:int(f["data"][d].attrs["num_samples"]) for d in demos}}
f.close()
json.dump(meta,open("/data/xinyua11/robomimic_runs/prescribe_sq/regions.json","w"),indent=1)
print("grid qy=%.4f qyaw=%.3f counts=%s"%(qy,qyaw,meta["region_counts"]))
