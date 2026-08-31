"""PROBE ONLY: build a small merged Can+Square hdf5 and verify robomimic 0.3.0 can load it.
Writes /data/xinyua11/robomimic_runs/xtask/merge_probe.hdf5 (20 Can + 20 Square demos)."""
import h5py, json, numpy as np, os
SRC = {"can":"/data/xinyua11/robomimic_datasets/can/ph/low_dim_v141.hdf5",
       "square":"/data/xinyua11/robomimic_datasets/square/ph/low_dim_v141.hdf5"}
OUT = "/data/xinyua11/robomimic_runs/xtask/merge_probe.hdf5"
NPER = 20
TASKID = {"can":0, "square":1}
if os.path.exists(OUT): os.remove(OUT)
out = h5py.File(OUT,"w"); g = out.create_group("data")
total = 0; gi = 0; provenance = {}; by_task = {t:[] for t in SRC}
for t, p in SRC.items():
    f = h5py.File(p,"r")
    demos = sorted(f["data"].keys(), key=lambda x:int(x.split("_")[1]))[:NPER]
    for d in demos:
        src = f["data"][d]; name = f"demo_{gi}"
        dst = g.create_group(name)
        T = src.attrs["num_samples"]
        for k in ("actions","dones","rewards","states"):
            dst.create_dataset(k, data=src[k][:])
        for grp in ("obs","next_obs"):
            og = dst.create_group(grp)
            for k in src[grp].keys():
                og.create_dataset(k, data=src[grp][k][:])
            oh = np.zeros((T,len(SRC)),dtype=np.float32); oh[:,TASKID[t]] = 1.0
            og.create_dataset("task_id", data=oh)
        dst.attrs["num_samples"] = T
        dst.attrs["task"] = t
        provenance[name] = {"task":t, "src_demo":d, "T":int(T)}
        by_task[t].append(name); total += int(T); gi += 1
    # env_args of the FIRST source becomes the file-level one (eval never uses it)
    if t == "can": g.attrs["env_args"] = f["data"].attrs["env_args"]
    f.close()
g.attrs["total"] = total
mg = out.create_group("mask")
key = lambda ds: np.array(sorted(ds, key=lambda x:int(x.split("_")[1])), dtype="S")
for t in SRC: mg.create_dataset(f"task_{t}", data=key(by_task[t]))
mg.create_dataset("all", data=key(list(provenance)))
mg.create_dataset("can_only", data=key(by_task["can"]))
out.close()
json.dump(provenance, open("/data/xinyua11/robomimic_runs/xtask/merge_probe_provenance.json","w"), indent=1)
print("wrote", OUT, "demos:", gi, "steps:", total)
