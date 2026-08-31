#!/bin/bash
cd /data/xinyua11/robomimic_runs/selection
while true; do
  N=$(/data/xinyua11/conda/envs/rmimic/bin/python -c "
import json,os
p='/data/xinyua11/robomimic_runs/capability/results.json'
print(len(json.load(open(p))) if os.path.exists(p) else 0)" 2>/dev/null)
  [ "${N:-0}" -ge 40 ] && break
  sleep 180
done
echo "$(date -Is) CAP-4 drained ($N/40) — launching 5-seed DemInf race" >> orchestrate.log
/data/xinyua11/conda/envs/rmimic/bin/python - <<'PYM'
import json,h5py,numpy as np
sc=json.load(open("/data/xinyua11/robomimic_runs/selection/deminf_scores.json"))["vae"]
demos=sorted(sc,key=lambda d:-sc[d])
f=h5py.File("/data/xinyua11/robomimic_runs/selection/can_mh_sel.hdf5","r+")
for nm,dl in (("deminf_top100",demos[:100]),("deminf_bottom100",demos[-100:])):
    if nm in f["mask"]: del f["mask"][nm]
    f["mask"].create_dataset(nm,data=np.array(sorted(dl,key=lambda d:int(d.split("_")[1])),dtype="S"))
f.close()
runs=[]
for s in (1,2,3,4,5):
    for arm,mask in (("bottom","deminf_bottom100"),("top","deminf_top100"),("random","sel_random100")):
        runs.append({"name":f"sel_{arm}_s{s}","config":"rnn","mask":mask,"seed":100+s})
json.dump(runs,open("/data/xinyua11/robomimic_runs/selection/manifest.json","w"),indent=1)
print("masks + 15-run manifest ready")
PYM
CONC=6 /data/xinyua11/conda/envs/rmimic/bin/python pool_sel.py >> sel_pool.log 2>&1
echo "$(date -Is) DemInf 5-seed race done" >> orchestrate.log
