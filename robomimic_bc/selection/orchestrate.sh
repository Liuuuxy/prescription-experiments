#!/bin/bash
cd /data/xinyua11/robomimic_runs/selection
# drain = all 50 variance-manifest runs present in results.json
while true; do
  N=$(/data/xinyua11/conda/envs/rmimic/bin/python -c "
import json
r=json.load(open('/data/xinyua11/robomimic_runs/variance/results.json'))
m=json.load(open('/data/xinyua11/robomimic_runs/variance/manifest.json'))
print(sum(1 for x in m if x['name'] in r))" 2>/dev/null)
  [ "$N" = "50" ] && break
  sleep 120
done
echo "$(date -Is) variance drained (50/50)" >> orchestrate.log
GATE=$(/data/xinyua11/conda/envs/rmimic/bin/python - <<'PYG'
import json,numpy as np
r=json.load(open("/data/xinyua11/robomimic_runs/variance/results.json"))
best=None
for mask in ("var_D0_20","var_D0_32","var_D0_44"):
    v=[x["test"]["J_deploy"] for x in r.values() if x["config"]=="rnn" and x["mask"]==mask]
    if len(v)>=5:
        m,s=np.mean(v),np.std(v,ddof=1)
        if 0.25<=m<=0.85 and (best is None or s<best[1]): best=(mask,round(s,4),round(m,3),len(v))
print(("PASS" if best and best[1]<=0.04 else "FAIL"), best)
PYG
)
echo "$(date -Is) learner gate: $GATE" >> orchestrate.log
if echo "$GATE" | grep -q PASS; then
  /data/xinyua11/conda/envs/rmimic/bin/python - <<'PYM'
import json,h5py,numpy as np
sc=json.load(open("/data/xinyua11/robomimic_runs/selection/deminf_scores.json"))["vae"]
demos=sorted(sc,key=lambda d:-sc[d])
f=h5py.File("/data/xinyua11/robomimic_runs/selection/can_mh_sel.hdf5","r+")
for nm,dl in (("deminf_top100",demos[:100]),("deminf_bottom100",demos[-100:])):
    if nm in f["mask"]: del f["mask"][nm]
    f["mask"].create_dataset(nm,data=np.array(sorted(dl,key=lambda d:int(d.split("_")[1])),dtype="S"))
f.close(); print("masks ok")
PYM
  /data/xinyua11/conda/envs/rmimic/bin/python - <<'PYN'
import json
runs=[]
for s in (1,2,3):
    for arm,mask in (("bottom","deminf_bottom100"),("top","deminf_top100"),("random","sel_random100")):
        runs.append({"name":f"sel_{arm}_s{s}","config":"rnn","mask":mask,"seed":100+s})
json.dump(runs,open("/data/xinyua11/robomimic_runs/selection/manifest.json","w"),indent=1)
PYN
  CONC=6 /data/xinyua11/conda/envs/rmimic/bin/python pool_sel.py >> sel_pool.log 2>&1
  echo "$(date -Is) DemInf race done" >> orchestrate.log
else
  echo "$(date -Is) gate FAILED — race deferred" >> orchestrate.log
fi
cd /data/xinyua11/robomimic_runs/capability && /data/xinyua11/conda/envs/rmimic/bin/python pool_cap.py >> cap_pool.log 2>&1
echo "$(date -Is) CAP-4 done" >> orchestrate.log
