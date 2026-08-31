"""Frozen eval sets for Square-PH: E_probe 25/region + sealed E_test 50/region from
fresh resets, stratified by the (y, yaw) grid in regions.json; q from 2000 resets.
Validates exact reset_to restore + bitwise-deterministic replay first."""
import json, os
import numpy as np
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
P="/data/xinyua11/robomimic_runs/prescribe_sq"
meta=json.load(open(f"{P}/regions.json"))
QY,QYAW,REG=meta["grid"]["qy"],meta["grid"]["qyaw"],meta["grid"]["labels"]
PER=75
ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs={"obs":{"low_dim":[
 "robot0_eef_pos","robot0_eef_quat","robot0_gripper_qpos","object"],"rgb":[]}})
env_meta=FileUtils.get_env_metadata_from_dataset(dataset_path=f"{P}/square_ph_work.hdf5")
env=EnvUtils.create_env_from_metadata(env_meta=env_meta,render=False,render_offscreen=False,use_image_obs=False)
adim=env.action_dimension
def region_of_obs(o):
    y=float(o["object"][1]); q=o["object"][3:7]
    yaw=float(np.arctan2(2*(q[3]*q[2]+q[0]*q[1]),1-2*(q[1]**2+q[2]**2)))
    return REG[2*int(y>QY)+int(yaw>QYAW)]
np.random.seed(999)
o0=env.reset(); st0=env.get_state()
o1=env.reset_to(st0)
assert np.allclose(o0["object"][:7],o1["object"][:7],atol=1e-7)
assert np.allclose(env.get_state()["states"],st0["states"],atol=1e-7)
def fp():
    env.reset_to(st0); out=[]
    for t in range(20):
        o,_,_,_=env.step(np.zeros(adim)); out.append(np.concatenate([o["object"],o["robot0_eef_pos"]]))
    return np.array(out)
assert np.abs(fp()-fp()).max()==0.0
print("VALIDATION PASS: exact restore + bitwise-deterministic replay",flush=True)
buckets={r:[] for r in REG}; shares=[]; i=0
while min(len(b) for b in buckets.values())<PER or i<2000:
    np.random.seed(200000+i)
    o=env.reset(); i+=1
    r=region_of_obs(o); shares.append(r)
    if len(buckets[r])<PER: buckets[r].append(env.get_state()["states"].copy())
from collections import Counter
c=Counter(shares); q={r:c[r]/i for r in REG}
print(f"{i} resets; q = "+" ".join(f"{r}:{q[r]:.3f}" for r in REG),flush=True)
def save(fn,lo,hi):
    st=np.stack([buckets[r][k] for r in REG for k in range(lo,hi)])
    lb=np.array([r for r in REG for _ in range(lo,hi)],dtype="U16")
    np.savez_compressed(f"{P}/{fn}",states=st,region=lb,per_region=hi-lo)
    print("saved",fn,st.shape,flush=True)
save("E_probe.npz",0,25); save("E_test.npz",25,75)
json.dump({"natural_reset_shares":q,"n_resets":i},open(f"{P}/deploy_shares.json","w"),indent=1)
