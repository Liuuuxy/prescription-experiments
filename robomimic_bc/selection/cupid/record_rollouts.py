"""CUPID step 1: replay the 100 frozen E_probe starts through a trained checkpoint,
recording every executed (obs, action) pair + the rollout's success. Deterministic, so the
recorded pairs are exactly the pairs whose log-likelihood defines grad J.
Usage: record_rollouts.py CKPT OUT.npz"""
import sys, json
import numpy as np, torch
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
ck,out=sys.argv[1],sys.argv[2]
dev="cuda" if torch.cuda.is_available() else "cpu"
policy,_=FileUtils.policy_from_checkpoint(ckpt_path=ck,device=dev,verbose=False)
PH="/data/xinyua11/robomimic_runs/prescribe_ph"
env_meta=FileUtils.get_env_metadata_from_dataset(dataset_path=f"{PH}/can_ph_work.hdf5")
env=EnvUtils.create_env_from_metadata(env_meta=env_meta,render=False,render_offscreen=False,use_image_obs=False)
ev=np.load(f"{PH}/E_probe.npz")
env.reset()
OBS=[];ACT=[];EP=[];RET=[]
for i in range(len(ev["states"])):
    policy.start_episode()
    ob=env.reset_to({"states":ev["states"][i]})
    ok=False
    for t in range(400):
        a=policy(ob=ob)
        OBS.append(np.concatenate([ob[k] for k in ("robot0_eef_pos","robot0_eef_quat","robot0_gripper_qpos","object")]))
        ACT.append(a); EP.append(i)
        ob,_,_,_=env.step(a)
        if env.is_success()["task"]: ok=True; break
    RET.append(1.0 if ok else -1.0)
np.savez_compressed(out,obs=np.array(OBS,dtype=np.float32),act=np.array(ACT,dtype=np.float32),
                    ep=np.array(EP),ret=np.array(RET))
print(f"recorded {len(OBS)} pairs over {len(RET)} rollouts, success {np.mean(np.array(RET)>0)*100:.0f}% -> {out}")
