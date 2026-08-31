import glob, json, shutil, sys, os
import numpy as np, torch
from robomimic.config import config_factory
import robomimic.scripts.train as T
P="/data/xinyua11/robomimic_runs/convergence"
PH="/data/xinyua11/robomimic_runs/prescribe_ph"
name,cfg,mask,seed=sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4])
EP=250; HORIZON=500; CKS=[10,20,30,50,75,100,150,200,250]
shutil.rmtree(f"{P}/out/{name}",ignore_errors=True)
c=config_factory(algo_name="bc")
c.experiment.name=name; c.experiment.validate=False
c.experiment.logging.terminal_output_to_txt=False
c.experiment.save.enabled=True; c.experiment.save.every_n_epochs=5
c.experiment.rollout.enabled=False
c.train.data=f"{PH}/can_ph_work.hdf5"; c.train.hdf5_filter_key=mask
c.train.output_dir=f"{P}/out"; c.train.num_epochs=EP; c.train.seed=seed
c.train.batch_size=100; c.train.hdf5_cache_mode="all"
c.observation.modalities.obs.low_dim=["robot0_eef_pos","robot0_eef_quat","robot0_gripper_qpos","object"]
c.observation.modalities.obs.rgb=[]
c.lock()
from robomimic.utils.dataset import SequenceDataset
def _eqw(self):
    w=[1.0/self._demo_id_to_demo_length[self._index_to_demo_id[i]] for i in range(len(self))]
    return torch.utils.data.WeightedRandomSampler(w,num_samples=len(self),replacement=True)
SequenceDataset.get_dataset_sampler=_eqw
dev=torch.device("cuda")
T.train(c,device=dev)
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
env_meta=FileUtils.get_env_metadata_from_dataset(dataset_path=f"{PH}/can_ph_work.hdf5")
env=EnvUtils.create_env_from_metadata(env_meta=env_meta,render=False,render_offscreen=False,use_image_obs=False)
probe=np.load(f"{PH}/E_probe.npz")
sel_idx=np.concatenate([np.arange(b*25,b*25+(13 if b%2 else 12)) for b in range(4)])
env.reset()
scores={}
for ep in CKS:
    ck=glob.glob(f"{P}/out/{name}/*/models/model_epoch_{ep}.pth")
    if not ck: continue
    pol,_=FileUtils.policy_from_checkpoint(ckpt_path=ck[0],device=dev,verbose=False)
    ok=0
    for i in sel_idx:
        pol.start_episode(); ob=env.reset_to({"states":probe["states"][i]})
        for t in range(HORIZON):
            ob,_,_,_=env.step(pol(ob=ob))
            if env.is_success()["task"]: ok+=1; break
    scores[str(ep)]=ok/len(sel_idx)
mx=max(scores.values()) if scores else 0
plateau=min((int(e) for e,v in scores.items() if mx>0 and v>=0.9*mx), default=None)
out={"name":name,"config":cfg,"mask":mask,"seed":seed,"scores":scores,"plateau_ep":plateau,"max_score":mx}
json.dump(out,open(f"{P}/out/{name}/cv_eval.json","w"),indent=1)
print("CV_DONE",json.dumps(out["scores"]),"plateau",plateau,flush=True)
for ck in glob.glob(f"{P}/out/{name}/*/models/model_epoch_*.pth"): os.remove(ck)
