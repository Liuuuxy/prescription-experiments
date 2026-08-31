import glob, json, shutil, sys, os
import numpy as np, torch
from robomimic.config import config_factory
import robomimic.scripts.train as T
P="/data/xinyua11/robomimic_runs/vardecomp2"
PH="/data/xinyua11/robomimic_runs/prescribe_ph"
name,cfg,mask,seed=sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4])
EP=500; HORIZON=500
shutil.rmtree(f"{P}/out/{name}",ignore_errors=True)
c=config_factory(algo_name="bc")
c.experiment.name=name; c.experiment.validate=False
c.experiment.logging.terminal_output_to_txt=False
c.experiment.save.enabled=True; c.experiment.save.every_n_epochs=50
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
q=json.load(open(f"{PH}/deploy_shares.json"))["natural_reset_shares"]
probe=np.load(f"{PH}/E_probe.npz")
# stratified 50-scene selection subset: first 12,13,12,13 of each 25-per-zone block
sel_idx=np.concatenate([np.arange(b*25,b*25+(13 if b%2 else 12)) for b in range(4)])
env.reset()
def run_states(policy,states,regs=None):
    steps=[]
    for st in states:
        policy.start_episode(); ob=env.reset_to({"states":st}); ss=-1
        for t in range(HORIZON):
            ob,_,_,_=env.step(policy(ob=ob))
            if env.is_success()["task"]: ss=t+1; break
        steps.append(ss)
    return np.array(steps)
sel={}
best_ep,best_score=None,-1
for ep in range(250,EP+1,50):
    ck=glob.glob(f"{P}/out/{name}/*/models/model_epoch_{ep}.pth")
    if not ck: continue
    pol,_=FileUtils.policy_from_checkpoint(ckpt_path=ck[0],device=dev,verbose=False)
    st=run_states(pol,probe["states"][sel_idx])
    sc=float(((st>0)&(st<=HORIZON)).mean())
    sel[str(ep)]=sc
    if sc>best_score: best_score, best_ep = sc, ep
ck=glob.glob(f"{P}/out/{name}/*/models/model_epoch_{best_ep}.pth")[0]
policy,_=FileUtils.policy_from_checkpoint(ckpt_path=ck,device=dev,verbose=False)
def ev(npz):
    d=np.load(f"{PH}/{npz}"); st=run_states(policy,d["states"]); rg=d["region"]
    out={}
    for tag,H in (("h500",500),("h400",400)):
        su=(st>0)&(st<=H)
        Jr={r:float(su[rg==r].mean()) for r in sorted(set(rg.tolist()))}
        out[tag]={"J_uniform":float(su.mean()),"J_deploy":float(sum(q[r]*Jr[r] for r in Jr)),"J_region":Jr}
    return out
out={"name":name,"config":cfg,"mask":mask,"seed":seed,"best_epoch":best_ep,
     "sel_scores":sel,"sel_best":best_score,"probe":ev("E_probe.npz"),"test":ev("E_test.npz")}
json.dump(out,open(f"{P}/out/{name}/var_eval.json","w"),indent=1)
print("VD2_DONE",json.dumps({"best_ep":best_ep,"sel":best_score,"test":out["test"]["h500"]["J_deploy"]}),flush=True)
# keep only best + final ckpt to save space
for ck in glob.glob(f"{P}/out/{name}/*/models/model_epoch_*.pth"):
    ep=int(ck.split("_epoch_")[1].split(".")[0])
    if ep not in (best_ep,EP): os.remove(ck)
