"""One variance-sweep run: same data, same eval, different POLICY CLASS.
Usage: run_var.py NAME CONFIG MASK SEED   CONFIG in {mlp, gmm, rnn, mlp160}
Estimand: run-to-run sd of J across seeds at fixed data (the program's binding constraint)."""
import glob, json, shutil, sys
import numpy as np, torch
from robomimic.config import config_factory
import robomimic.scripts.train as T
PH="/data/xinyua11/robomimic_runs/prescribe_ph"
P="/data/xinyua11/robomimic_runs/vardecomp"
name,cfg,mask,seed=sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4])
EP=300; HORIZON=500
shutil.rmtree(f"{P}/out/{name}",ignore_errors=True)
c=config_factory(algo_name="bc")
c.experiment.name=name; c.experiment.validate=False
c.experiment.logging.terminal_output_to_txt=False
c.experiment.save.enabled=True; c.experiment.save.every_n_epochs=EP
c.experiment.rollout.enabled=False
c.train.data="/data/xinyua11/robomimic_runs/prescribe_ph/can_ph_work.hdf5"; c.train.hdf5_filter_key=mask
c.train.output_dir=f"{P}/out"; c.train.num_epochs=EP; c.train.seed=seed
c.train.batch_size=100; c.train.hdf5_cache_mode="all"
c.observation.modalities.obs.low_dim=["robot0_eef_pos","robot0_eef_quat","robot0_gripper_qpos","object"]
c.observation.modalities.obs.rgb=[]
if cfg=="gmm":
    c.algo.gmm.enabled=True
elif cfg=="rnn":
    c.algo.rnn.enabled=True; c.train.seq_length=10; c.algo.actor_layer_dims=()
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
ck=glob.glob(f"{P}/out/{name}/*/models/model_epoch_{EP}*.pth"); assert len(ck)==1,ck
policy,_=FileUtils.policy_from_checkpoint(ckpt_path=ck[0],device=dev,verbose=False)
env_meta=FileUtils.get_env_metadata_from_dataset(dataset_path=f"{PH}/can_ph_work.hdf5")
env=EnvUtils.create_env_from_metadata(env_meta=env_meta,render=False,render_offscreen=False,use_image_obs=False)
q=json.load(open(f"{PH}/deploy_shares.json"))["natural_reset_shares"]
env.reset()
def ev(npz):
    d=np.load(f"{PH}/{npz}"); st,rg=d["states"],d["region"]; steps=[]
    for i in range(len(st)):
        policy.start_episode(); ob=env.reset_to({"states":st[i]}); sstep=-1
        for t in range(HORIZON):
            ob,_,_,_=env.step(policy(ob=ob))
            if env.is_success()["task"]: sstep=t+1; break
        steps.append(sstep)
    steps=np.array(steps)
    out={}
    for tag,H in (("h500",500),("h400",400)):
        su=(steps>0)&(steps<=H)
        Jr={r:float(su[rg==r].mean()) for r in sorted(set(rg.tolist()))}
        out[tag]={"J_uniform":float(su.mean()),"J_deploy":float(sum(q[r]*Jr[r] for r in Jr)),"J_region":Jr}
    out["success_step"]=steps.tolist()
    return out
out={"name":name,"config":cfg,"mask":mask,"seed":seed,"probe":ev("E_probe.npz"),"test":ev("E_test.npz")}
json.dump(out,open(f"{P}/out/{name}/var_eval.json","w"),indent=1)
print("VAR_EVAL",json.dumps({"cfg":cfg,"seed":seed,"probe":out["probe"]["h500"]["J_deploy"],"test":out["test"]["h500"]["J_deploy"]}),flush=True)
